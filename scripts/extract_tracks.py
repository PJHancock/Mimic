#!/usr/bin/env python3
"""Extract hand and object tracking data from demonstration videos.

Processes videos using calibration data to convert pixel coordinates to
normalized workspace coordinates. Outputs raw tracks and processed trajectory.

Usage:
    uv run python scripts/extract_tracks.py \\
        --calibration data/annotations/calibration.json \\
        --video-dir data/raw/ \\
        --output-dir data/tracks/
"""

import argparse
import json
import sys
from pathlib import Path

import cv2

from mimic.tracking import CoordinateMapper, CSRTObjectTracker, HandTracker, find_initial_bbox, process_trajectory


def extract_tracks_from_video(
    video_path: str,
    calibration_path: str,
    table_width_m: float,
    table_height_m: float,
    output_dir: str,
) -> dict:
    """Extract hand and object tracks from a single video.

    Args:
        video_path: Path to input video.
        calibration_path: Path to calibration JSON.
        table_width_m: Table width in meters.
        table_height_m: Table height in meters.
        output_dir: Directory to save outputs.

    Returns:
        Dictionary with extraction results and stats.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load calibration
    mapper = CoordinateMapper(table_width_m=table_width_m, table_height_m=table_height_m)
    try:
        mapper.load(str(calibration_path))
    except FileNotFoundError:
        return {
            "status": "error",
            "video": str(video_path),
            "error": f"Calibration file not found: {calibration_path}",
        }

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "status": "error",
            "video": str(video_path),
            "error": "Could not open video",
        }

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\nProcessing: {video_path.name}")
    print(f"  Resolution: {width} x {height}")
    print(f"  FPS: {fps}, Frames: {frame_count}")

    # Initialize trackers
    hand_tracker = HandTracker(min_detection_confidence=0.3)  # Lowered threshold to catch hands at odd angles
    object_tracker = CSRTObjectTracker()

    hand_tracks = []
    object_tracks = []
    frame_idx = 0

    # Read first frame to initialize object tracker
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    if not ret:
        return {
            "status": "error",
            "video": str(video_path),
            "error": "Could not read first frame",
        }

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect red object in first frame
    # Use config values; hardcoded for MVP (red Solo cup)
    hsv_lower = [0, 100, 100]
    hsv_upper = [10, 255, 255]
    hsv_lower_wrap = [170, 100, 100]
    hsv_upper_wrap = [180, 255, 255]
    min_contour_area = 500

    bbox = find_initial_bbox(
        frame_rgb,
        hsv_lower,
        hsv_upper,
        hsv_lower_wrap,
        hsv_upper_wrap,
        min_contour_area=min_contour_area,
    )

    if bbox is None:
        print("  WARNING: Red object not detected in frame 0")
        print("  Interactive bounding box selection...")
        # Fallback: manual ROI selection
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        try:
            bbox = cv2.selectROI("Select object bounding box", frame_bgr, fromCenter=False)
            cv2.destroyAllWindows()
            if bbox == (0, 0, 0, 0):
                cap.release()
                hand_tracker.release()
                return {
                    "status": "error",
                    "video": str(video_path),
                    "error": "No object bounding box selected",
                }
        except Exception as e:
            cap.release()
            hand_tracker.release()
            return {
                "status": "error",
                "video": str(video_path),
                "error": f"ROI selection failed: {e}",
            }

    print(f"  Object bbox: {bbox}")
    object_tracker.init(frame_rgb, bbox)

    # Process frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Hand tracking
        hand_track = hand_tracker.process(frame_rgb, frame_idx=frame_idx)
        if hand_track is not None:
            hand_tracks.append(hand_track)

        # Object tracking
        object_track = object_tracker.update(frame_rgb, frame_idx=frame_idx)
        if object_track is not None:
            object_tracks.append(object_track)

        frame_idx += 1

    cap.release()
    hand_tracker.release()

    print(f"  Detected hands: {len(hand_tracks)} frames")
    print(f"  Tracked object: {len(object_tracks)} frames")

    # Map object tracks to workspace coordinates
    mapped_tracks = []
    for track in object_tracks:
        workspace_coord = mapper.pixel_to_workspace(track.center_2d)
        # Create new track with workspace coordinates
        from mimic.common.types import ObjectTrack
        mapped_track = ObjectTrack(
            frame_idx=track.frame_idx,
            center_2d=workspace_coord,
            bbox=track.bbox,
            confidence=track.confidence,
        )
        mapped_tracks.append(mapped_track)

    # Process trajectory
    trajectory = process_trajectory(
        mapped_tracks,
        num_waypoints=30,
        smooth_window=5,
        smooth_order=2,
    )

    print(f"  Processed trajectory: {len(trajectory)} waypoints")

    # Save outputs
    video_stem = video_path.stem

    # Save hand tracks
    hand_tracks_path = output_dir / f"{video_stem}_hand_tracks.json"
    hand_tracks_data = [
        {
            "frame_idx": track.frame_idx,
            "wrist_2d": track.wrist_2d,
            "fingertips_2d": track.fingertips_2d,
            "finger_closure": track.finger_closure,
            "confidence": track.confidence,
        }
        for track in hand_tracks
    ]
    with open(hand_tracks_path, "w") as f:
        json.dump(hand_tracks_data, f, indent=2)

    # Save object tracks
    object_tracks_path = output_dir / f"{video_stem}_object_tracks.json"
    object_tracks_data = [
        {
            "frame_idx": track.frame_idx,
            "center_2d": track.center_2d,
            "bbox": track.bbox,
            "confidence": track.confidence,
        }
        for track in mapped_tracks
    ]
    with open(object_tracks_path, "w") as f:
        json.dump(object_tracks_data, f, indent=2)

    # Save trajectory
    trajectory_path = output_dir / f"{video_stem}_trajectory.json"
    trajectory_data = {
        "waypoints": trajectory,
        "num_waypoints": len(trajectory),
        "start": trajectory[0] if trajectory else None,
        "end": trajectory[-1] if trajectory else None,
    }
    with open(trajectory_path, "w") as f:
        json.dump(trajectory_data, f, indent=2)

    return {
        "status": "success",
        "video": str(video_path),
        "hand_tracks": len(hand_tracks),
        "object_tracks": len(object_tracks),
        "trajectory_waypoints": len(trajectory),
        "outputs": {
            "hand_tracks": str(hand_tracks_path),
            "object_tracks": str(object_tracks_path),
            "trajectory": str(trajectory_path),
        },
    }


def main():
    """Main extraction workflow."""
    parser = argparse.ArgumentParser(
        description="Extract hand and object tracking from demonstration videos"
    )
    parser.add_argument(
        "--calibration",
        required=True,
        help="Path to calibration JSON",
    )
    parser.add_argument(
        "--video-dir",
        default="data/raw/",
        help="Directory containing video files",
    )
    parser.add_argument(
        "--output-dir",
        default="data/tracks/",
        help="Directory to save tracking outputs",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=0.6,
        help="Table width in meters",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=0.4,
        help="Table height in meters",
    )

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.calibration).exists():
        print(f"Error: Calibration file not found: {args.calibration}")
        return 1

    if not Path(args.video_dir).exists():
        print(f"Error: Video directory not found: {args.video_dir}")
        return 1

    # Find videos
    video_extensions = [".mp4", ".mov", ".avi", ".mkv"]
    videos = [
        p for p in Path(args.video_dir).iterdir()
        if p.suffix.lower() in video_extensions
    ]

    if not videos:
        print(f"Error: No videos found in {args.video_dir}")
        return 1

    videos.sort()

    print("\n" + "=" * 60)
    print("TRACKING EXTRACTION")
    print("=" * 60)
    print(f"\nCalibration: {args.calibration}")
    print(f"Table dimensions: {args.width}m × {args.height}m")
    print(f"Videos found: {len(videos)}")
    print(f"Output directory: {args.output_dir}\n")

    results = []
    for video_path in videos:
        result = extract_tracks_from_video(
            str(video_path),
            args.calibration,
            args.width,
            args.height,
            args.output_dir,
        )
        results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "error"]

    print(f"\nSuccessful: {len(successful)}/{len(results)}")
    for r in successful:
        print(f"  ✓ {Path(r['video']).name}")
        print(f"    Hands: {r['hand_tracks']}, Object: {r['object_tracks']}, Trajectory: {r['trajectory_waypoints']}")

    if failed:
        print(f"\nFailed: {len(failed)}/{len(results)}")
        for r in failed:
            print(f"  ✗ {Path(r['video']).name}: {r['error']}")

    # Save summary
    summary_path = Path(args.output_dir) / "extraction_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSummary saved to: {summary_path}")
    print("=" * 60)

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
