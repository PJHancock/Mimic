#!/usr/bin/env python3
"""Visualize a consolidated task input: video + resolved actions + tracking."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def load_task_input(task_input_file: str) -> dict:
    """Load and identify the canonical post-model artifact."""
    with open(task_input_file) as f:
        payload = json.load(f)
    if payload.get("schema") != "mimic.demo_task_input.v1":
        raise ValueError("Visualization requires mimic.demo_task_input.v1")
    return payload


def _action_segments(resolved_actions: list[dict]) -> list[dict]:
    """Derive display-only segments; they are not persisted robot inputs."""
    segments = []
    start = 0
    for index in range(1, len(resolved_actions) + 1):
        at_end = index == len(resolved_actions)
        if not at_end and resolved_actions[index]["phase"] == resolved_actions[start]["phase"]:
            continue
        frames = resolved_actions[start:index]
        confidences = [frame["confidence"] for frame in frames if frame["confidence"] is not None]
        segments.append(
            {
                "action": frames[0]["phase"],
                "start_time": frames[0]["timestamp_s"],
                "end_time": frames[-1]["timestamp_s"],
                "duration": frames[-1]["timestamp_s"] - frames[0]["timestamp_s"],
                "avg_confidence": float(np.mean(confidences)) if confidences else None,
            }
        )
        start = index
    return segments


def create_visualization(
    video_path: str,
    task_input_file: str,
    output_path: str,
):
    """Create side-by-side visualization of video + predictions.

    Creates two output videos:
    1. Original video with tracking overlay and action labels
    2. Side-by-side comparison of original and annotated versions
    """
    task_input = load_task_input(task_input_file)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create output directory
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup video writer for annotated version
    annotated_path = output_dir / "annotated.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_annotated = cv2.VideoWriter(str(annotated_path), fourcc, fps, (width, height))

    # Setup video writer for side-by-side version
    sidebyside_path = output_dir / "sidebyside.mp4"
    out_sidebyside = cv2.VideoWriter(str(sidebyside_path), fourcc, fps, (width * 2, height))

    actions_by_frame = {
        frame["frame_idx"]: frame for frame in task_input["resolved_actions"]
    }
    tracks_by_frame = {
        frame["frame_idx"]: frame for frame in task_input["object_tracks"]
    }
    action_segments = _action_segments(task_input["resolved_actions"])

    frame_idx = 0
    current_action = None

    print("\nCreating visualizations...")
    print(f"  Video: {width}x{height} @ {fps:.1f} FPS")
    print(f"  Output: {output_dir}")

    with tqdm(total=frame_count, desc="  Processing frames", leave=False) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Classifier and tracker streams may have different sampling density.
            # For display only, retain the latest resolved action while reading
            # the tracker observation at the exact source-video frame.
            source_frame_idx = frame_idx + 1
            accepted = actions_by_frame.get(source_frame_idx)
            if accepted is not None:
                current_action = accepted
            action = current_action["phase"] if current_action else "UNKNOWN"
            action_conf = current_action["confidence"] if current_action else None
            track = tracks_by_frame.get(source_frame_idx)
            position = track["position"] if track else None

            # Create annotated frame
            annotated = frame.copy()

            # Draw tracking circle if available
            if position and position["x"] is not None:
                x, y = int(position["x"]), int(position["y"])
                conf = position["confidence"]

                # Draw circle at tracked position
                color = (0, 255, 0) if conf > 0.5 else (0, 165, 255)
                cv2.circle(annotated, (x, y), 15, color, 2)
                cv2.circle(annotated, (x, y), 3, color, -1)

                # Draw confidence text
                conf_text = f"Track: {conf:.1%}"
                cv2.putText(
                    annotated,
                    conf_text,
                    (x + 20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

            # Draw action label and confidence
            action_color = (0, 255, 0) if action == "IDLE" else (0, 0, 255)
            confidence_text = "n/a" if action_conf is None else f"{action_conf:.1%}"
            action_text = f"{action} ({confidence_text})"

            # Background for text
            text_size = cv2.getTextSize(
                action_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                2,
            )[0]

            x_pos = 20
            y_pos = 50
            cv2.rectangle(
                annotated,
                (x_pos - 5, y_pos - text_size[1] - 5),
                (x_pos + text_size[0] + 5, y_pos + 5),
                (0, 0, 0),
                -1,
            )

            cv2.putText(
                annotated,
                action_text,
                (x_pos, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                action_color,
                2,
            )

            # Add timestamp
            timestamp = frame_idx / fps
            timestamp_text = f"T: {timestamp:.2f}s"
            cv2.putText(
                annotated,
                timestamp_text,
                (20, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            # Write annotated frame
            out_annotated.write(annotated)

            # Create side-by-side frame
            sidebyside = np.zeros((height, width * 2, 3), dtype=np.uint8)
            sidebyside[:, :width] = frame  # Original on left
            sidebyside[:, width:] = annotated  # Annotated on right

            # Add labels
            cv2.putText(
                sidebyside,
                "Original",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                sidebyside,
                "Annotated (Tracking + Action)",
                (width + 20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )

            # Write side-by-side frame
            out_sidebyside.write(sidebyside)

            frame_idx += 1
            pbar.update(1)

    cap.release()
    out_annotated.release()
    out_sidebyside.release()

    print(f"   ✓ Annotated video: {annotated_path}")
    print(f"   ✓ Side-by-side video: {sidebyside_path}")

    # Print summary
    print("\nAction Segments Detected:")
    for seg in action_segments:
        confidence = seg["avg_confidence"]
        confidence_text = "n/a" if confidence is None else f"{confidence:.1%}"
        print(
            f"  {seg['action']:10s} "
            f"{seg['start_time']:6.2f}s - {seg['end_time']:6.2f}s "
            f"(dur: {seg['duration']:5.2f}s, conf: {confidence_text})"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Visualize pipeline results with video annotations"
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to original video file",
    )
    parser.add_argument(
        "--task-input",
        type=str,
        required=True,
        help="Path to mimic.demo_task_input.v1 JSON from process_demo_video",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="viz_output/",
        help="Output directory for visualization videos",
    )

    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: Video file not found: {args.video}")
        return 1

    task_input_file = Path(args.task_input)
    if not task_input_file.exists():
        print(f"ERROR: Task input file not found: {args.task_input}")
        return 1

    print("\n" + "=" * 70)
    print("VISUALIZATION: VIDEO + PREDICTIONS")
    print("=" * 70)
    print(f"\nVideo: {args.video}")
    print(f"Task input: {args.task_input}")
    print(f"Output: {args.output}")

    try:
        create_visualization(str(video_path), str(task_input_file), args.output)
        print("\n" + "=" * 70)
        print("✓ Visualization complete!")
        print("=" * 70)
        return 0
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
