#!/usr/bin/env python3
"""Extract a single frame from a video for camera calibration.

Usage:
    uv run python scripts/extract_calibration_frame.py \\
        --video data/raw/IMG_2006.MOV \\
        --output data/raw/calibration_frame.png
"""

import argparse
import sys
from pathlib import Path

import cv2


def main():
    """Extract first frame from video."""
    parser = argparse.ArgumentParser(
        description="Extract first frame from video for calibration"
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to video file (MOV, MP4, AVI, etc.)",
    )
    parser.add_argument(
        "--output",
        default="data/raw/calibration_frame.png",
        help="Output path for calibration frame (PNG or JPG)",
    )
    parser.add_argument(
        "--frame-number",
        type=int,
        default=0,
        help="Frame number to extract (default: 0 = first frame)",
    )

    args = parser.parse_args()

    # Validate inputs
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found: {args.video}")
        return 1

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video: {args.video}")
        return 1

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\nVideo: {video_path.name}")
    print(f"  Resolution: {width} × {height}")
    print(f"  FPS: {fps}, Total frames: {frame_count}")

    # Validate frame number
    if args.frame_number >= frame_count:
        print(f"Error: Frame {args.frame_number} exceeds total frames ({frame_count})")
        cap.release()
        return 1

    # Seek to frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame_number)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Could not read frame {args.frame_number}")
        return 1

    # Save frame
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), frame)

    print(f"\n✓ Calibration frame saved to: {output_path}")
    print(f"\nNext step: Run calibration")
    print(f"  uv run python scripts/calibrate_camera.py \\")
    print(f"    --image {args.output} \\")
    print(f"    --width 0.508 \\")
    print(f"    --height 0.762 \\")
    print(f"    --output data/annotations/calibration.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
