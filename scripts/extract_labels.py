#!/usr/bin/env python3
"""Extract action labels from all demonstration videos using audio narration.

Processes all videos in data/raw/ to extract frame-level action labels from audio
using Wav2Vec2 ASR and CTC decoding. Labels are saved as one-hot encoded matrices.

Usage:
    uv run python scripts/extract_labels.py

    Or with custom paths:
    uv run python scripts/extract_labels.py \\
        --video-dir data/raw/ \\
        --output-dir data/labels/
"""

import argparse
import sys
from pathlib import Path
from tqdm import tqdm

from mimic.data_pipeline.create_labels import process_pipeline


def main():
    """Extract labels from all videos."""
    parser = argparse.ArgumentParser(
        description="Extract action labels from all demonstration videos"
    )
    parser.add_argument(
        "--video-dir",
        default="data/raw/",
        help="Directory containing video files",
    )
    parser.add_argument(
        "--output-dir",
        default="data/labels/",
        help="Directory to save label files",
    )
    parser.add_argument(
        "--reaction-offset",
        type=float,
        default=-0.25,
        help="Motor reaction offset in seconds (default: -0.25)",
    )

    args = parser.parse_args()

    # Validate inputs
    video_dir = Path(args.video_dir)
    if not video_dir.exists():
        print(f"ERROR: Video directory not found: {args.video_dir}")
        return 1

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find videos
    video_extensions = [".mp4", ".mov", ".avi", ".mkv"]
    videos = sorted([
        p for p in video_dir.iterdir()
        if p.suffix.lower() in video_extensions
    ])

    if not videos:
        print(f"ERROR: No videos found in {args.video_dir}")
        return 1

    print("\n" + "=" * 70)
    print("LABEL EXTRACTION FROM VIDEO AUDIO")
    print("=" * 70)
    print(f"\nVideo directory: {args.video_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Reaction offset: {args.reaction_offset}s")
    print(f"Videos found: {len(videos)}\n")

    results = []

    for video_path in tqdm(videos, desc="Processing videos"):
        try:
            output_path = output_dir / f"{video_path.stem}.npy"

            print(f"\n  Processing: {video_path.name}")
            print(f"  Output: {output_path}")

            frame_matrix = process_pipeline(
                video_path=str(video_path),
                output_npy_path=str(output_path),
                reaction_offset_sec=args.reaction_offset,
            )

            results.append({
                "status": "success",
                "video": video_path.name,
                "output": str(output_path),
                "shape": frame_matrix.shape,
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "status": "error",
                "video": video_path.name,
                "error": str(e),
            })

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "error"]

    print(f"\nSuccessful: {len(successful)}/{len(results)}")
    for r in successful:
        print(f"  ✓ {r['video']}: {r['shape']}")

    if failed:
        print(f"\nFailed: {len(failed)}/{len(results)}")
        for r in failed:
            print(f"  ✗ {r['video']}: {r.get('error', 'unknown error')}")

    print("\n" + "=" * 70)
    if not failed:
        print("✓ All labels extracted successfully!")
        print("  Ready to train action classifier with embeddings + labels")
    else:
        print(f"⚠ {len(failed)} videos failed to process")
    print("=" * 70)

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
