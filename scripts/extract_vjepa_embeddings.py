#!/usr/bin/env python3
"""Extract V-JEPA 2 embeddings from demonstration videos.

Processes videos using ResNet50 feature encoder to generate 1024-dimensional
embeddings for each frame. Embeddings are ready for downstream action classifier.

Usage:
    uv run python scripts/extract_vjepa_embeddings.py \\
        --video-dir data/raw/ \\
        --output-dir data/embeddings/ \\
        --device cuda
"""

import argparse
import sys
import json
from pathlib import Path

import numpy as np
import torch
import cv2
from tqdm import tqdm

from mimic.vision import VJepaEncoder


def extract_embeddings_from_video(
    video_path: str,
    encoder: VJepaEncoder,
    frame_stride: int = 1,
) -> tuple:
    """Extract embeddings from a video file.

    Args:
        video_path: Path to video file
        encoder: VJepaEncoder instance
        frame_stride: Extract every Nth frame

    Returns:
        (embeddings_array, frame_indices, metadata_dict)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        return None, None, None

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0

    print(f"\n  Video: {Path(video_path).name}")
    print(f"    Resolution: {width} × {height}")
    print(f"    FPS: {fps:.1f}, Duration: {duration:.1f}s, Total frames: {frame_count}")

    embeddings = []
    frame_indices = []
    frame_idx = 0

    # Process video with progress bar
    with tqdm(total=frame_count, desc="  Extracting embeddings", leave=False) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_stride == 0:
                # Extract embedding
                emb = encoder.extract_embedding(frame)
                if emb is not None:
                    embeddings.append(emb.numpy())
                    frame_indices.append(frame_idx)

            frame_idx += 1
            pbar.update(1)

    cap.release()

    if not embeddings:
        print(f"  WARNING: No embeddings extracted")
        return None, None, None

    # Stack into array
    embeddings_array = np.stack(embeddings)  # Shape: (num_frames, 1024)

    metadata = {
        "video_path": str(video_path),
        "fps": float(fps),
        "duration": float(duration),
        "frame_count": frame_count,
        "resolution": (width, height),
        "embeddings_shape": embeddings_array.shape,
        "frame_indices": frame_indices,
        "frame_stride": frame_stride,
        "encoding_timestamp": str(np.datetime64('now')),
    }

    print(f"  ✓ Extracted {len(embeddings)} embeddings")

    return embeddings_array, frame_indices, metadata


def save_embeddings(
    embeddings: np.ndarray,
    frame_indices: list,
    metadata: dict,
    output_path: str,
) -> bool:
    """Save embeddings and metadata to disk.

    Args:
        embeddings: (num_frames, 1024) array
        frame_indices: List of frame indices
        metadata: Dictionary with metadata
        output_path: Path to save (without extension)

    Returns:
        True if successful
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Save embeddings as numpy
        np.save(f"{output_path}.npy", embeddings)

        # Save metadata as JSON
        with open(f"{output_path}_meta.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  Saved to: {output_path}.npy")
        return True

    except Exception as e:
        print(f"  ERROR saving embeddings: {e}")
        return False


def main():
    """Main extraction pipeline."""
    parser = argparse.ArgumentParser(
        description="Extract V-JEPA embeddings from demonstration videos"
    )
    parser.add_argument(
        "--video-dir",
        default="data/raw/",
        help="Directory containing video files",
    )
    parser.add_argument(
        "--output-dir",
        default="data/embeddings/",
        help="Directory to save embeddings",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device: cuda or cpu",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Extract every Nth frame (default: 1 = all frames)",
    )
    parser.add_argument(
        "--model",
        default="timesformer",
        help="Model backend: vjepa2, timesformer, i3d",
    )

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.video_dir).exists():
        print(f"ERROR: Video directory not found: {args.video_dir}")
        return 1

    # Find videos
    video_extensions = [".mp4", ".mov", ".avi", ".mkv"]
    videos = [
        p for p in Path(args.video_dir).iterdir()
        if p.suffix.lower() in video_extensions
    ]

    if not videos:
        print(f"ERROR: No videos found in {args.video_dir}")
        return 1

    videos.sort()

    print("\n" + "=" * 60)
    print("V-JEPA EMBEDDING EXTRACTION")
    print("=" * 60)
    print(f"\nDevice: {args.device}")
    print(f"Model: {args.model}")
    print(f"Frame stride: {args.frame_stride}")
    print(f"Videos found: {len(videos)}")
    print(f"Output directory: {args.output_dir}\n")

    # Initialize encoder
    print("Loading encoder model...")
    encoder = VJepaEncoder(device=args.device, model_name=args.model)
    if encoder.model is None:
        print("ERROR: Failed to load encoder model")
        return 1

    # Process videos
    results = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for video_path in videos:
        embeddings, frame_indices, metadata = extract_embeddings_from_video(
            str(video_path),
            encoder,
            frame_stride=args.frame_stride,
        )

        if embeddings is not None:
            # Save embeddings
            output_path = output_dir / video_path.stem
            save_embeddings(embeddings, frame_indices, metadata, str(output_path))

            results.append({
                "status": "success",
                "video": video_path.name,
                "embeddings_shape": embeddings.shape,
                "output": str(output_path),
            })
        else:
            results.append({
                "status": "error",
                "video": video_path.name,
                "error": "No embeddings extracted",
            })

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "error"]

    print(f"\nSuccessful: {len(successful)}/{len(results)}")
    for r in successful:
        shape = r["embeddings_shape"]
        print(f"  ✓ {r['video']}: {shape[0]} frames × {shape[1]} dims")

    if failed:
        print(f"\nFailed: {len(failed)}/{len(results)}")
        for r in failed:
            print(f"  ✗ {r['video']}: {r.get('error', 'unknown error')}")

    # Save summary
    summary_path = output_dir / "extraction_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSummary saved to: {summary_path}")
    print("=" * 60)
    print("\nEmbeddings ready for action classification!")
    print(f"Next step: Use embeddings + audio labels to train classifier")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
