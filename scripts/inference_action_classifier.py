#!/usr/bin/env python3
"""Run action inference on video embeddings using trained classifier.

Takes a video (or precomputed embeddings) and predicts frame-level actions.

Usage:
    # From embeddings file
    uv run python scripts/inference_action_classifier.py \\
        --embeddings data/embeddings/IMG_2006.npy \\
        --model models/action_classifier_lstm.pt

    # Or specify fps/duration for timestamps
    uv run python scripts/inference_action_classifier.py \\
        --embeddings data/embeddings/IMG_2006.npy \\
        --model models/action_classifier_lstm.pt \\
        --fps 30
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import json

from mimic.vision.action_classifier import ActionClassifier


# Action class names (must match training order)
ACTION_NAMES = ["IDLE", "APPROACH", "GRASP", "MOVE", "RELEASE"]


def run_inference(embeddings_path: str, model_path: str, fps: float = 30.0):
    """Run action classification on embeddings.

    Args:
        embeddings_path: Path to embeddings .npy file
        model_path: Path to trained model .pt file
        fps: Frames per second for timestamp calculation

    Returns:
        Dict with actions, confidences, timestamps
    """
    print("\n" + "=" * 70)
    print("ACTION CLASSIFIER INFERENCE")
    print("=" * 70)

    # Load embeddings
    print(f"\n1. Loading embeddings from: {embeddings_path}")
    embeddings_path = Path(embeddings_path)
    if not embeddings_path.exists():
        print(f"ERROR: Embeddings file not found: {embeddings_path}")
        return None

    embeddings = np.load(embeddings_path)
    num_frames = embeddings.shape[0]
    print(f"   Loaded: {embeddings.shape} (num_frames={num_frames}, dim=1024)")

    # Load model
    print(f"\n2. Loading trained model from: {model_path}")
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"ERROR: Model file not found: {model_path}")
        return None

    classifier = ActionClassifier(embedding_dim=1024, num_actions=5, model_type="lstm")
    classifier.load(str(model_path))
    print("   Model loaded successfully")

    # Run inference
    print(f"\n3. Running inference on {num_frames} frames...")
    actions, confidences = classifier.predict(embeddings)

    # Compute timestamps
    print(f"\n4. Computing timestamps (fps={fps})...")
    timestamps = np.arange(num_frames) / fps

    # Aggregate into contiguous segments
    segments = []
    current_action = actions[0]
    current_start = 0
    current_conf_sum = confidences[0]
    frame_count = 1

    for i in range(1, num_frames):
        if actions[i] == current_action:
            current_conf_sum += confidences[i]
            frame_count += 1
        else:
            # End of segment
            avg_confidence = current_conf_sum / frame_count
            segments.append({
                "action": ACTION_NAMES[current_action],
                "start_frame": int(current_start),
                "end_frame": int(i - 1),
                "start_time": float(timestamps[current_start]),
                "end_time": float(timestamps[i - 1]),
                "duration": float(timestamps[i - 1] - timestamps[current_start]),
                "avg_confidence": float(avg_confidence),
            })

            current_action = actions[i]
            current_start = i
            current_conf_sum = confidences[i]
            frame_count = 1

    # Add final segment
    avg_confidence = current_conf_sum / frame_count
    segments.append({
        "action": ACTION_NAMES[current_action],
        "start_frame": int(current_start),
        "end_frame": int(num_frames - 1),
        "start_time": float(timestamps[current_start]),
        "end_time": float(timestamps[-1]),
        "duration": float(timestamps[-1] - timestamps[current_start]),
        "avg_confidence": float(avg_confidence),
    })

    # Print results
    print(f"\n" + "=" * 70)
    print("PREDICTED ACTION SEGMENTS")
    print("=" * 70)
    print(f"\nTotal frames: {num_frames}")
    print(f"Video duration: {timestamps[-1]:.2f}s")
    print(f"Segments detected: {len(segments)}\n")

    for i, seg in enumerate(segments):
        print(f"Segment {i+1}: {seg['action']}")
        print(f"  Time: {seg['start_time']:.2f}s - {seg['end_time']:.2f}s ({seg['duration']:.2f}s)")
        print(f"  Frames: {seg['start_frame']} - {seg['end_frame']}")
        print(f"  Confidence: {seg['avg_confidence']:.1%}")
        print()

    # Print per-frame summary (first and last 10 frames)
    print("Per-frame predictions (first 10 frames):")
    for i in range(min(10, num_frames)):
        print(f"  Frame {i}: {ACTION_NAMES[actions[i]]} (conf: {confidences[i]:.1%})")

    if num_frames > 10:
        print(f"  ... ({num_frames - 20} frames omitted) ...")

    if num_frames > 20:
        print("Per-frame predictions (last 10 frames):")
        for i in range(max(0, num_frames - 10), num_frames):
            print(f"  Frame {i}: {ACTION_NAMES[actions[i]]} (conf: {confidences[i]:.1%})")

    print("\n" + "=" * 70)

    result = {
        "num_frames": int(num_frames),
        "duration": float(timestamps[-1]),
        "fps": float(fps),
        "per_frame": {
            "actions": [ACTION_NAMES[int(a)] for a in actions],
            "confidences": [float(c) for c in confidences],
            "timestamps": [float(t) for t in timestamps],
        },
        "segments": segments,
    }

    return result


def main():
    """Main inference script."""
    parser = argparse.ArgumentParser(
        description="Run action inference on video embeddings"
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        required=True,
        help="Path to embeddings .npy file",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model .pt file",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frames per second (for timestamps)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional: save results to JSON",
    )

    args = parser.parse_args()

    # Run inference
    result = run_inference(
        embeddings_path=args.embeddings,
        model_path=args.model,
        fps=args.fps,
    )

    if result is None:
        return 1

    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"✓ Results saved to: {output_path}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
