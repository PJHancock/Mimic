#!/usr/bin/env python3
"""Run action inference on video embeddings using trained classifier.

Takes a video (or precomputed embeddings) and predicts frame-level actions.

Usage:
    # From embeddings file
    uv run python scripts/inference_action_classifier.py \\
        --embeddings data/embeddings/IMG_2006.npy \\
        --model models/action_classifier_lstm.pt \\
        --fps 30

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

from mimic.integration import (
    build_action_inference_artifacts,
    checkpoint_sha256,
    load_skill_system,
    predictions_from_probabilities,
    write_results,
)
from mimic.vision.action_classifier import ActionClassifier


def run_inference(
    embeddings_path: str,
    model_path: str,
    fps: float,
    skill_config_path: str = "configs/skills/pick_place.yaml",
):
    """Run action classification on embeddings.

    Args:
        embeddings_path: Path to embeddings .npy file
        model_path: Path to trained model .pt file
        fps: Frames per second for timestamp calculation

    Returns:
        Validated diagnostic-score and single-state robot artifacts.
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

    skill_system = load_skill_system(skill_config_path)
    classifier = ActionClassifier(
        embedding_dim=1024,
        num_actions=skill_system.catalog.class_count,
        model_type="lstm",
    )
    classifier.load(str(model_path), catalog=skill_system.catalog)
    print("   Model loaded successfully")

    # Run inference
    print(f"\n3. Running inference on {num_frames} frames...")
    probabilities = classifier.predict_probabilities(embeddings)

    # Compute timestamps
    print(f"\n4. Computing timestamps (fps={fps})...")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    timestamps = np.arange(num_frames) / fps
    predictions = predictions_from_probabilities(
        probabilities,
        timestamps,
        skill_system.catalog,
    )
    artifacts = build_action_inference_artifacts(
        predictions,
        skill_system,
        checkpoint_sha256(model_path),
    )

    # Aggregate the accepted robot states for display only.
    robot_frames = artifacts.robot_actions.frames
    segments = []
    current_action = robot_frames[0].phase
    current_start = 0
    segment_confidences = [robot_frames[0].confidence]

    for i in range(1, num_frames):
        frame = robot_frames[i]
        if frame.phase == current_action:
            segment_confidences.append(frame.confidence)
        else:
            available = [value for value in segment_confidences if value is not None]
            segments.append(
                {
                    "action": current_action.value,
                    "start_frame": robot_frames[current_start].frame_idx,
                    "end_frame": robot_frames[i - 1].frame_idx,
                    "start_time": float(timestamps[current_start]),
                    "end_time": float(timestamps[i - 1]),
                    "duration": float(timestamps[i - 1] - timestamps[current_start]),
                    "avg_confidence": float(np.mean(available)) if available else None,
                }
            )

            current_action = frame.phase
            current_start = i
            segment_confidences = [frame.confidence]

    # Add final segment
    available = [value for value in segment_confidences if value is not None]
    segments.append(
        {
            "action": current_action.value,
            "start_frame": robot_frames[current_start].frame_idx,
            "end_frame": robot_frames[-1].frame_idx,
            "start_time": float(timestamps[current_start]),
            "end_time": float(timestamps[-1]),
            "duration": float(timestamps[-1] - timestamps[current_start]),
            "avg_confidence": float(np.mean(available)) if available else None,
        }
    )

    # Print results
    print("\n" + "=" * 70)
    print("PREDICTED ACTION SEGMENTS")
    print("=" * 70)
    print(f"\nTotal frames: {num_frames}")
    print(f"Video duration: {timestamps[-1]:.2f}s")
    print(f"Segments detected: {len(segments)}\n")

    for i, seg in enumerate(segments):
        print(f"Segment {i+1}: {seg['action']}")
        print(f"  Time: {seg['start_time']:.2f}s - {seg['end_time']:.2f}s ({seg['duration']:.2f}s)")
        print(f"  Frames: {seg['start_frame']} - {seg['end_frame']}")
        confidence = seg["avg_confidence"]
        print(f"  Confidence: {confidence:.1%}" if confidence is not None else "  Confidence: n/a")
        print()

    # Print per-frame summary (first and last 10 frames)
    print("Per-frame predictions (first 10 frames):")
    for i in range(min(10, num_frames)):
        frame = robot_frames[i]
        confidence = "n/a" if frame.confidence is None else f"{frame.confidence:.1%}"
        print(f"  Frame {frame.frame_idx}: {frame.phase.value} (conf: {confidence})")

    if num_frames > 10:
        print(f"  ... ({num_frames - 20} frames omitted) ...")

    if num_frames > 20:
        print("Per-frame predictions (last 10 frames):")
        for i in range(max(0, num_frames - 10), num_frames):
            frame = robot_frames[i]
            confidence = "n/a" if frame.confidence is None else f"{frame.confidence:.1%}"
            print(f"  Frame {frame.frame_idx}: {frame.phase.value} (conf: {confidence})")

    print("\n" + "=" * 70)

    return artifacts


def main():
    """Main inference script."""
    parser = argparse.ArgumentParser(description="Run action inference on video embeddings")
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
        required=True,
        help="Source-video FPS used for timestamps; never inferred",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional processed robot-action JSON output",
    )
    parser.add_argument(
        "--scores-output",
        type=str,
        default=None,
        help="Optional diagnostic full-score JSON; defaults beside --output",
    )
    parser.add_argument(
        "--skill-config",
        type=str,
        default="configs/skills/pick_place.yaml",
        help="Skill catalog, graph, and explicit post-state settings YAML",
    )

    args = parser.parse_args()

    # Run inference
    try:
        result = run_inference(
            embeddings_path=args.embeddings,
            model_path=args.model,
            fps=args.fps,
            skill_config_path=args.skill_config,
        )
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    if result is None:
        return 1

    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        write_results(output_path, result.robot_actions)
        print(f"✓ Robot actions saved to: {output_path}")
        scores_path = (
            Path(args.scores_output)
            if args.scores_output
            else output_path.with_name(f"{output_path.stem}_scores{output_path.suffix}")
        )
        write_results(scores_path, result.scores)
        print(f"✓ Diagnostic scores saved to: {scores_path}\n")
    elif args.scores_output:
        write_results(args.scores_output, result.scores)
        print(f"✓ Diagnostic scores saved to: {args.scores_output}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
