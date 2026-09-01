#!/usr/bin/env python3
"""End-to-end demo video processing: tracks + embeddings + actions.

Processes a new demo video through the complete pipeline:
  1. Extract object position tracks (from tracking module)
  2. Extract V-JEPA embeddings from frames
  3. Predict action sequences from embeddings
  4. Combine into unified output with timestamps

Usage:
    uv run python scripts/process_demo_video.py \\
        --video new_demo.mov \\
        --output results/new_demo/

    Or use trained model:
    uv run python scripts/process_demo_video.py \\
        --video new_demo.mov \\
        --model models/action_classifier_lstm.pt \\
        --output results/new_demo/
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2
from tqdm import tqdm

from mimic.vision import VJepaEncoder
from mimic.vision.action_classifier import ActionClassifier
from mimic.tracking import ObjectTracker

# Action class names
ACTION_NAMES = ["IDLE", "APPROACH", "GRASP", "MOVE", "RELEASE"]


def extract_tracks(video_path: str, device: str = "cpu"):
    """Extract object position tracks from video.

    Returns:
        Dict with frame-by-frame (x, y, confidence) positions
    """
    print("\n1. Extracting object position tracks...")

    tracker = ObjectTracker(device=device)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0

    print(f"   Video: {Path(video_path).name}")
    print(f"   Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))} × {int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"   FPS: {fps:.1f}, Duration: {duration:.1f}s, Frames: {frame_count}")

    tracks = []
    frame_idx = 0

    with tqdm(total=frame_count, desc="   Extracting tracks", leave=False) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Track object
            position = tracker.track(frame)

            if position is not None:
                x, y, conf = position
                tracks.append({
                    "frame": int(frame_idx),
                    "time": float(frame_idx / fps),
                    "x": float(x),
                    "y": float(y),
                    "confidence": float(conf),
                })
            else:
                tracks.append({
                    "frame": int(frame_idx),
                    "time": float(frame_idx / fps),
                    "x": None,
                    "y": None,
                    "confidence": 0.0,
                })

            frame_idx += 1
            pbar.update(1)

    cap.release()

    print(f"   ✓ Extracted {len(tracks)} position samples")
    return {
        "fps": float(fps),
        "duration": float(duration),
        "frame_count": frame_count,
        "positions": tracks,
    }


def extract_embeddings(video_path: str, device: str = "cpu"):
    """Extract V-JEPA embeddings from video.

    Returns:
        (embeddings array, fps, frame_count)
    """
    print("\n2. Extracting V-JEPA embeddings...")

    encoder = VJepaEncoder(device=device, model_name="timesformer")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    embeddings = []
    frame_idx = 0

    with tqdm(total=frame_count, desc="   Extracting embeddings", leave=False) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            emb = encoder.extract_embedding(frame)
            if emb is not None:
                embeddings.append(emb.numpy())

            frame_idx += 1
            pbar.update(1)

    cap.release()

    embeddings_array = np.stack(embeddings)
    print(f"   ✓ Extracted {embeddings_array.shape[0]} embeddings ({embeddings_array.shape[1]}D)")

    return embeddings_array, fps, frame_count


def predict_actions(embeddings: np.ndarray, model_path: str, fps: float):
    """Predict action sequence from embeddings.

    Returns:
        Dict with per-frame and segment predictions
    """
    print("\n3. Predicting action sequences...")

    classifier = ActionClassifier(embedding_dim=1024, num_actions=5, model_type="lstm")
    classifier.load(model_path)
    print(f"   ✓ Model loaded from: {model_path}")

    actions, confidences = classifier.predict(embeddings)
    print(f"   ✓ Predicted actions for {len(actions)} frames")

    # Compute timestamps
    timestamps = np.arange(len(actions)) / fps

    # Aggregate into segments
    segments = []
    current_action = actions[0]
    current_start = 0
    current_conf_sum = confidences[0]
    frame_count = 1

    for i in range(1, len(actions)):
        if actions[i] == current_action:
            current_conf_sum += confidences[i]
            frame_count += 1
        else:
            avg_conf = current_conf_sum / frame_count
            segments.append({
                "action": ACTION_NAMES[current_action],
                "start_frame": int(current_start),
                "end_frame": int(i - 1),
                "start_time": float(timestamps[current_start]),
                "end_time": float(timestamps[i - 1]),
                "duration": float(timestamps[i - 1] - timestamps[current_start]),
                "avg_confidence": float(avg_conf),
            })

            current_action = actions[i]
            current_start = i
            current_conf_sum = confidences[i]
            frame_count = 1

    # Add final segment
    avg_conf = current_conf_sum / frame_count
    segments.append({
        "action": ACTION_NAMES[current_action],
        "start_frame": int(current_start),
        "end_frame": int(len(actions) - 1),
        "start_time": float(timestamps[current_start]),
        "end_time": float(timestamps[-1]),
        "duration": float(timestamps[-1] - timestamps[current_start]),
        "avg_confidence": float(avg_conf),
    })

    print(f"   ✓ Detected {len(segments)} action segments")

    return {
        "fps": float(fps),
        "num_frames": len(actions),
        "per_frame": {
            "actions": [ACTION_NAMES[int(a)] for a in actions],
            "confidences": [float(c) for c in confidences],
            "timestamps": [float(t) for t in timestamps],
        },
        "segments": segments,
    }


def combine_results(tracks_data: dict, actions_data: dict) -> dict:
    """Combine tracking and action predictions.

    Returns:
        Unified result with synchronized data
    """
    print("\n4. Combining tracks and actions...")

    num_frames = len(actions_data["per_frame"]["actions"])
    fps = actions_data["fps"]
    timestamps = np.array(actions_data["per_frame"]["timestamps"])

    combined_frames = []
    for i in range(num_frames):
        # Get position at this frame
        position = tracks_data["positions"][i] if i < len(tracks_data["positions"]) else None

        # Get action at this frame
        action = actions_data["per_frame"]["actions"][i]
        action_conf = actions_data["per_frame"]["confidences"][i]

        frame_data = {
            "frame_index": int(i),
            "timestamp": float(timestamps[i]),
            "position": {
                "x": position["x"] if position else None,
                "y": position["y"] if position else None,
                "confidence": position["confidence"] if position else 0.0,
            } if position else None,
            "action": action,
            "action_confidence": float(action_conf),
        }
        combined_frames.append(frame_data)

    print(f"   ✓ Combined {num_frames} frames")

    return {
        "metadata": {
            "created": datetime.now().isoformat(),
            "fps": float(fps),
            "total_frames": num_frames,
            "duration": float(timestamps[-1]) if num_frames > 0 else 0,
        },
        "per_frame": combined_frames,
        "action_segments": actions_data["segments"],
        "tracking_summary": {
            "total_positions": sum(1 for p in tracks_data["positions"] if p["x"] is not None),
            "fps": tracks_data["fps"],
        },
    }


def main():
    """Process video through full pipeline."""
    parser = argparse.ArgumentParser(
        description="Process demo video through full pipeline"
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/action_classifier_lstm.pt",
        help="Path to trained action classifier model",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/",
        help="Output directory for results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device: cpu or cuda",
    )

    args = parser.parse_args()

    # Validate inputs
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: Video file not found: {args.video}")
        return 1

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: Model file not found: {args.model}")
        return 1

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("DEMO VIDEO PROCESSING PIPELINE")
    print("=" * 70)
    print(f"\nVideo: {args.video}")
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print(f"Device: {args.device}")

    try:
        # Run pipeline
        tracks_data = extract_tracks(str(video_path), device=args.device)
        embeddings, fps, frame_count = extract_embeddings(str(video_path), device=args.device)
        actions_data = predict_actions(embeddings, str(model_path), fps)
        results = combine_results(tracks_data, actions_data)

        # Save results
        print("\n5. Saving results...")
        results_file = output_dir / f"{video_path.stem}_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"   ✓ Saved to: {results_file}")

        # Print summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"\nTotal frames: {results['metadata']['total_frames']}")
        print(f"Duration: {results['metadata']['duration']:.2f}s")
        print(f"FPS: {results['metadata']['fps']:.1f}")
        print(f"\nAction segments: {len(results['action_segments'])}")
        for seg in results['action_segments']:
            print(f"  {seg['action']:10s} {seg['start_time']:6.2f}s - {seg['end_time']:6.2f}s (conf: {seg['avg_confidence']:.1%})")

        print(f"\nTracked positions: {results['tracking_summary']['total_positions']}/{results['metadata']['total_frames']}")
        print("\n" + "=" * 70)

        return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
