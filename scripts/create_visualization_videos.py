#!/usr/bin/env python3
"""Create annotated and side-by-side visualization videos from inference results.

Usage:
    uv run python scripts/create_visualization_videos.py \\
        --video data/raw/IMG_2013.MOV \\
        --model models/action_classifier_Istm.pt \\
        --scores results/IMG_2013_scores.json \\
        --robot-actions results/IMG_2013_robot_actions.json \\
        --output results/IMG_2013_final/visualization/

Or use defaults:
    uv run python scripts/create_visualization_videos.py --video data/raw/IMG_2013.MOV
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

from mimic.integration import load_skill_system, checkpoint_sha256, load_robot_actions
from mimic.vision import VJepaEncoder
from mimic.vision.action_classifier import ActionClassifier
from mimic.common.types import ActionPhase


def extract_embeddings_and_predictions(
    video_path: str,
    model_path: str,
    skill_config_path: str,
    device: str = "cpu",
    context_window: int = 32,
):
    """Extract embeddings from video and predict actions."""
    print("\n1. Extracting embeddings...")

    encoder = VJepaEncoder(device=device, model_name="timesformer")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    embeddings = []
    embedding_frame_indices = []
    frame_idx = 0

    with tqdm(total=frame_count, desc="   Extracting embeddings", leave=False) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            emb = encoder.extract_embedding(frame)
            if emb is not None:
                embeddings.append(emb.numpy())
                embedding_frame_indices.append(frame_idx + 1)

            frame_idx += 1
            pbar.update(1)

    cap.release()
    embeddings_array = np.stack(embeddings)
    print(f"   ✓ Extracted {embeddings_array.shape[0]} embeddings ({embeddings_array.shape[1]}D)")

    print("\n2. Predicting actions...")
    skill_system = load_skill_system(skill_config_path)
    classifier = ActionClassifier(
        embedding_dim=1024,
        num_actions=skill_system.catalog.class_count,
        model_type="lstm",
    )

    try:
        classifier.load(model_path, catalog=skill_system.catalog)
    except ValueError as e:
        if "Legacy checkpoint" in str(e):
            print(f"   ⚠ Using legacy model checkpoint")
            classifier.load(model_path, catalog=None)
        else:
            raise

    probabilities = classifier.predict_probabilities(embeddings_array, context_window=context_window)
    print(f"   ✓ Predicted scores for {len(probabilities)} frames")

    return fps, frame_count, probabilities, embedding_frame_indices, skill_system.catalog


def _phase_to_rgb(phase: ActionPhase) -> tuple:
    """Map action phase to RGB color."""
    color_map = {
        ActionPhase.APPROACH: (0, 165, 255),      # Orange
        ActionPhase.GRASP: (0, 255, 0),           # Green
        ActionPhase.RETRACT: (255, 0, 0),         # Blue
        ActionPhase.PLACE: (0, 255, 255),         # Cyan
        ActionPhase.RELEASE: (255, 255, 0),       # Yellow
        ActionPhase.RETREAT: (255, 0, 255),       # Magenta
    }
    return color_map.get(phase, (128, 128, 128))


def _draw_action_text(frame: np.ndarray, action: str, confidence: Optional[float], y_offset: int = 50):
    """Draw action text on frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.0
    thickness = 3
    color = (0, 255, 0)

    if confidence is not None:
        text = f"{action} ({confidence:.2%})"
    else:
        text = action

    cv2.putText(frame, text, (20, y_offset), font, font_scale, color, thickness)
    return frame


def _draw_tracking_marker(frame: np.ndarray, x: float, y: float, radius: int = 15):
    """Draw a circle marker for object detection."""
    if x is not None and y is not None:
        # Convert to integer coordinates
        cx, cy = int(x), int(y)
        # Draw circle outline
        cv2.circle(frame, (cx, cy), radius, (0, 255, 255), 2)
        # Draw crosshair
        cv2.line(frame, (cx - 8, cy), (cx + 8, cy), (0, 255, 255), 2)
        cv2.line(frame, (cx, cy - 8), (cx, cy + 8), (0, 255, 255), 2)
    return frame


def create_annotated_video(
    input_video: str,
    output_path: Path,
    fps: float,
    frame_count: int,
    action_predictions: list,
    tracking_data: dict,
    skill_catalog,
):
    """Create video with action annotations and tracking markers overlaid."""
    print(f"\n3. Creating annotated video: {output_path.name}")

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_video}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))

    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video writer: {output_path}")

    # Create maps for quick lookup
    pred_map = {pred["frame_idx"]: pred for pred in action_predictions}
    track_map = {track["frame"]: track for track in tracking_data.get("positions", [])}

    frame_idx = 1
    with tqdm(total=frame_count, desc="   Writing annotated frames", leave=False) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Get prediction for this frame
            if frame_idx in pred_map:
                pred = pred_map[frame_idx]
                action_phase = ActionPhase(pred["phase"])
                confidence = pred["confidence"]
                _draw_action_text(frame, action_phase.value, confidence)

            # Draw tracking marker if available
            if frame_idx - 1 in track_map:  # Frame numbers are 0-indexed in tracks
                track = track_map[frame_idx - 1]
                if track.get("x") is not None and track.get("y") is not None:
                    _draw_tracking_marker(frame, track["x"], track["y"])

            writer.write(frame)
            frame_idx += 1
            pbar.update(1)

    cap.release()
    writer.release()
    print(f"   ✓ Annotated video saved: {output_path}")


def create_sidebyside_video(
    input_video: str,
    annotated_video: str,
    output_path: Path,
    fps: float,
    frame_count: int,
):
    """Create side-by-side comparison video."""
    print(f"\n4. Creating side-by-side video: {output_path.name}")

    cap_original = cv2.VideoCapture(input_video)
    cap_annotated = cv2.VideoCapture(annotated_video)

    if not cap_original.isOpened() or not cap_annotated.isOpened():
        raise FileNotFoundError("Cannot open input videos")

    frame_width = int(cap_original.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap_original.get(cv2.CAP_PROP_FRAME_HEIGHT))
    combined_width = frame_width * 2
    combined_height = frame_height

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (combined_width, combined_height))

    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video writer: {output_path}")

    with tqdm(total=frame_count, desc="   Writing side-by-side frames", leave=False) as pbar:
        while True:
            ret_orig, frame_orig = cap_original.read()
            ret_annot, frame_annot = cap_annotated.read()

            if not (ret_orig and ret_annot):
                break

            combined = np.hstack([frame_orig, frame_annot])
            writer.write(combined)
            pbar.update(1)

    cap_original.release()
    cap_annotated.release()
    writer.release()
    print(f"   ✓ Side-by-side video saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=str, required=True, help="Input video file")
    parser.add_argument(
        "--model",
        type=str,
        default="models/action_classifier_lstm.pt",
        help="Classifier model path",
    )
    parser.add_argument(
        "--skill-config",
        type=str,
        default="configs/skills/pick_place.yaml",
        help="Skill configuration YAML",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for videos",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "mps"),
        default="cpu",
        help="Torch device",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=32,
        help="LSTM context window size",
    )

    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: Video not found: {args.video}")
        return 1

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: Model not found: {args.model}")
        return 1

    # Determine output directory
    if args.output is None:
        output_dir = Path("results") / video_path.stem / "visualization"
    else:
        output_dir = Path(args.output)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("VIDEO VISUALIZATION PIPELINE")
    print("=" * 70)
    print(f"\nVideo: {args.video}")
    print(f"Model: {args.model}")
    print(f"Output: {output_dir}")

    try:
        # Extract embeddings and predict actions
        fps, frame_count, probabilities, embedding_frame_indices, catalog = \
            extract_embeddings_and_predictions(
                str(video_path),
                str(model_path),
                args.skill_config,
                device=args.device,
                context_window=args.context_window,
            )

        # Extract object tracking data
        print("\n1.5. Extracting object position tracks...")
        from mimic.tracking import find_initial_bbox

        hsv_lower = [0, 100, 100]
        hsv_upper = [10, 255, 255]
        hsv_lower_wrap = [170, 100, 100]
        hsv_upper_wrap = [180, 255, 255]
        min_contour_area = 500

        cap = cv2.VideoCapture(str(video_path))
        frame_count_actual = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        tracks = []
        frame_idx = 0
        with tqdm(total=frame_count_actual, desc="   Extracting tracks", leave=False) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                bbox = find_initial_bbox(
                    frame_rgb,
                    hsv_lower,
                    hsv_upper,
                    hsv_lower_wrap,
                    hsv_upper_wrap,
                    min_contour_area,
                )

                if bbox:
                    x, y, w, h = bbox
                    center_x = x + w / 2
                    center_y = y + h / 2
                    tracks.append({
                        "frame": frame_idx,
                        "x": float(center_x),
                        "y": float(center_y),
                    })
                else:
                    tracks.append({
                        "frame": frame_idx,
                        "x": None,
                        "y": None,
                    })

                frame_idx += 1
                pbar.update(1)

        cap.release()
        tracking_data = {"positions": tracks}
        print(f"   ✓ Extracted {len(tracks)} position samples")

        # Convert probabilities to action predictions
        action_predictions = []
        for i, frame_idx in enumerate(embedding_frame_indices):
            predicted_action_idx = np.argmax(probabilities[i])
            action_phase = ActionPhase(catalog.labels[predicted_action_idx])
            confidence = float(probabilities[i, predicted_action_idx])

            action_predictions.append({
                "frame_idx": frame_idx,
                "phase": action_phase.value,
                "confidence": confidence,
            })

        # Create annotated video
        annotated_path = output_dir / "annotated.mp4"
        create_annotated_video(
            str(video_path),
            annotated_path,
            fps,
            frame_count_actual,
            action_predictions,
            tracking_data,
            catalog,
        )

        # Create side-by-side video
        sidebyside_path = output_dir / "sidebyside.mp4"
        create_sidebyside_video(
            str(video_path),
            str(annotated_path),
            sidebyside_path,
            fps,
            frame_count_actual,
        )

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"\nAnnotated video: {annotated_path}")
        print(f"Side-by-side video: {sidebyside_path}")
        print(f"Total frames: {frame_count_actual}")
        print(f"FPS: {fps:.1f}")

        return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
