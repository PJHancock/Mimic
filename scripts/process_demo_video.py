#!/usr/bin/env python3
"""End-to-end demo video processing: tracks + embeddings + actions + robot simulation.

Processes a new demo video through the complete pipeline:
  1. Extract object position tracks (from tracking module)
  2. Extract V-JEPA embeddings from frames
  3. Predict action sequences from embeddings
  4. Build one consolidated task input with independent action/tracker streams
  5. Generate robot waypoints from predictions
  6. Run robot simulation with inferred waypoints

Usage:
    uv run python scripts/process_demo_video.py \\
        --video new_demo.mov \\
        --output results/new_demo/

    Or use trained model:
    uv run python scripts/process_demo_video.py \\
        --video new_demo.mov \\
        --model models/action_classifier_lstm.pt \\
        --output results/new_demo/

    With robot simulation:
    uv run python scripts/process_demo_video.py \\
        --video new_demo.mov \\
        --model models/action_classifier_lstm.pt \\
        --config path/to/experiment_robot.yaml \\
        --calibration data/annotations/calibrations.json \\
        --robot-pipeline-config path/to/experiment_robot_pipeline.yaml \\
        --output results/new_demo/ \\
        --simulate-robot

    uv run python scripts/process_demo_video.py \
    --video path/to/input_video.mov \
    --model models/action_classifier_lstm.pt \
    --skill-config path/to/experiment_skill.yaml \
    --output results/my_run/
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from tqdm import tqdm

from mimic.common.types import PickPlaceWaypoints
from mimic.integration import (
    RobotActionResults,
    build_demo_task_input,
    build_robot_pipeline,
    build_action_inference_artifacts,
    checkpoint_sha256,
    load_skill_system,
    predictions_from_probabilities,
    write_world_waypoints,
    write_results,
)
from mimic.vision import VJepaEncoder
from mimic.vision.action_classifier import ActionClassifier
from mimic.tracking import find_initial_bbox


def extract_tracks(video_path: str, device: str = "cpu"):
    """Extract object position tracks from video using color-based detection.

    Returns:
        Dict with frame-by-frame (x, y, confidence) positions
    """
    print("\n1. Extracting object position tracks...")

    hsv_lower = [0, 100, 100]
    hsv_upper = [10, 255, 255]
    hsv_lower_wrap = [170, 100, 100]
    hsv_upper_wrap = [180, 255, 255]
    min_contour_area = 500

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width_px = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height_px = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0

    print(f"   Video: {Path(video_path).name}")
    print(f"   Resolution: {frame_width_px} × {frame_height_px}")
    print(f"   FPS: {fps:.1f}, Duration: {duration:.1f}s, Frames: {frame_count}")

    tracks = []
    frame_idx = 0

    with tqdm(total=frame_count, desc="   Extracting tracks", leave=False) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Color-based detection for each frame
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
                confidence = 0.8  # Placeholder confidence
                tracks.append(
                    {
                        "frame": int(frame_idx),
                        "time": float(frame_idx / fps),
                        "x": float(center_x),
                        "y": float(center_y),
                        "confidence": float(confidence),
                    }
                )
            else:
                tracks.append(
                    {
                        "frame": int(frame_idx),
                        "time": float(frame_idx / fps),
                        "x": None,
                        "y": None,
                        "confidence": 0.0,
                    }
                )

            frame_idx += 1
            pbar.update(1)

    cap.release()

    print(f"   ✓ Extracted {len(tracks)} position samples")
    return {
        "fps": float(fps),
        "duration": float(duration),
        "frame_count": frame_count,
        "frame_width_px": frame_width_px,
        "frame_height_px": frame_height_px,
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

    return embeddings_array, fps, frame_count, tuple(embedding_frame_indices)


def _accepted_action_segments(actions: RobotActionResults) -> list[dict]:
    """Collapse adjacent accepted states for reporting, never for robot input."""
    frames = actions.frames
    segments = []
    current_phase = frames[0].phase
    start = 0
    confidences = [frames[0].confidence]
    for index, frame in enumerate(frames[1:], start=1):
        if frame.phase == current_phase:
            confidences.append(frame.confidence)
            continue
        available = [value for value in confidences if value is not None]
        segments.append(
            {
                "action": current_phase.value,
                "start_frame": frames[start].frame_idx,
                "end_frame": frames[index - 1].frame_idx,
                "start_time": frames[start].timestamp_s,
                "end_time": frames[index - 1].timestamp_s,
                "duration": frames[index - 1].timestamp_s - frames[start].timestamp_s,
                "avg_confidence": float(np.mean(available)) if available else None,
            }
        )
        current_phase = frame.phase
        start = index
        confidences = [frame.confidence]
    available = [value for value in confidences if value is not None]
    segments.append(
        {
            "action": current_phase.value,
            "start_frame": frames[start].frame_idx,
            "end_frame": frames[-1].frame_idx,
            "start_time": frames[start].timestamp_s,
            "end_time": frames[-1].timestamp_s,
            "duration": frames[-1].timestamp_s - frames[start].timestamp_s,
            "avg_confidence": float(np.mean(available)) if available else None,
        }
    )
    return segments


def predict_actions(
    embeddings: np.ndarray,
    model_path: str,
    fps: float,
    skill_config_path: str,
    frame_indices: Sequence[int],
    context_window: int = 32,
):
    """Predict action sequence from embeddings.

    Args:
        embeddings: Array of embeddings to classify
        model_path: Path to trained model
        fps: Frames per second for timing
        skill_config_path: Path to skill system config
        frame_indices: Frame indices for embeddings
        context_window: Context window size for LSTM (±N frames, default 32)

    Returns:
        Validated score and post-processed single-state artifacts.
    """
    print("\n3. Predicting action sequences...")

    skill_system = load_skill_system(skill_config_path)
    classifier = ActionClassifier(
        embedding_dim=1024,
        num_actions=skill_system.catalog.class_count,
        model_type="lstm",
    )
    # Load model - try with catalog first (v2 format), fall back to legacy format
    try:
        classifier.load(model_path, catalog=skill_system.catalog)
    except ValueError as e:
        if "Legacy checkpoint" in str(e):
            print(f"   ⚠ Using legacy model checkpoint (no catalog metadata)")
            classifier.load(model_path, catalog=None)
        else:
            raise
    print(f"   ✓ Model loaded from: {model_path}")

    probabilities = classifier.predict_probabilities(embeddings, context_window=context_window)
    print(f"   ✓ Predicted complete scores for {len(probabilities)} frames")
    print(f"   ✓ Context window: ±{context_window} frames")

    # Compute timestamps
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    timestamps = (np.asarray(frame_indices) - 1) / fps
    predictions = predictions_from_probabilities(
        probabilities,
        timestamps,
        skill_system.catalog,
        frame_indices=frame_indices,
    )
    artifacts = build_action_inference_artifacts(
        predictions,
        skill_system,
        checkpoint_sha256(model_path),
    )
    segments = _accepted_action_segments(artifacts.robot_actions)
    print(f"   ✓ Detected {len(segments)} action segments")
    return artifacts


def run_robot_simulation(
    config_path: str,
    waypoints: PickPlaceWaypoints,
    output_dir: Path,
    video_stem: str,
) -> dict:
    """Run robot simulation with inferred waypoints.

    Args:
        config_path: Path to robot execution config
        waypoints: Validated world-space pick-place waypoints
        output_dir: Directory to save simulation results
        video_stem: Stem of video file for naming outputs

    Returns:
        Dict with simulation results
    """
    print("\n6. Running robot simulation...")

    config_path_obj = Path(config_path)
    if not config_path_obj.exists():
        print(f"   ⚠ Config not found: {config_path}")
        print("   Skipping robot simulation")
        return {"status": "skipped", "reason": "config_not_found"}

    sim_output_dir = output_dir / "simulation"
    sim_output_dir.mkdir(parents=True, exist_ok=True)

    log_file = sim_output_dir / f"{video_stem}_execution.jsonl"
    events = []

    def record_event(event):
        events.append(event)

    try:
        from mimic.robot.factory import build_executor

        executor = build_executor(config_path_obj, record_event)
        print(f"   ✓ Executor built from config: {config_path}")

        report = executor.run(waypoints)

        # Save execution log
        with open(log_file, "w") as f:
            for event in events:
                json.dump(event, f, allow_nan=False)
                f.write("\n")

        print(f"   ✓ Simulation executed (success={report.success})")
        print(f"   ✓ Execution log saved to: {log_file}")

        return {
            "status": "completed",
            "success": report.success,
            "log_file": str(log_file),
            "report": asdict(report) if hasattr(report, "__dataclass_fields__") else str(report),
        }

    except Exception as e:
        print(f"   ✗ Simulation failed: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "log_file": str(log_file),
        }


def main():
    """Process video through full pipeline."""
    parser = argparse.ArgumentParser(description="Process demo video through full pipeline")
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
    parser.add_argument(
        "--config",
        type=str,
        help="Path to robot execution config (required for --simulate-robot)",
    )
    parser.add_argument(
        "--calibration",
        type=str,
        help="Camera homography JSON (required for --simulate-robot)",
    )
    parser.add_argument(
        "--retargeting-config",
        type=str,
        default="configs/retargeting.yaml",
        help="Table-to-MuJoCo mapping and tabletop clone YAML",
    )
    parser.add_argument(
        "--robot-pipeline-config",
        type=str,
        help="Explicit path and waypoint construction YAML (required for --simulate-robot)",
    )
    parser.add_argument(
        "--episode",
        type=int,
        help="One-based complete episode to simulate when multiple are present",
    )
    parser.add_argument(
        "--skill-config",
        type=str,
        default="configs/skills/pick_place.yaml",
        help="Skill catalog, graph, and explicit post-state settings YAML",
    )
    parser.add_argument(
        "--simulate-robot",
        action="store_true",
        help="Run robot simulation with inferred waypoints",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=32,
        help="Context window size for LSTM (±N frames, default 32)",
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

    if args.simulate_robot:
        required_robot_inputs = {
            "--config": args.config,
            "--calibration": args.calibration,
            "--robot-pipeline-config": args.robot_pipeline_config,
        }
        missing = [name for name, value in required_robot_inputs.items() if not value]
        if missing:
            print(f"ERROR: {', '.join(missing)} required when using --simulate-robot")
            return 1
        for name, value in {
            **required_robot_inputs,
            "--retargeting-config": args.retargeting_config,
        }.items():
            if not Path(value).is_file():
                print(f"ERROR: {name} file not found: {value}")
                return 1

    print("\n" + "=" * 70)
    print("DEMO VIDEO PROCESSING PIPELINE")
    print("=" * 70)
    print(f"\nVideo: {args.video}")
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print(f"Device: {args.device}")
    if args.simulate_robot:
        print(f"Robot Config: {args.config}")
        print("Simulate Robot: Yes")

    try:
        # Run pipeline
        tracks_data = extract_tracks(str(video_path), device=args.device)
        embeddings, fps, frame_count, embedding_frame_indices = extract_embeddings(
            str(video_path), device=args.device
        )
        actions_data = predict_actions(
            embeddings,
            str(model_path),
            fps,
            args.skill_config,
            embedding_frame_indices,
            context_window=args.context_window,
        )
        print("\n4. Building consolidated robot task input...")
        task_input = build_demo_task_input(tracks_data, actions_data.robot_actions)
        print(
            "   ✓ Preserved "
            f"{len(task_input.resolved_actions)} resolved actions and "
            f"{len(task_input.object_tracks)} tracker samples"
        )

        # Save results
        print("\n5. Saving results...")
        task_input_file = output_dir / f"{video_path.stem}_task_input.json"
        scores_file = output_dir / f"{video_path.stem}_scores.json"
        write_results(scores_file, actions_data.scores)
        write_results(task_input_file, task_input)
        print(f"   ✓ Robot task input: {task_input_file}")
        print(f"   ✓ Diagnostic scores: {scores_file}")

        # Run robot simulation if requested
        simulation_result = None
        if args.simulate_robot:
            robot_artifacts = build_robot_pipeline(
                task_input_source=task_input,
                calibration_source=args.calibration,
                retargeting_source=args.retargeting_config,
                pipeline_config_source=args.robot_pipeline_config,
                episode=args.episode,
            )
            world_waypoints_file = output_dir / f"{video_path.stem}_world_waypoints.json"
            write_world_waypoints(
                world_waypoints_file,
                robot_artifacts.waypoints,
                overwrite=True,
            )
            print(f"   ✓ World waypoints: {world_waypoints_file}")
            simulation_result = run_robot_simulation(
                args.config,
                robot_artifacts.waypoints,
                output_dir,
                video_path.stem,
            )

        # Print summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"\nTotal video frames: {task_input.video.frame_count}")
        print(f"Duration: {task_input.video.duration_s:.2f}s")
        print(f"FPS: {task_input.video.fps:.1f}")
        action_segments = _accepted_action_segments(actions_data.robot_actions)
        print(f"\nAction segments: {len(action_segments)}")
        for seg in action_segments:
            confidence = seg["avg_confidence"]
            confidence_text = "n/a" if confidence is None else f"{confidence:.1%}"
            print(
                f"  {seg['action']:10s} {seg['start_time']:6.2f}s - {seg['end_time']:6.2f}s (conf: {confidence_text})"
            )

        print(
            "\nTracked positions: "
            f"{sum(frame.position is not None for frame in task_input.object_tracks)}"
            f"/{task_input.video.frame_count}"
        )

        if simulation_result:
            print(f"\nRobot Simulation: {simulation_result['status']}")
            if simulation_result["status"] == "completed":
                print(f"  Success: {simulation_result['success']}")
                print(f"  Log: {simulation_result['log_file']}")

        print("\n" + "=" * 70)

        return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
