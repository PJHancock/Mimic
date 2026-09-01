#!/usr/bin/env python3
"""End-to-end demo video processing: tracks + embeddings + actions + robot simulation.

Processes a new demo video through the complete pipeline:
  1. Extract object position tracks (from tracking module)
  2. Extract V-JEPA embeddings from frames
  3. Predict action sequences from embeddings
  4. Combine into unified output with timestamps
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
        --config config.yaml \\
        --output results/new_demo/ \\
        --simulate-robot
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
from dataclasses import asdict

import numpy as np
import cv2
from tqdm import tqdm

from mimic.common.types import ActionPhase, PickPlaceWaypoints, ToolPose
from mimic.vision import VJepaEncoder
from mimic.vision.action_classifier import ActionClassifier
from mimic.tracking import find_initial_bbox

# Action class names
ACTION_NAMES = [phase.value for phase in ActionPhase]


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
                tracks.append({
                    "frame": int(frame_idx),
                    "time": float(frame_idx / fps),
                    "x": float(center_x),
                    "y": float(center_y),
                    "confidence": float(confidence),
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

    classifier = ActionClassifier(
        embedding_dim=1024,
        num_actions=len(ACTION_NAMES),
        model_type="lstm",
    )
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


def generate_default_waypoints() -> dict:
    """Generate default pick-place waypoints for robot simulation.

    These are placeholder waypoints that should be replaced with
    actual values inferred from tracking data or learned from demonstration.

    Returns:
        Dict with waypoint format for robot simulation
    """
    # Default poses in world coordinates - adjust based on your robot/scene
    default_pose = {
        "position": [0.5, 0.0, 0.2],
        "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
    }

    return {
        "approach": {"position": [0.5, 0.0, 0.4], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]},
        "grasp": {"position": [0.5, 0.0, 0.1], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]},
        "lift": {"position": [0.5, 0.0, 0.3], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]},
        "lower": {"position": [0.6, 0.1, 0.1], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]},
        "retreat": {"position": [0.6, 0.1, 0.4], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]},
        "path": [
            {"position": [0.52, 0.02, 0.25], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]},
            {"position": [0.54, 0.04, 0.25], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]},
            {"position": [0.58, 0.08, 0.2], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]},
        ],
        "goal_position": [0.6, 0.1, 0.0],
    }


def run_robot_simulation(
    config_path: str,
    waypoints: dict,
    output_dir: Path,
    video_stem: str,
) -> dict:
    """Run robot simulation with inferred waypoints.

    Args:
        config_path: Path to robot execution config
        waypoints: Dict with pick-place waypoints
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

        # Convert waypoints dict to the expected format and run
        report = executor.run(PickPlaceWaypoints(**waypoints))

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
    parser.add_argument(
        "--config",
        type=str,
        help="Path to robot execution config (required for --simulate-robot)",
    )
    parser.add_argument(
        "--simulate-robot",
        action="store_true",
        help="Run robot simulation with inferred waypoints",
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

    if args.simulate_robot and not args.config:
        print("ERROR: --config is required when using --simulate-robot")
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
        print(f"Simulate Robot: Yes")

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

        # Run robot simulation if requested
        simulation_result = None
        if args.simulate_robot:
            waypoints = generate_default_waypoints()
            simulation_result = run_robot_simulation(
                args.config,
                waypoints,
                output_dir,
                video_path.stem,
            )

            # Add simulation result to output
            results["simulation"] = simulation_result

            # Save updated results with simulation
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2, allow_nan=False)
            print(f"   ✓ Updated results with simulation data")

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

        if simulation_result:
            print(f"\nRobot Simulation: {simulation_result['status']}")
            if simulation_result['status'] == 'completed':
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
