# Scripts

Entry points for running different parts of the Mimic pipeline.

**Note**: All scripts can be run with `uv run` or from an activated `.venv`:
```bash
# Option 1: Using uv run (recommended)
uv run python scripts/train_temporal_model.py --config configs/default.yaml

# Option 2: Activate venv first
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
python scripts/train_temporal_model.py --config configs/default.yaml
```

## Data Pipeline

### record_demo.py
Record a human demonstration video.
```bash
python scripts/record_demo.py --duration 15 --output data/raw/demo_001.mp4
```

### transcribe_audio.py
Transcribe speech narration using Whisper.
```bash
python scripts/transcribe_audio.py --video data/raw/demo_001.mp4 --output data/annotations/demo_001.json
```

### create_annotations.py
Create frame-level labels from speech timestamps.
```bash
python scripts/create_annotations.py --video data/raw/demo_001.mp4 --transcript data/annotations/demo_001.json
```

## Vision Module

### extract_embeddings.py
Extract V-JEPA embeddings for all demonstrations.
```bash
python scripts/extract_embeddings.py --video-dir data/raw --output-dir data/embeddings
```

### train_temporal_model.py
Train the temporal action classification model.
```bash
python scripts/train_temporal_model.py --config configs/default.yaml
```

### evaluate_model.py
Evaluate trained model on test set.
```bash
python scripts/evaluate_model.py --model outputs/model.pth --config configs/default.yaml
```

## Tracking & Geometry

### extract_tracks.py
Extract hand and object tracks from videos.
```bash
python scripts/extract_tracks.py --video-dir data/raw --output-dir data/tracks
```

### calibrate_camera.py
Perform camera calibration for coordinate mapping.
```bash
python scripts/calibrate_camera.py --calibration-video data/raw/calibration.mp4
```

## Robot Pipeline

### inference_action_classifier.py

Run classifier inference, graph-aware post-processing, and export separate
diagnostic and robot-action files:

```bash
uv run python scripts/inference_action_classifier.py \
  --embeddings data/embeddings/demo.npy \
  --model models/action_classifier_lstm.pt \
  --fps 30 \
  --skill-config path/to/experiment_skill.yaml \
  --output results/demo_robot_actions.json
```

The supplied skill config must contain explicit post-state thresholds. The
committed `configs/skills/pick_place.yaml` intentionally retains `null` template
values and therefore fails closed for inference. The robot output contains one
resolved phase per timestep; a sibling `_scores.json` file retains all model
probabilities. Load robot actions with `mimic.integration.load_robot_actions`.

### run_inference.py
Run the full inference pipeline on a video.
```bash
python scripts/run_inference.py --video data/raw/demo_test.mp4 --model outputs/model.pth --visualize
```

### simulate_robot.py
Execute explicitly configured, processed world-space waypoints in headless MuJoCo.
```bash
uv run --group robot python scripts/simulate_robot.py \
  --config path/to/experiment.yaml --waypoints path/to/world_waypoints.json \
  --log outputs/robot_attempt.jsonl
```

Use `fetch_panda_model.py` for the one-time pinned asset download. Use
`verify_panda.py --output outputs/robot_verification/new_attempt` with `uv run --group robot`
for the fixed diagnostic fixture. These entry points require explicit arguments,
not the general default config. See [Robot Execution](../docs/ROBOT_EXECUTION.md)
for unresolved general configuration and the successful fixed-fixture result.

## Utilities

### explore_data.py
Quick inspection of dataset.
```bash
python scripts/explore_data.py --video-dir data/raw
```

### benchmark.py
Benchmark different components.
```bash
python scripts/benchmark.py --component vision
```

---

**Note**: All scripts read configuration from `configs/default.yaml` by default. Override with `--config path/to/config.yaml`.
