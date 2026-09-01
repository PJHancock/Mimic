# Scripts

Use `uv run python scripts/<name>.py --help` for complete options. Installed pipeline entry points are preferred for normal operation.

## Supported pipelines

Complete video-to-Panda simulation:

```bash
uv run --group robot mimic --video path/to/demo.mov --robot panda
```

Inference only:

```bash
uv run mimic-video-pipeline path/to/demo.mov
```

Existing task input to world waypoints:

```bash
uv run mimic-robot-pipeline \
  --task-input results/demo/demo_task_input.json \
  --calibration data/annotations/calibrations.json \
  --pipeline-config configs/robot_pipeline.yaml \
  --waypoints results/demo/demo_world_waypoints.json
```

## Data and calibration

- `extract_calibration_frame.py` — save a selected video frame for calibration.
- `calibrate_camera.py` — interactively compute and save a pixel-to-table homography.
- `extract_labels.py` — derive frame labels from narrated demonstrations.
- `extract_tracks.py` — extract hand/object tracks.

## Features and classification

- `extract_vjepa_embeddings.py` — cache frame features. See the encoder-status warning in `docs/VJEPA_CLASSIFIER_PIPELINE.md`.
- `validate_vjepa_embeddings.py` — inspect feature statistics and projections.
- `train_action_classifier.py` — train the temporal classifier and record its catalog provenance.
- `inference_action_classifier.py` — run classifier/postprocessor inference from cached embeddings.
- `process_demo_video.py` — implementation used by the installed video pipeline.
- `visualize_results.py` — render action/tracking overlays from `mimic.demo_task_input.v1`.

## Robot simulation

- `fetch_panda_model.py` — fetch the pinned Menagerie Panda assets once.
- `simulate_robot.py` — execute validated world waypoints; requires `--group robot`.
- `verify_panda.py` — run the fixed diagnostic fixture and write detailed evidence.

On macOS, use `mjpython` with `simulate_robot.py --viewer`. See `docs/ROBOT_EXECUTION.md` for the exact command and simulation constraints.
