# Mimic: Learning Manipulation Skills from Human Demonstration

A system that learns manipulation skills from human video demonstrations and executes them with a simulated Franka Panda robot.

## Project Overview

**Goal:** Watch a human perform a tabletop manipulation task, understand what action is happening, extract where objects are moving, and reproduce the task with a simulated robot arm.

**Core Idea:** Separate visual understanding (What is happening?) from geometry (Where is it happening?), then retarget to robot control.

See [docs/PROJECT_OVERVIEW.md](./docs/PROJECT_OVERVIEW.md) for detailed technical specification.

## Project Structure

```
mimic/
├── data/                    # Datasets (raw videos, embeddings, annotations)
├── src/mimic/               # Main package
│   ├── data_pipeline/       # Audio-derived label preparation
│   ├── vision/              # Frame features & temporal action model
│   ├── tracking/            # Hand/object tracking & coordinate mapping
│   ├── skills/              # Versioned labels, relationship graph, state resolver
│   ├── robot/               # Task representation & robot control
│   ├── integration/         # End-to-end pipeline
│   └── common/              # Shared types & utilities
├── scripts/                 # Executable entry points
├── configs/                 # Configuration files
├── tests/                   # Unit & integration tests
├── results/                 # Selected reproducible pipeline artifacts
└── docs/                    # Documentation
```

## Quick Start

### Setup
```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Run tests
uv run pytest tests/
```

### Training the Temporal Model
```bash
uv run python scripts/train_action_classifier.py \
  --embeddings-dir data/embeddings \
  --labels-dir data/labels \
  --output-dir models
```

### Running Full Pipeline
```bash
uv sync --group robot
uv run --group robot mimic --video data/raw/demo_test.mp4 --robot panda
```

The `mimic` command runs tracking and model inference, builds Panda world
waypoints with the committed Panda defaults, executes MuJoCo, and writes all
artifacts under `results/demo_test/`. Simulation recording is enabled by default;
the final line prints the absolute path to
`results/demo_test/demo_test.mimic.mp4`.

```text
results/demo_test/
├── demo_test_scores.json
├── demo_test_task_input.json
├── demo_test_world_waypoints.json
├── demo_test_execution.jsonl
└── demo_test.mimic.mp4
```

Use `--device mps` for Apple Silicon inference or `--device cuda` for an NVIDIA
GPU. The default `cpu` device affects only Torch video inference, not MuJoCo
control. `--output`, `--model`, `--video-out PATH`, `--no-video-out`, `--viewer`,
and `--dry-run` provide the supported top-level overrides.

The current `panda` profile resolves the checked-in classifier, skill system,
camera calibration, retargeting, path/waypoint settings, and complete Panda
simulation configuration explicitly. Videos with multiple complete episodes
retain the lower pipeline's fail-closed behavior.

### Robot Simulation
```bash
uv run python scripts/fetch_panda_model.py
uv run --group robot python scripts/simulate_robot.py \
  --config path/to/experiment.yaml --waypoints path/to/world_waypoints.json \
  --log outputs/robot_attempt.jsonl
```

To watch the execution on macOS, run the same command with `mjpython` and
`--viewer`; see [Robot Execution](docs/ROBOT_EXECUTION.md) for the full command.

To build those world waypoints from an existing consolidated post-model task
input without rerunning video inference:

```bash
uv run mimic-robot-pipeline \
  --task-input results/demo/demo_task_input.json \
  --calibration data/annotations/calibrations.json \
  --pipeline-config configs/robot_pipeline.yaml \
  --waypoints results/demo/demo_world_waypoints.json
```

Add `--robot-config path/to/experiment_robot.yaml --log outputs/robot_attempt.jsonl`
to execute the generated waypoints immediately. The committed
`configs/robot_pipeline.yaml` contains the simulation-only 4 cm cube fixture
geometry; use a separate experiment config for another object or scene.
Add `--video-out outputs/simulation.mp4` to record that execution, or pass
`--video-out` without a path to create a timestamp-named MP4 in the working
directory.

The top-level `panda` profile uses the checked-in simulation fixture. The generic
`configs/robots/panda.yaml` remains an unconfigured contract template. See
[Robot Execution](docs/ROBOT_EXECUTION.md) for setup, interfaces, and limits.

## Team Workspace

Each subsystem is independently maintained:

- **Data Pipeline** (`src/mimic/data_pipeline/`) — Audio-derived label preparation
- **Vision** (`src/mimic/vision/`) — Frame features and temporal classifier training
- **Tracking** (`src/mimic/tracking/`) — Hand/object tracking, coordinate transforms
- **Robot** (`src/mimic/robot/`) — Task representation, state machine, IK, control
- **Integration** (`src/mimic/integration/`) — End-to-end inference pipeline

Each team member can work in their folder with minimal interference. See [CONTRIBUTING.md](./CONTRIBUTING.md) for conventions.

## Shared Interfaces

Shared records in `src/mimic/common/types.py` include `ActionPrediction`,
`ObjectTrack`, `ExtractedTask`, `RetargetedTask`, `ToolPose`, and
`PickPlaceWaypoints`.

`configs/default.yaml` is the single source of project defaults. `src/mimic/config.py`
loads it and layers experiment-specific and local YAML overrides on top.

### Offline Task Definition

The Task Extractor and NumPy-backed Coordinate Retargeter accept already labeled
predictions and table-space object tracks keyed by shared source-video frame IDs.
Tracking coordinates are meters (top-left origin, +X right, +Y down).
Tasks and retargeting preserve every demonstration sample. The robot-independent
Path Processor then selects `direct`, `corners_only`, exact `none`, or `cubic`
geometry. `cubic` is the configured default; `direct` remains available for
endpoint-only paths. Retargeting requires the explicit mapping committed in
`configs/retargeting.yaml`; deployments should use a separately validated mapping.

See [Task Extraction and Retargeting](docs/TASK_EXTRACTION_AND_RETARGETING.md)
for the input contract, mapping, path-processing behavior, and failure handling.

## Hardware

- **Feature extraction and classifier training**: RTX 3090 (24GB VRAM)
- **Data collection, tracking, simulation, integration**: M4 Pro MacBook
- Embeddings are cached for fast iteration on temporal model

## Key Components

### Visual Understanding
- **Encoder abstraction**: current integrated backend is framewise ResNet50; see
  [classifier documentation](docs/VJEPA_CLASSIFIER_PIPELINE.md)
- **Classifier**: learned LSTM temporal model for action phases
- **Output**: Configured composite-skill probabilities. The default preset is
  IDLE, HOVER, GRASP, CARRY, RELEASE.

### Geometric Tracking
- **Hand**: MediaPipe landmarks
- **Object**: current HSV/CSRT tracking path; SAM2 remains optional future work
- **Mapping**: Camera → table coordinates → robot workspace

### Robot Execution
- **Task Representation**: Composite skills and separately tracked path geometry
- **State Machine**: Versioned skill catalog plus an explicit relationship graph
- **IK & Control**: Convert Cartesian waypoints to Panda joint targets
- **Simulation**: MuJoCo Franka Panda with gripper

## MVP Success Criterion

1. Record unseen human demonstration
2. Extract frame features
3. Classify composite skills (IDLE, HOVER, GRASP, CARRY, RELEASE by default)
4. Track object trajectory
5. Map to robot workspace
6. Solve IK and execute in MuJoCo
7. **Result**: Object moved to approximately demonstrated destination (with different start/end locations than training)

## Stretch Goals

- Language grounding (action commands from speech)
- Orientation estimation & reproduction
- Multiple object tracking
- Expanded action vocabulary (PUSH, PULL, ROTATE, etc.)
- Variable camera viewpoints
- Real robot transfer

## References

- MediaPipe: Hand tracking
- MuJoCo: Physics simulation
- Franka Research 3: Robot control

---

For issues, see [CONTRIBUTING.md](./CONTRIBUTING.md). For technical details, see [docs/](./docs/).
