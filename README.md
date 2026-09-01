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
│   ├── data_pipeline/       # Recording, transcription, annotation
│   ├── vision/              # V-JEPA & temporal action model
│   ├── tracking/            # Hand/object tracking & coordinate mapping
│   ├── skills/              # Versioned labels, relationship graph, state resolver
│   ├── robot/               # Task representation & robot control
│   ├── integration/         # End-to-end pipeline
│   └── common/              # Shared types & utilities
├── scripts/                 # Executable entry points
├── notebooks/               # Exploratory analysis
├── configs/                 # Configuration files
├── tests/                   # Unit & integration tests
├── experiments/             # Results & experiment logs
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

### Recording a Demonstration
```bash
python scripts/record_demo.py --duration 15 --output data/raw/demo_001.mp4
```

### Training the Temporal Model
```bash
python scripts/train_temporal_model.py --config configs/default.yaml
```

### Running Full Pipeline
```bash
python scripts/run_inference.py --video data/raw/demo_test.mp4 --visualize
```

### Robot Simulation
```bash
uv sync --group robot
uv run python scripts/fetch_panda_model.py
uv run --group robot python scripts/simulate_robot.py \
  --config path/to/experiment.yaml --waypoints path/to/world_waypoints.json \
  --log outputs/robot_attempt.jsonl
```

To build those world waypoints from existing postprocessed actions and pixel
tracking results without rerunning video inference:

```bash
uv run mimic-robot-pipeline \
  --actions results/demo/demo_robot_actions.json \
  --results results/demo/demo_results.json \
  --calibration data/annotations/calibrations.json \
  --pipeline-config path/to/experiment_robot_pipeline.yaml \
  --waypoints results/demo/demo_world_waypoints.json
```

Add `--robot-config path/to/experiment_robot.yaml --log outputs/robot_attempt.jsonl`
to execute the generated waypoints immediately. The committed
`configs/robot_pipeline.yaml` is a fail-closed template: copy it and explicitly
set the scene's world-Z coordinates and fixed tool quaternion before use.

Requires explicit scene/tool geometry and acceptance criteria; the Panda config is
an unconfigured template. See [Robot Execution](docs/ROBOT_EXECUTION.md) for setup,
interfaces, tests, and the retained simulation limit-check failure.

## Team Workspace

Each subsystem is independently maintained:

- **Data Pipeline** (`src/mimic/data_pipeline/`) — Recording, transcription, annotation
- **Vision** (`src/mimic/vision/`) — V-JEPA embeddings, temporal classifier training
- **Tracking** (`src/mimic/tracking/`) — Hand/object tracking, coordinate transforms
- **Robot** (`src/mimic/robot/`) — Task representation, state machine, IK, control
- **Integration** (`src/mimic/integration/`) — End-to-end inference pipeline

Each team member can work in their folder with minimal interference. See [CONTRIBUTING.md](./CONTRIBUTING.md) for conventions.

## Shared Interfaces

All modules use common types in `src/mimic/common/types.py`:
- `Video`, `Frame`, `ObjectTrack`, `HandTrack`
- `ActionPhase`, `TaskRepresentation`, `RobotCommand`

Configuration is centralized in `src/mimic/config.py`. Experiment parameters go in `configs/`.

### Offline Task Definition

The Task Extractor and NumPy-backed Coordinate Retargeter accept already labeled
predictions and table-space object tracks keyed by shared source-video frame IDs.
Tracking coordinates are meters (top-left origin, +X right, +Y down).
Tasks and retargeting preserve every demonstration sample. The robot-independent
Path Processor then selects `direct`, `corners_only`, exact `none`, or `cubic`
geometry. `direct` remains the default. Retargeting requires explicit mapping
values; `configs/retargeting.yaml` intentionally leaves deployment values unset.

See [Task Extraction and Retargeting](docs/TASK_EXTRACTION_AND_RETARGETING.md)
for the input contract, mapping, path-processing behavior, and failure handling.

## Hardware

- **V-JEPA inference & embedding extraction**: RTX 3090 (24GB VRAM)
- **Data collection, tracking, simulation, integration**: M4 Pro MacBook
- Embeddings are cached for fast iteration on temporal model

## Key Components

### Visual Understanding
- **Encoder**: Frozen V-JEPA 2 for spatiotemporal embeddings
- **Classifier**: Learned temporal model (GRU or transformer) for action phases
- **Output**: Configured composite-skill probabilities. The default preset is
  IDLE, HOVER, GRASP, CARRY, RELEASE.

### Geometric Tracking
- **Hand**: MediaPipe landmarks
- **Object**: SAM2 or similar tracker
- **Mapping**: Camera → table coordinates → robot workspace

### Robot Execution
- **Task Representation**: Composite skills and separately tracked path geometry
- **State Machine**: Versioned skill catalog plus an explicit relationship graph
- **IK & Control**: Convert Cartesian waypoints to Panda joint targets
- **Simulation**: MuJoCo Franka Panda with gripper

## MVP Success Criterion

1. Record unseen human demonstration
2. Extract V-JEPA embeddings
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

- V-JEPA 2: Meta's self-supervised video representation
- MediaPipe: Hand tracking
- SAM2: Video segmentation & tracking
- MuJoCo: Physics simulation
- Franka Research 3: Robot control

---

For issues, see [CONTRIBUTING.md](./CONTRIBUTING.md). For technical details, see [docs/](./docs/).
