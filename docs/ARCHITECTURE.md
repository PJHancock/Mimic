# System Architecture

High-level design and module interactions for the Mimic project.

## Overview

```
Human Video
    ↓
┌─────────────────────────────────────────────────────────────┐
│                  Vision Pipeline                            │
│  ┌──────────────┐    ┌──────────────────┐                 │
│  │ V-JEPA 2     │───→│ Temporal Action  │                 │
│  │ (frozen)     │    │ Classifier (GRU) │                 │
│  └──────────────┘    └──────────────────┘                 │
│                      APPROACH/GRASP/MOVE/RELEASE          │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│                  Tracking Pipeline                          │
│  ┌──────────────┐    ┌──────────────────┐                 │
│  │ Hand Track   │    │ Object Track     │                 │
│  │ (MediaPipe)  │────│ (SAM2)           │                 │
│  └──────────────┘    └──────────────────┘                 │
│         ↓                    ↓                              │
│    ┌────────────────────────────────┐                      │
│    │ Coordinate Mapper              │                      │
│    │ Camera → Table → Robot Coords   │                      │
│    └────────────────────────────────┘                      │
└─────────────────────────────────────────────────────────────┘
    ↓
    ├─→ Action Predictions + Tracked Geometry
    ↓
┌─────────────────────────────────────────────────────────────┐
│                  Robot Pipeline                            │
│  ┌──────────────┐    ┌──────────────────┐                 │
│  │ Task Symbol  │───→│ State Machine    │                 │
│  │ Repr.        │    │ (APPROACH→GRASP→ │                 │
│  └──────────────┘    │  MOVE→RELEASE)   │                 │
│                      └──────────────────┘                 │
│                                ↓                           │
│                      ┌──────────────────┐                 │
│                      │ Trajectory Gen   │                 │
│                      │ (smooth, retarget)                 │
│                      └──────────────────┘                 │
│                                ↓                           │
│                      ┌──────────────────┐                 │
│                      │ Inverse Kinematics                │
│                      │ Cartesian→Joints │                 │
│                      └──────────────────┘                 │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│                  Panda + MuJoCo                             │
│  ┌──────────────┐    ┌──────────────────┐                 │
│  │ Arm Control  │    │ Gripper Control  │                 │
│  │ (~100Hz)     │    │ (~10Hz)          │                 │
│  └──────────────┘    └──────────────────┘                 │
└─────────────────────────────────────────────────────────────┘
    ↓
   Success: Object at target location
```

## Module Organization

### `src/mimic/common/`
**Shared types and constants**
- `types.py` — Data classes (`ActionPhase`, `ObjectTrack`, `TaskRepresentation`, etc.)
- `constants.py` — Shared constants and defaults

### `src/mimic/data_pipeline/`
**Data collection and preparation**
- `recording.py` — Video recording utilities
- `transcription.py` — Speech-to-text (Whisper)
- `annotation.py` — Label generation from speech
- Exports: Video clips, annotations, transcript timestamps

### `src/mimic/vision/`
**Visual understanding**
- `vjepa_encoder.py` — V-JEPA 2 inference and embedding caching
- `temporal_model.py` — GRU/Transformer classifier
- `training.py` — Training loop and evaluation
- `inference.py` — Prediction interface
- Exports: `ActionPrediction` sequence (phase + confidence over time)

### `src/mimic/tracking/`
**Geometric tracking**
- `hand_tracker.py` — MediaPipe hand landmark detection
- `object_tracker.py` — SAM2 or similar object tracking
- `coordinate_mapping.py` — Camera ↔ table ↔ robot coordinate transforms
- `trajectory.py` — Trajectory smoothing and resampling
- Exports: `HandTrack` and `ObjectTrack` sequences

### `src/mimic/robot/`
**Robot control logic**
- `task.py` — Task representation (action sequence + geometry)
- `state_machine.py` — Deterministic state-based behavior
- `trajectory_gen.py` — Waypoint generation and smoothing
- `inverse_kinematics.py` — Cartesian to joint-space conversion
- `controller.py` — Panda arm and gripper control
- `simulation.py` — MuJoCo interface
- Exports: `RobotCommand` sequence for simulation/hardware

### `src/mimic/integration/`
**End-to-end pipeline**
- `pipeline.py` — Orchestrates all modules
- `visualization.py` — Debugging and result visualization
- Exports: Full video → robot execution

## Data Flow

### Inference Pipeline
```
Input: Video file
  ↓
Vision Module:
  • Extract V-JEPA embeddings (cached)
  • Temporal model predicts actions
  ↓
Tracking Module:
  • Hand & object tracking
  • Coordinate mapping
  ↓
Robot Module:
  • Build task representation
  • Generate robot commands
  ↓
Output: MuJoCo simulation or hardware commands
```

### Training Pipeline
```
Input: Annotated demonstrations
  ↓
Vision:
  • Precompute V-JEPA embeddings (once)
  • Create dataset from embeddings + labels
  • Train temporal model
  ↓
Output: Trained model checkpoint
```

## Key Design Decisions

### 1. Frozen V-JEPA
- V-JEPA 2 weights are not trained
- Only temporal classifier is learned
- Rationale: V-JEPA is expensive to train; good pretrained features exist

### 2. Separate Geometry
- Tracking is independent of action recognition
- Rationale: Decouples concerns; easier to debug; allows swapping trackers

### 3. Symbolic Task Representation
- Task is independent of robot embodiment
- Rationale: Can transfer to different robots; easier to reason about

### 4. State Machine Control
- Robot uses deterministic rules based on action phases
- Rationale: Interpretable, safe, not requiring learned control policy

### 5. Coordinate Normalization
- Workspace coordinates are normalized [0,1]
- Rationale: Generalizes across different table sizes; robot-agnostic

## Interfaces

### Between Vision and Tracking
```python
# Vision outputs
predictions: List[ActionPrediction]  # per frame

# Tracking consumes
action_phase: ActionPhase  # current phase (from vision)
```

### Between Tracking and Robot
```python
# Tracking outputs
hand_tracks: List[HandTrack]
object_tracks: List[ObjectTrack]
trajectory: List[Tuple[float, float]]  # normalized coords

# Robot consumes
task: TaskRepresentation
```

### Between Robot and Simulation
```python
# Robot outputs
commands: List[RobotCommand]

# Simulation consumes
commands: List[RobotCommand]
```

## Configuration

Global configuration in `src/mimic/config.py`:
```python
from mimic.config import get_config

cfg = get_config()
device = cfg["device"]  # "cuda", "cpu", "mps"
fps = cfg["fps"]
```

Experiment-specific configs override defaults:
```bash
python scripts/train.py --config configs/experiment_1.yaml
```

## Testing Strategy

- **Unit tests** in `tests/test_<module>/` for each module
- **Integration tests** in `tests/test_integration.py` for cross-module boundaries
- **Fixtures** for common test data (synthetic videos, embeddings, etc.)

Example:
```
tests/
├── test_vision/
│   └── test_temporal_model.py
├── test_tracking/
│   └── test_coordinate_mapping.py
├── test_robot/
│   └── test_state_machine.py
└── test_integration.py
```

## Performance Targets

| Component | Latency | Throughput |
|-----------|---------|-----------|
| V-JEPA embedding | ~100ms/frame | RTX 3090 |
| Temporal classifier | ~1ms/frame | CPU |
| Hand tracking | ~50ms/frame | CPU |
| Object tracking | ~100ms/frame | GPU optional |
| IK solver | ~10ms/waypoint | CPU |
| Arm controller | 10Hz | MuJoCo |

---

See [PROJECT_OVERVIEW.md](./PROJECT_OVERVIEW.md) for detailed component specifications.
