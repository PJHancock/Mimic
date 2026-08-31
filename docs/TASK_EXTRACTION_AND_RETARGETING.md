# Offline task extraction and coordinate retargeting

This is the user-approved contract for pipeline steps 2 and 3 on the robot
environment branch. It supersedes older descriptions of normalized tracking
coordinates and mandatory second-based timing at these two boundaries.

## Scope

- Input: one complete, already labeled, single-object demonstration.
- Step 2: `TaskExtractor` recovers task geometry and phase boundaries.
- Step 3: `CoordinateRetargeter` maps source geometry into a named target frame.
- State post-processing is deferred. Invalid phase sequences are rejected, not repaired.
- No robot actuation, IK, collision planning, trajectory smoothing, heights,
  velocities, or demonstration-speed replay happens here.

## Input contract

`ActionPrediction.frame_idx` and `ObjectTrack.frame_idx` refer to the **same
one-based source-video frame numbering**, not independent sample counters.
Frame IDs must be positive integers, unique and strictly increasing within
each stream. Predictions and tracks may have different sampling rates.

`ActionPrediction.phase` supplies a resolved label. After collapsing adjacent
repeats, the input must contain exactly:

`APPROACH -> GRASP -> MOVE -> RELEASE`

`ActionPrediction.timestamp` is optional. If supplied, it remains nonnegative,
finite video time in seconds. Extraction uses frames; it never substitutes frame
counts for seconds or derives timestamps from a default FPS.

`ObjectTrack.table_xy_cm` contains **already calibrated table coordinates**:

- Units: centimeters.
- Origin: top-left of the table.
- Positive X: right.
- Positive Y: down.

The former `center_2d` keyword has been removed deliberately. Callers must update
the field name and supply calibrated table coordinates, not merely relabel pixel
values. Image calibration remains upstream. Existing `center_3d`/`bbox` fields
are not consumed by extraction.

All tracks must share one `object_id`; consistently absent IDs are accepted for
a caller-provided single-object stream. Mixed IDs, including mixed known/unknown
IDs, are rejected. Tracker confidence is preserved (finite, within `[0, 1]`),
with no new rejection threshold.

## Task extraction

```python
from mimic.common import ActionPhase, ActionPrediction, ObjectTrack, PathMode
from mimic.robot import extract_task

predictions = [
    ActionPrediction(1, ActionPhase.APPROACH, 0.9),
    ActionPrediction(10, ActionPhase.GRASP, 0.9),
    ActionPrediction(20, ActionPhase.MOVE, 0.9),
    ActionPrediction(40, ActionPhase.RELEASE, 0.9),
]
tracks = [
    ObjectTrack(10, table_xy_cm=(10, 20)),
    ObjectTrack(15, table_xy_cm=(12, 22)),
    ObjectTrack(25, table_xy_cm=(20, 30)),
    ObjectTrack(40, table_xy_cm=(30, 20)),
]

task = extract_task(predictions, tracks)
assert task.grasp_frame == 10
assert task.release_frame == 40
assert task.get_path() == ((10.0, 20.0), (30.0, 20.0))
assert len(task.get_path(PathMode.FOLLOW)) == 4
```

The first GRASP frame and first RELEASE frame define the endpoints. Both require
an **exact tracking observation**. Missing either raises `TaskExtractionError`;
there is no nearest-frame match, interpolation, or fallback.

`ExtractedTask` contains immutable phase boundaries and copied, immutable
`TablePathSample` records. Endpoints are derived from those records. Editing input
lists, arrays, or predictions afterward cannot alter an extracted task.

`demonstrated_path` retains every available observation from GRASP onset through
RELEASE onset, inclusive. Each sample keeps its original frame, coordinates,
confidence, and phase. Phase intervals are `[onset, next_onset)`; a tracking frame
between predictions inherits the supplied segment label. This is not a new
classification or a claim that missing observations were reconstructed.

`move_trajectory_xy_cm` exposes only MOVE-phase observations, maintaining the
distinction between the canonical MOVE trajectory and the full handled interval.
Interior tracking gaps remain visible in frame IDs. No gaps are filled; the
MOVE-only subset can be empty. Downstream FOLLOW execution must decide whether
the available geometry is sufficient before generating a safe trajectory.

The legacy `TaskRepresentation` type still describes normalized workspace
coordinates. Do not put centimeters or target meters into its normalized fields.

## Required mapping configuration

`configs/retargeting.yaml` is intentionally **not runnable calibration**. All
deployment placement/orientation values are `null`. Populate it explicitly, or
provide an equivalent mapping dictionary; otherwise configuration validation fails.

Required fields:

| Field | Meaning |
|---|---|
| `source_frame` | Must be `table`, matching the extraction contract |
| `target_frame` | A distinct, nonempty target coordinate-frame name |
| `table_origin_target_xy_m` | Location of table `(0, 0)` in target XY meters |
| `table_x_axis_target_xy` | Unit direction of increasing table X in target XY |
| `table_y_axis_target_xy` | Unit direction of increasing table Y in target XY |

For a column-vector table position `p_cm`, the mapping is:

```text
A = [table_x_axis_target_xy  table_y_axis_target_xy]  # axes are columns
p_target_m = table_origin_target_xy_m + A @ (p_cm / 100)
```

This preserves physical distances: centimeters are converted to meters exactly
once. The XY axis vectors must be unit length and perpendicular. Numerical axis
validation uses an absolute tolerance of `1e-10` with zero relative tolerance;
this only accommodates floating-point representation, not calibration uncertainty
or task-success tolerance. Axes are never silently normalized. Reflections in
the 2D plane are allowed when explicitly configured; there is no guessed Y flip.
Resizing, shear, clipping, and automatic calibration are not supported.

Pydantic validates required fields, tuple sizes, real finite numeric values and
unknown keys. YAML lists are converted to immutable tuples; numeric strings and
booleans are rejected. No mapping values are taken from Panda constants.

```python
from mimic.config import Config
from mimic.robot import retarget_task

# Deliberately raises validation errors until the deployment values are supplied.
mapping = Config("configs/retargeting.yaml").get("retargeting")
target_task = retarget_task(task, mapping)

direct_xy_m = target_task.get_path()          # two transport endpoints
follow_xy_m = target_task.get_path("FOLLOW") # all retained mapped samples
```

`RetargetedTask` retains its immutable `source_task`, a named `target_frame`, and
one target XY point per source observation. Frame IDs/phases/confidence are
available through that source task. A mapping does not move the simulated object,
adapt to its current position, or prove robot reachability.

## Path selection and downstream integration

Both source and target tasks support `get_path(mode=PathMode.DIRECT)`:

- `DIRECT`: endpoints of a straight-line object transport path.
- `FOLLOW`: preserved geometric samples including the same endpoints.

Selection never overwrites the stored demonstration. No file serialization is
required by this milestone. An unknown mode raises `ValueError`.

`DIRECT` is not an arm-home-to-goal command, and `FOLLOW` does not assign human
timing to a robot. The later Path Processor must produce suitable trajectories,
script vertical motion, and supply tool orientation. The existing execution-side
`PickPlaceWaypoints` expects full tool poses and cannot consume these XY tasks
directly. No implicit adapter or guessed height/orientation is introduced.

The package exposes extraction/retargeting without importing MuJoCo, Mink, or
Torch. Existing execution exports load their dependencies only when requested.

## Verification

```bash
uv run pytest tests/test_robot/test_task_extractor.py tests/test_robot/test_coordinate_retargeter.py
```

Tests use synthetic mappings, not deployment defaults. They check exact onset
matching, invalid input rejection, sampling-rate independence, explicit axis
rotation/reflection, cm-to-m conversion, round trips, immutable source retention,
unset configuration rejection, and backend-independent imports. These checks do
not demonstrate simulation success or physical robot safety.
