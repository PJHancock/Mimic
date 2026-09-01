# Offline task extraction, coordinate retargeting, and path processing

This is the user-approved contract for pipeline steps 2 through 4 on the robot
environment branch. It supersedes older descriptions of normalized tracking
coordinates and mandatory second-based timing at these two boundaries.

## Scope

- Input: one or more complete, already resolved, single-object episodes.
- Step 2: `TaskExtractor` recovers task geometry and phase boundaries.
- Step 3: `CoordinateRetargeter` maps source geometry into a named target frame.
- Step 4: `PathProcessor` explicitly selects or interpolates mapped XY geometry.
- State post-processing occurs upstream of this boundary. The persisted
  `mimic.demo_task_input.v1` handoff exposes only its single resolved phase per
  classifier timestep; invalid phase sequences are still rejected, not repaired
  here.
- No robot actuation, IK, collision planning, heights, velocities, or
  demonstration-speed replay happens here.

## Input contract

`ActionPrediction.frame_idx` and `ObjectTrack.frame_idx` refer to the **same
one-based source-video frame numbering**, not independent sample counters.
Frame IDs must be positive integers, unique and strictly increasing within
each stream. Predictions and tracks may have different sampling rates.

`ActionPrediction.phase` supplies a resolved label. After collapsing adjacent
repeats, each successful episode must contain either:

`IDLE -> HOVER -> GRASP -> CARRY -> RELEASE -> HOVER -> IDLE`

or:

`IDLE -> HOVER -> GRASP -> CARRY -> RELEASE -> IDLE`

IDLE is a legal terminal state from every classifier state, but an earlier IDLE
ends an incomplete/aborted sequence and does not make it extractable.

Adjacent episodes share their boundary `IDLE`. `extract_task` requires exactly
one episode; `extract_tasks` returns every complete episode in the timeline.
The extractor does not apply the relationship graph or repair incomplete input.
`mimic.integration.load_task_actions` is the validated persisted adapter for this
sequence. `load_robot_actions` remains the equivalent adapter for classifier-only
action artifacts. Raw `state_scores` and legacy top-one classifier files are
rejected at that boundary, so the robot-side extractor never selects among model
labels.

`ActionPrediction.timestamp` is optional. If supplied, it remains nonnegative,
finite video time in seconds. Extraction uses frames; it never substitutes frame
counts for seconds or derives timestamps from a default FPS.
`ActionPrediction.confidence` is the model probability of the accepted state
when a valid model detection exists. It is `None` for an accepted fallback state
rather than a fabricated probability; extraction does not threshold it.

`ObjectTrack.table_xy_m` contains **already calibrated table coordinates**:

- Units: meters.
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
from mimic.common import ActionPhase, ActionPrediction, ObjectTrack
from mimic.robot import extract_task

predictions = [
    ActionPrediction(1, ActionPhase.IDLE, 0.9),
    ActionPrediction(5, ActionPhase.HOVER, 0.9),
    ActionPrediction(10, ActionPhase.GRASP, 0.9),
    ActionPrediction(20, ActionPhase.CARRY, 0.9),
    ActionPrediction(40, ActionPhase.RELEASE, 0.9),
    ActionPrediction(45, ActionPhase.HOVER, 0.9),
    ActionPrediction(50, ActionPhase.IDLE, 0.9),
]
tracks = [
    ObjectTrack(10, table_xy_m=(0.10, 0.20)),
    ObjectTrack(15, table_xy_m=(0.12, 0.22)),
    ObjectTrack(25, table_xy_m=(0.20, 0.30)),
    ObjectTrack(40, table_xy_m=(0.30, 0.20)),
]

task = extract_task(predictions, tracks)
assert task.grasp_frame == 10
assert task.release_frame == 40
assert task.path_xy_m == (
    (0.10, 0.20), (0.12, 0.22), (0.20, 0.30), (0.30, 0.20)
)
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

`carry_trajectory_xy_m` exposes only CARRY-phase observations, maintaining the
distinction between the canonical CARRY trajectory and the full handled interval.
Interior tracking gaps remain visible in frame IDs. No gaps are filled; the
CARRY-only subset can be empty. Downstream path processing must decide whether
the available geometry is sufficient before generating a safe trajectory.

The legacy `TaskRepresentation` type still describes normalized workspace
coordinates. Do not put source or target meters into its normalized fields.

## Required mapping configuration

`configs/retargeting.yaml` defines the default tabletop clone in `mujoco_world`.
The Panda base is the world origin, 0.15 m behind the center of the filmed
table's left/near long edge. Positive X points into the table, positive Y points
left from the robot's perspective, and positive Z points upward. A deployment
with a different scene must override the mapping explicitly rather than silently
reusing this placement.

The sibling `tabletop_clone` section supplies the measured `0.508 m x 0.762 m`
footprint and explicit `0.15 m` robot setback to `add_tabletop_clone`. Its minimal
`0.01 m` thickness extends below the `z=0` top surface and is a simulation
default, not a measured property. The builder adds no robot geometry; it creates
the bounded table in front of a named robot-base marker.

For this default `0.762 m` table depth, the configured mapping reduces to:

```text
x_mujoco = 0.15 + x_table
y_mujoco = 0.381 - y_table
```

Thus table `(0, 0.381)` is its near-edge center at world `(0.15, 0)`, the Panda
base is world `(0, 0)`, and the tabletop occupies `x in [0.15, 0.658]` and
`y in [-0.381, 0.381]`. No Panda workspace dimensions participate in the
conversion.

Required fields:

| Field | Meaning |
|---|---|
| `source_frame` | Must be `table`, matching the extraction contract |
| `target_frame` | A distinct, nonempty target coordinate-frame name |
| `table_origin_target_xy_m` | Location of table `(0, 0)` in target XY meters |
| `table_x_axis_target_xy` | Unit direction of increasing table X in target XY |
| `table_y_axis_target_xy` | Unit direction of increasing table Y in target XY |

For a column-vector table position `p_table_m`, the mapping is:

```text
A = [table_x_axis_target_xy  table_y_axis_target_xy]  # axes are columns
p_target_m = table_origin_target_xy_m + A @ p_table_m
```

This preserves physical distances without a unit conversion at the task boundary.
The XY axis vectors must be unit length and perpendicular. Numerical axis
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

mapping = Config("configs/retargeting.yaml").get("retargeting")
target_task = retarget_task(task, mapping)
mapped_xy_m = target_task.path_xy_m  # all retained samples, one-for-one
```

`RetargetedTask` retains its immutable `source_task`, a named `target_frame`, and
one target XY point per source observation. Frame IDs/phases/confidence are
available through that source task. A mapping does not move the simulated object,
adapt to its current position, or prove robot reachability.

## Path processing and downstream integration

`ExtractedTask.path_xy_m` and `RetargetedTask.path_xy_m` always expose the full
retained geometry. Selection and interpolation happen only after retargeting:

```python
from mimic.robot import process_path

direct = process_path(target_task, {"interpolation": "direct"})
exact = process_path(target_task, {"interpolation": "none"})
corners = process_path(
    target_task,
    {"interpolation": "corners_only", "corner_max_deviation_m": 0.005},
)
cubic = process_path(
    target_task,
    {
        "interpolation": "cubic",
        "corner_max_deviation_m": 0.005,
        "output_spacing_m": 0.01,
        "maximum_spline_deviation_m": 0.01,
    },
)
```

The numerical values above demonstrate the API and are not deployment tuning.
The authoritative project default is the `cubic` configuration in
`configs/default.yaml`; the simulation pipeline repeats those explicit values in
`configs/robot_pipeline.yaml` because pipeline artifacts are validated as a
self-contained execution configuration.
The supported policies are:

| `interpolation` | Geometry |
|---|---|
| `direct` | Exact grasp and release endpoints; the former default behavior |
| `none` | Every mapped sample, unchanged and in source order |
| `corners_only` | Ramer-Douglas-Peucker simplification using a required metre tolerance |
| `cubic` | Natural parametric SciPy cubic through retained corners, sampled by spatial spacing |

`none` means exact following of the available piecewise-linear samples; it cannot
recover unobserved continuous human motion. `corners_only` returns only original
samples, always including the exact endpoints. `cubic` uses cumulative chord
length rather than frame index, so irregular video sampling does not set spline
shape or robot speed. It rejects a zero-length path or a curve exceeding the
configured maximum distance from the observed polyline rather than falling back
to another mode. Arc length and maximum deviation are evaluated on a dense,
deterministic numerical sampling of the spline; this is a validation approximation,
not a proof over every point on the continuous curve.

`ProcessedPath` retains the `RetargetedTask`, interpolation choice, control
points, and their original source indices. Processing never overwrites source or
mapped geometry. Lowercase configuration values are intentional; old `DIRECT`
and `FOLLOW` values fail validation. The old `FOLLOW` behavior maps to `none`.

Path processing does not assign human timing to the robot. `WaypointBuilder`
converts a `ProcessedPath` into the existing `PickPlaceWaypoints` contract only
when the caller supplies every world-Z coordinate, the object-goal Z coordinate,
and a fixed unit `wxyz` tool quaternion:

```python
from mimic.robot import build_waypoints

waypoints = build_waypoints(
    cubic,
    {
        "approach_z_m": 0.20,
        "grasp_z_m": 0.03,
        "lift_z_m": 0.20,
        "transport_z_m": 0.20,
        "lower_z_m": 0.03,
        "retreat_z_m": 0.20,
        "object_goal_z_m": 0.02,
        "tool_quaternion_wxyz": (0.0, 1.0, 0.0, 0.0),
    },
)
```

These values only illustrate the API; they are not deployment calibration.
The builder preserves every processed XY point and never derives height or
orientation from a Panda model. Complete tool poses are subsequently checked
against the selected robot profile by the execution stack. The internal executor
skill label remains `FOLLOW_PATH`; it is execution metadata rather than a
path-processing option. Execution may approximate intermediate path waypoints
under its explicit online-handoff radius; source, retargeted, processed, and
serialized waypoint geometry remain unchanged. Final path arrival remains exact
under the configured measured pose tolerance.

The package exposes extraction/retargeting without importing MuJoCo, Mink, or
Torch. Existing execution exports load their dependencies only when requested.

## Saved task-input integration

`mimic.integration.build_robot_pipeline` connects the committed offline contracts
without rerunning perception:

```json
{
  "schema": "mimic.demo_task_input.v1",
  "video": {
    "created_at": "...",
    "fps": 30.0,
    "frame_count": 140,
    "duration_s": 4.67,
    "tracking_coordinate_frame": "image_pixels",
    "image_width_px": 1920,
    "image_height_px": 1080
  },
  "catalog": { "schema_version": 2, "fingerprint": "...", "labels": ["..."] },
  "checkpoint_sha256": "...",
  "postprocessing": { "fingerprint": "...", "settings": {}, "guard_policy": "..." },
  "resolved_actions": [
    { "frame_idx": 1, "timestamp_s": 0.0, "phase": "IDLE", "confidence": 0.9,
      "decision_source": "model" }
  ],
  "object_tracks": [
    { "frame_idx": 1, "timestamp_s": 0.0,
      "position": { "x": 100.0, "y": 200.0, "confidence": 0.8 } }
  ]
}
```

`position` is `null` when the tracker has no observation. Action and tracking
arrays are independently ordered by the same one-based source-video frame IDs;
they need not contain the same frames.

```text
mimic.demo_task_input.v1 (resolved actions + independent image-pixel tracks)
  + calibration homography
  -> ObjectTrack(table_xy_m)
  -> ExtractedTask
  -> RetargetedTask
  -> ProcessedPath
  -> PickPlaceWaypoints
```

The consolidated artifact stores classifier catalog, checkpoint, and
post-processing provenance once. Its `resolved_actions` array contains exactly
one accepted phase per classifier timestep, while `object_tracks` preserves the
independently sampled tracker stream. The adapter exposes only resolved actions
to task extraction and verifies that calibration table dimensions match the
configured MuJoCo tabletop clone. Missing pixel detections remain missing; exact
GRASP and RELEASE observations are still required by task extraction. With more
than one complete episode, the caller must explicitly select a one-based episode.
Calibration JSON also declares the decoded image width and height. Pixel tracks
outside that frame are rejected with a rotation/resolution error instead of
being extrapolated through the homography. The committed short-demo homography
is expressed in the same `1920 x 1080` landscape frame used by tracking. Derived
action segments and tracking summaries are computed for display only and are not
persisted in the robot handoff.

The command-line entry point writes the existing executor JSON contract:

```bash
uv run mimic-robot-pipeline \
  --task-input results/demo/demo_task_input.json \
  --calibration data/annotations/calibrations.json \
  --retargeting-config configs/retargeting.yaml \
  --pipeline-config configs/robot_pipeline.yaml \
  --waypoints results/demo/demo_world_waypoints.json
```

Supplying `--robot-config` and a `--log` path invokes the existing headless
MuJoCo executor after waypoint generation. `configs/robot_pipeline.yaml` uses the
verified simulation fixture assumptions: a z=0 tabletop, a 4 cm cube centered at
z=0.02 m, 0.17 m clearance poses, and a fixed downward Panda tool orientation.
At reset-only simulation initialization, the named free-joint cube is placed at
the selected task's retargeted grasp pose before any physics or command step.
Those values are explicit fixture settings, not physical-robot calibration or a
general default for differently sized objects.

## Verification

```bash
uv run pytest tests/test_robot/test_task_extractor.py tests/test_robot/test_coordinate_retargeter.py tests/test_robot/test_path_processing.py tests/test_robot/test_waypoint_builder.py
```

Tests use synthetic mappings, not deployment defaults. They check exact onset
matching, invalid input rejection, sampling-rate independence, explicit axis
rotation/reflection, cm-to-m conversion, round trips, immutable source retention,
unset configuration rejection, all four path policies, strict mode-specific
settings, endpoint preservation, deviation rejection, and backend-independent
imports. These checks do not demonstrate simulation success or physical robot
safety.
