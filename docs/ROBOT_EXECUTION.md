# Robot execution: IK, gripper, and skills

The execution block is implemented with a Mink backend, a Ruckig position-reference
layer, explicit model bindings, a standard Menagerie Panda gripper driver, and
deterministic skill sequencing.
It accepts **already retargeted and processed world-space tool poses**. It does
not estimate object paths, create a coordinate mapping, smooth tracking, infer
task heights, or add learned labels. Task extraction and retargeting can be
developed independently.

## Setup

Use the existing uv workflow from the repository root:

```sh
uv sync --group robot
uv run python scripts/fetch_panda_model.py
uv run --group robot pytest tests/test_robot/
```

The `robot` group requires Python >=3.10 explicitly; the base package's Python
>=3.9 declaration is unchanged. Mink is pinned to 1.3.0 and Ruckig to 0.19.4;
the exact resolved MuJoCo, DAQP, and qpsolvers versions are recorded in `uv.lock`.
Use `--group robot` when executing robot code. Backend imports do not load Torch
or perception models.

The model download is explicit setup, never runtime behavior. It retrieves only
the standard Panda XML, referenced meshes, README, and Apache-2.0 license from
Menagerie revision `da76818e269b82289eba39808e2fb91d679d6994`. The local manifest
records SHA-256 checksums. The approximately 34 MB cache is excluded from Git.
Run downloads into a fresh directory; the downloader refuses overwrites.

## Interfaces and robot replacement

| Component | Responsibility |
| --- | --- |
| `RobotProfile` / `ModelBindings` | Named joints and actuators, tool offset, workspace and speed limits; validate the loaded model. |
| `WaypointBuilder` | Add explicit world-Z geometry and fixed orientation to a processed robot-independent XY path. |
| `IKSolver` / `MinkIKSolver` | One differential IK step from measured joint state; return named arm targets, errors, and status. |
| `RuckigPositionIK` | Convert geometric IK results into persistent, jerk-limited position-servo references and monitor measured speed/tracking lag. |
| `GripperDriver` | Translate total nominal opening into model-specific actuator controls. |
| `GripperLogic` | Open/close/hold lifecycle and candidate-grasp evidence. |
| `RobotController` | Arm/gripper scheduling and single-use control samples. |
| `RobotIO` / `MuJoCoAdapter` | Detached observations, validated actuator writes, and physics stepping. |
| `SkillExecutor` | Feedback-gated hover, descent, closure, lift, path following, guarded placement, opening, and retreat. |

Another fixed-base arm with scalar position-controlled joints needs a profile,
per-joint trajectory limits, a gripper driver/observer if the hand differs, and
explicit application wiring.
The interfaces do not assume seven joints, contiguous indices, or equal position
and velocity vector dimensions. Torque actuation requires a different controller
adapter; it is rejected rather than treated as a position command. The factory
currently wires Panda explicitly; there is no implicit fallback or plugin registry.

`ToolPose` uses MuJoCo **world coordinates in meters** and unit quaternions in
**(w, x, y, z)** order. `tool_offset` is the body-to-tool transform. The IK backend
converts the requested tool pose to its tracked body pose using that transform.
No hand-origin/grasp-center equivalence is assumed. `PickPlaceWaypoints` supplies
each subskill's tool pose plus the intended object-center goal. Its poses must
share one orientation. The caller must define and validate that downward
orientation, including yaw, for its scene.

The existing `RobotCommand` constructor is unchanged. `command_target()` requires
the caller to declare its quaternion order and a fixed orientation for omitted
orientations. Positions must already be world meters. Duration and path sampling
remain upstream responsibilities.

## Configuration and execution

`configs/robots/panda.yaml` is a contract template, **not a calibrated runnable
experiment**. Required but unresolved fields are null and fail before execution.
Supply the scene, named object, reset keyframe, saved home preset, physical tool
offset, solver settings, and grasp/placement acceptance criteria. A home preset
uses either a MuJoCo keyframe or an explicit full mapping of named arm joints;
its measured per-joint arrival tolerances are also required.
The template retains existing 100 Hz arm / 10 Hz gripper settings and existing
workspace bounds. No conflicting height defaults are selected.

`models/panda_pick_place_scene.xml` is the checked-in simulation scene wrapper
used by `configs/robots/panda_complete.yaml`. After the one-time pinned asset
download, it loads the unmodified Menagerie Panda, the approved left-edge
tabletop clone with its near edge 0.15 m in front of the Panda base, and a 4 cm,
30 g free-joint cube. The scene-specific `pick_place_home` keyframe includes all
Panda, finger, and cube coordinates. Before physics starts, the simulation-only
executor replaces the cube's keyframe translation with the selected task's
retargeted grasp pose and clears its free-joint velocity. Mid-execution object
teleportation is rejected.
The complete simulation config uses Mink's supported full-correction
`task_gain: 1.0`, matching the verified fixed Panda fixture; it is not an actuator
gain and remains subject to the configured IK and trajectory limits.
The complete config still uses `orientation_cost: 0.1` and
`posture_cost: 0.01`. A read-only diagnostic on the corrected/setback grasp hover
showed that combination settling outside the configured pose tolerances after
2,000 steps, while either disabling the posture objective or raising orientation
cost to `1.0` reached the same pose. The separately verified fixture uses both
`posture_cost: 0` and `orientation_cost: 1`. These objective weights remain an
explicit experiment decision; tolerances and planning bounds were not weakened.
The same config uses the verified fixture's gripper completion criteria:
`0.001 m` width tolerance, `0.002 m` empty-grasp threshold, `0.1 N` contact force,
`0.1 s` contact duration, and `2.0 s` movement timeout. These remain
simulation-fixture settings rather than validated physical-object thresholds.

`measured_gripper_joint_tolerance_m: 0.00001` is a user-authorized allowance for
each named Panda finger's **measured position** at either joint limit. This
10-micrometer value is an engineering allowance (about three times the initial
3.33-micrometer excursion), not a derived physical constant. It replaces Mink's
1-micrometer observation allowance for those fingers only; it is not added to it.
Other state checks retain their previous behavior, and arm/actuator command checks
remain exact. The library API accepts a per-joint mapping through
`MinkIKSolver(..., measured_gripper_tolerances_m={...})`; omitting the mapping
preserves the prior strict behavior. Tolerances cannot be assigned to arm joints,
non-slide joints, or joints that move the tracked tool/arm.

Raw measured coordinates are retained in both logs and the private solver state.
The named fingers still have zero-velocity equality constraints. Their position
inequality rows are omitted from the arm solve, because an accepted out-of-range
measurement cannot simultaneously be frozen and commanded back inside its limit.
They are validated separately before every solve. No model arrays, actuator limits,
gripper target widths, or arm inequalities are widened.

Load configuration through the existing `Config` class; its local override behavior
still applies. The factory logs the effective robot configuration. Relative model
paths are resolved against the configuration file, not the working directory.
Arm intervals must be integer multiples of the scene's unchanged simulation step,
and arm frequency must be an integer multiple of gripper frequency.

For the Panda profile, `velocity_limits` is the deliberately lower 0.5 rad/s
simulation operating envelope. `trajectory.hardware_velocity_limits`,
`acceleration_limits`, and `jerk_limits` record the manufacturer FER maxima.
Ruckig applies the operating velocity plus the sourced acceleration and jerk
limits to the command reference. The per-joint tracking-error bounds stop a
stalled or poorly tracking servo before reference wind-up. These arrays are
ordered by `profile.arm_joints`, so a different robot can provide its own count,
names, units, and sourced limits without changing the controller.

During `FOLLOW_PATH`, the complete Panda simulation config uses online handoff
within a measured `0.03 m` Cartesian radius for every intermediate waypoint.
Intermediate points do not require `AT_TARGET` or zero velocity: the executor
commits the current Ruckig tick, then replaces the target while preserving its
position, velocity, and acceleration reference. The final path waypoint still
requires measured `AT_TARGET` arrival. Contact, slip, timeout, joint-limit, and
tracking-error gates remain active through every handoff. Each handoff is logged
with its waypoint index, measured distance, radius, and IK status.

Release uses a separate guarded-placement primitive. `PLACE_APPROACH` first moves
to the configured clearance above the final tool pose. `LOWER` then advances an
online Cartesian Z reference while holding the goal XY and fixed orientation.
The configured descent acceleration ramps the reference-speed request; measured
downward TCP speed consumes the remaining speed budget, and execution fails
closed if measured descent exceeds the configured maximum. Intermediate descent
references do not require zero velocity or `AT_TARGET`.

The simulation adapter supplies an active-contact predicate between the named
object subtree and the configured `support_geom`. On first object-support contact,
the executor holds the measured tool pose. Release begins only after support and
object-speed stability persist for `placement_contact_confirmation_s` and the
measured object position is inside `placement_tolerance_m`. The airborne
object/tool slip and finger-contact-loss gates remain active through
`PLACE_APPROACH` and free-space `LOWER`; after support onset, expected table
constraint motion is evaluated by the placement gate instead of the airborne
rigid-offset rule. Contact onset, loss, confirmation, commanded pose, and measured
descent speed are retained in the execution log.

For the checked-in cube scene, `configs/robots/panda_complete.yaml` uses a
`0.015 m` approach clearance, `0.05 m/s` maximum measured descent,
`0.5 m/s^2` reference acceleration, and `0.05 s` support confirmation. These are
explicit simulation-fixture settings, not Panda constants or hardware-safe
values. The generic Panda template leaves them unresolved.

Contextual `HOVER` return-home actions bypass Cartesian IK and send the saved
joint configuration through the same persistent Ruckig reference, measured-speed
checks, tracking-error bound, and model limits. Only named arm joints are read
from a keyframe; gripper and object coordinates are never part of the preset.
Normal return-home does not reset or teleport simulation and uses gripper HOLD.

To execute an explicitly configured scene and processed task:

```sh
uv run --group robot python scripts/simulate_robot.py \
  --config path/to/experiment.yaml \
  --waypoints path/to/world_waypoints.json \
  --log outputs/robot_attempt.jsonl
```

Add `--viewer` for a live, approximately real-time display. On macOS, passive
MuJoCo rendering must run under `mjpython`:

```sh
uv run --group robot mjpython scripts/simulate_robot.py \
  --config configs/robots/panda_complete.yaml \
  --waypoints results/short_demo/IMG_2067_world_waypoints.json \
  --log results/short_demo/IMG_2067_viewed_execution.jsonl \
  --viewer
```

The viewer observes the same validated physics steps as headless execution; it
does not introduce a second physics loop or change simulation/control rates. The
runner holds the final frame until the window is closed. Reusing a `--log` path
replaces that file's previous contents before the new attempt is recorded.

The integrated saved-results command also accepts `--viewer`. It selects
`mjpython` for the simulation child process automatically on macOS.

Both `scripts/simulate_robot.py` and `mimic-robot-pipeline` accept
`--video-out PATH` to record the simulation as MP4. Omitting only the PATH (for
example, `--video-out`) writes `mimic-simulation-<current-time>.mp4` in the
working directory. Recording samples completed validated physics intervals at
30 FPS; it does not step or pace the simulation. Existing video files are not
overwritten. On macOS, offscreen recording also requires `mjpython`; the
integrated pipeline selects it automatically.

The waypoint JSON fields are `approach`, `grasp`, `lift`, `path` (a nonempty list),
`lower`, `retreat`, and `goal_position`. Each pose contains `position` and
`quaternion_wxyz`; `goal_position` is the object's desired world-space center.
The entry point consumes validated waypoint JSON. It exits nonzero on a failed
execution and retains JSONL observations and reasons.

## Preserved constraints and failure behavior

- IK owns a separate MuJoCo configuration and freezes all non-arm DOFs. It never
  writes live joint/object coordinates. Only explicit reset initializes them.
- Position, actuator, and per-tick speed limits are checked. DAQP's primal
  feasibility tolerance is aligned with the explicit per-tick postcondition;
  invalid commands are rejected, never clipped. Arm/gripper writes are validated
  together.
- Standard Panda opening is nominally 0–0.08 m. Its driver preserves the full
  mapping to tendon controls 0–255, after validating transmission, coupling,
  gains, force range, and joint ranges. Normal OPEN actions use the configured
  0.0799 m command so physics does not continually push against the hard stop.
  The driver deliberately rejects the MJX variant's mapping.
- Closure alone is not grasp success. Both fingers must contact the target
  object with configured force/duration and nonempty opening. Lift confirms
  grasp; relative object/tool motion and contacts monitor transport. Release
  requires opening, detachment, and settling before placement is evaluated.
- On failure, the attempt stops stepping and records its final observation.
  It does not automatically retry, open the gripper, weld the object, adjust
  gains, or alter model limits. No hardware interface is included.
- Physical collisions remain enabled. No collision-avoidance planner is included;
  IK convergence is not a collision-free or dynamically feasible path certificate.

Mink 1.3.0's `SO3.log()` multiplies by `sign(w)`, which can erase an exact half-turn
when `w == 0`. Arrival is therefore measured independently using MuJoCo's
`mju_subQuat`; a degenerate frame-task linearization is rejected. Otherwise a
nonzero error can yield a valid but unproductive local IK step; execution times
out instead of treating that as arrival or proving global unreachability.

## Verification and resolved discrepancy

On 2026-08-31, the final fixture used Python 3.12.14, MuJoCo 3.12.0,
Mink 1.3.0, Ruckig 0.19.4, qpsolvers 4.13.0, and DAQP 0.9.1 on macOS
ARM64. The checked-in standard Panda model remains the pinned Menagerie asset.
No actuator gain, force range, joint range, simulation step, collision setting,
or IK objective was changed.

Run the fixed diagnostic fixture separately:

```sh
uv run --group robot python scripts/verify_panda.py \
  --output outputs/robot_verification/new_attempt
```

It defines a 4 cm, 30 g cube, a z=0 table, and a tool offset derived from the
standard fingertip-pad geometry. Its numerical settings are fixture assumptions,
not a general object calibration.

The original run had two independent failures:

1. Compliant finger motion exceeded Mink's default one-micrometer observation
   allowance by about 3.33 micrometers. The approved 10-micrometer measured-state
   allowance fixes that check without changing any model or command limit.
2. Differential IK was re-seeded from measurement and its one-step result was sent
   directly to a position servo. The resulting reference stayed only 0.005 rad
   ahead of the arm, producing about 0.05 rad/s rather than a persistent 0.5 rad/s
   trajectory. Gravity, collision, and actuator-force experiments excluded those
   as the primary cause.

The implemented `RuckigPositionIK` layer plans a persistent joint reference,
retains measured Cartesian feedback for arrival, applies explicit velocity,
acceleration, and jerk constraints, and fails closed on sourced measured-speed,
joint-position, or per-joint tracking-error violations. At a waypoint transition,
DAQP's default numerical feasibility tolerance admitted a 0.005000232 rad step
against a 0.005 rad bound. Its primal tolerance is now aligned to the existing
postcondition; the configured speed bound was not widened.

The fixture also exposed and now verifies three gripper/execution changes:

- OPEN uses 0.0799 m total width, 0.1 mm inside the model's 0.08 m hard stop.
  The full driver range remains available and unchanged.
- A successfully completed OPEN or CLOSE action is latched, so an old movement
  timer cannot later become a false timeout during HOLD. Contact-loss and slip
  checks still govern transport.
- The fixture's slip threshold is 0.015 m, retaining the earlier 0.010 m value
  in source history and comments. The final run observed 2.22 mm during lift,
  4.50 mm during path following, and 10.07 mm during lowering.

The successful trace is in
`outputs/robot_verification/ruckig_gripper_resolution_3/`. It completed grasp,
transport, release, and retreat in 19.23 simulated seconds. Final object-position
error was 4.60 mm against the explicit 10 mm fixture acceptance threshold. The
largest joint-reference lag was 0.0591 rad against the 0.1 rad fail-stop bound;
largest measured joint speed was 0.5031 rad/s, below the sourced Panda hardware
limits. No finger crossed its 0.04 m model limit.

Focused tests verify the Ruckig layer on a different two-axis, position-actuated
robot with different joint names and prismatic units. They cover target arrival,
command velocity/acceleration/jerk bounds, fixed timing, hardware ceilings,
tracking-lag stop behavior, measured-speed rejection, and settings dimension
validation. Gripper tests cover the safe OPEN target and completion latch.

The general `configs/robots/panda.yaml` remains a fail-fast contract template.
It deliberately does not turn this one cube fixture into a claimed production
calibration. A real runnable configuration still needs its scene/tool geometry,
IK objectives, object-specific grasp evidence, and task success criteria.

Not verified: Linux runtime behavior, prerecorded perception-to-execution
integration, other object shapes/masses/friction values, collision-aware global
planning, or any physical robot. Simulation success is not hardware evidence.

## Remaining decisions

| Rating | Decision | Why it remains |
| --- | --- | --- |
| Critical | Deployment scene, object body/home state, physical tool center, downward yaw, and world/table mapping | These define the task geometry. The standard robot model supplies link geometry, but it cannot choose application frames or the intended grasp point. The general factory deliberately fails while they are null. |
| Critical | IK objectives and acceptance criteria in the deployment configuration | Pose costs/tolerances, contact evidence, lift/slip/loss/settling checks, and placement tolerance define success for the intended object set. Fixture values only validate one cube. |
| Critical | Deployment mapping and waypoint policy | The checked-in cube fixture supplies explicit values, but another scene or object requires separately validated mapping, heights, orientation, and sampling. |
| Critical | Validation of state post-processing settings | The committed preset is explicit and runnable; held-out temporal predictions must still justify its confidence, margin, persistence, and missing-detection values. |
| Tunable later | Home-preset measured joint-arrival tolerances | The target joint configuration is explicit, but arrival tolerances should be selected for the active robot model and servo tracking behavior. |
| Tunable later | 0.5 rad/s operating envelope, 0.1 rad tracking bound, and 2000-step private planning bound | They are explicit, bounded, and successful in the fixed fixture. Broader workspace trials can optimize them without changing subsystem semantics. |
| Tunable later | 0.0799 m OPEN target, 10-micrometer measured-finger allowance, and 15 mm fixture slip bound | They are measured simulation policies rather than manufacturer constants. Recheck them when the gripper model or intended object set changes. |
| Tunable later | Collision-aware/global planning | The project contract leaves it optional for the constrained tabletop MVP. It becomes critical if obstacles or wider workspaces are introduced. |
