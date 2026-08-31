# Robot execution: IK, gripper, and skills

The execution block is implemented with a Mink backend, explicit model bindings,
a standard Menagerie Panda gripper driver, and deterministic skill sequencing.
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
>=3.9 declaration is unchanged. Mink is pinned to 1.3.0; the exact resolved
MuJoCo, DAQP, and qpsolvers versions are recorded in `uv.lock`. Use `--group robot`
when executing robot code. Backend imports do not load Torch or perception models.

The model download is explicit setup, never runtime behavior. It retrieves only
the standard Panda XML, referenced meshes, README, and Apache-2.0 license from
Menagerie revision `da76818e269b82289eba39808e2fb91d679d6994`. The local manifest
records SHA-256 checksums. The approximately 34 MB cache is excluded from Git.
Run downloads into a fresh directory; the downloader refuses overwrites.

## Interfaces and robot replacement

| Component | Responsibility |
| --- | --- |
| `RobotProfile` / `ModelBindings` | Named joints and actuators, tool offset, workspace and speed limits; validate the loaded model. |
| `IKSolver` / `MinkIKSolver` | One differential IK step from measured joint state; return named arm targets, errors, and status. |
| `GripperDriver` | Translate total nominal opening into model-specific actuator controls. |
| `GripperLogic` | Open/close/hold lifecycle and candidate-grasp evidence. |
| `RobotController` | Arm/gripper scheduling and single-use control samples. |
| `RobotIO` / `MuJoCoAdapter` | Detached observations, validated actuator writes, and physics stepping. |
| `SkillExecutor` | Feedback-gated hover, descent, closure, lift, path following, lowering, opening, and retreat. |

Another fixed-base arm with scalar position-controlled joints needs a profile,
a gripper driver/observer if the hand differs, and explicit application wiring.
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
Supply the scene, named object, home keyframe, physical tool offset, documented
velocity limits, solver settings, and grasp/placement acceptance criteria.
The template retains existing 100 Hz arm / 10 Hz gripper settings and existing
workspace bounds. No conflicting height defaults are selected.

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

To execute an explicitly configured scene and processed task:

```sh
uv run --group robot python scripts/simulate_robot.py \
  --config path/to/experiment.yaml \
  --waypoints path/to/world_waypoints.json \
  --log outputs/robot_attempt.jsonl
```

The waypoint JSON fields are `approach`, `grasp`, `lift`, `path` (a nonempty list),
`lower`, `retreat`, and `goal_position`. Each pose contains `position` and
`quaternion_wxyz`; `goal_position` is the object's desired world-space center.
The entry point consumes JSON, not the README's earlier placeholder pickle task.
It exits nonzero on a failed execution and retains JSONL observations and reasons.

## Preserved constraints and failure behavior

- IK owns a separate MuJoCo configuration and freezes all non-arm DOFs. It never
  writes live joint/object coordinates. Only explicit reset initializes them.
- Position, actuator, and per-tick speed limits are checked. Invalid commands
  are rejected, never clipped. Arm/gripper writes are validated together.
- Standard Panda opening is nominally 0–0.08 m. Its driver translates that to
  tendon controls 0–255, after validating transmission, coupling, gains, force
  range, and joint ranges. It deliberately rejects the MJX variant's mapping.
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

## Verification and retained discrepancy

On 2026-08-31, verification used Python 3.12.14, MuJoCo 3.12.0, Mink 1.3.0,
qpsolvers 4.13.0, and DAQP 0.9.1 on macOS ARM64. Focused tests cover model binding,
limit/error rejection, private IK state, Panda forward/inverse kinematics,
actual Panda finger motion, a two-axis arm with reordered actuators and a free
object, independent gripper scheduling, and execution failure gates. Sequencing
fakes are labeled as such and do not prove physical grasp.

Initial implementation checks: 29 execution-specific tests passed; the shared repository suite
passed all 124 tests (including concurrently added geometry tests and three
preexisting placeholder integration tests). Formatting/lint checks on the new
execution modules passed, the uv lockfile checked successfully, and both source
and wheel distributions built. These checks do not establish physical task success.

Run the fixed diagnostic fixture separately:

```sh
uv run --group robot python scripts/verify_panda.py \
  --output outputs/robot_verification/new_attempt
```

It defines a 4 cm, 30 g cube, a z=0 table, and a tool offset derived from the
standard fingertip-pad geometry. Its numerical settings are fixed **test
assumptions**, not production calibration. After the original failure, only the
separate measured-gripper tolerance was changed, with user authorization.

The first attempt stopped in HOVER at simulation time **0.01 s**: finger_joint2
was **0.0400033266 m**, above its **0.04 m** upper bound by about **3.33 micrometers**.
Mink's strict configuration check rejected this state. No grasp, transport, or
release occurred; object-goal error remained approximately **0.10 m**. The trace
and snapshot are in `outputs/robot_verification/initial/`.
The final rerun in `outputs/robot_verification/final/` reproduced the same failure.

Hypothesis: the strict kinematic-state check conflicts with compliant physical
joint limits during motion. MuJoCo documents soft contacts and limits; see its
[constraint model](https://mujoco.readthedocs.io/en/latest/computation/index.html).
A separate unmodified constant-home-control run, without IK, also showed a much
smaller finger overshoot (about 0.63 micrometers). This supports the compliance
hypothesis but does not establish which acceptance policy the project should use.

The user subsequently authorized the bounded measured-state tolerance described
above. With 10 micrometers allowed per finger, the original limit rejection is
resolved: the diagnostic runs for 10 seconds, and its largest measured finger
excursion remains 3.33 micrometers. A comparison of saved metadata confirms the
model, solver/gripper settings, waypoints, rates, and library versions are unchanged.
Both the initial failure and new trace are retained.

The new run stops with **HOVER: target not achieved before timeout**. At 10 seconds,
the tool position error is about **0.39544 m**, and orientation error about
**0.29090 rad**. No grasp, transport, or release occurred; object-goal error remains
approximately 0.10 m. Results, traces, and settings comparison are under
`outputs/robot_verification/measured_tolerance_10um/`.

A private kinematic comparison using the same target and solver settings reaches
the target in 296 iterations, with position error about 4.13 micrometers and
orientation error 1.65e-6 rad. Follow-up experiments confirmed that resetting the
position-servo reference to one bounded step ahead of measured position on every
tick is the cause of the slow/divergent tracking. Zero gravity reproduced the
predicted approximately 0.05 rad/s effective speed; removing Panda contacts and
arm force limits did not alter the failing trace. A persistent measured-feedback
position reference reached the same hover tolerance in 4.60 seconds. No production
control policy or protected parameter was changed. See
[`ARM_TRACKING_INVESTIGATION.md`](ARM_TRACKING_INVESTIGATION.md) for evidence,
downstream failures exposed by the diagnostic controller, and solution options.

After this change, 16 new tolerance-boundary tests pass and the full suite passes
140 tests. They cover both limit sides, out-of-allowance rejection, raw-state
preservation, strict arm/command bounds, unchanged model arrays, and tolerance
isolation by joint name. Production scene geometry and task-success criteria
remain required decisions.

Full physical pick-and-place, prerecorded perception-to-execution integration,
Linux runtime behavior, and hardware execution have **not** been verified.
