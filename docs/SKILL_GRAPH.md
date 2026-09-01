# Modular skill graph and composite handlers

The temporal model and robot share one versioned `SkillCatalog`. Each entry has
one training index and one robot handler ID, so a checkpoint's output order can
be checked against the catalog fingerprint before any state is accepted. The
default catalog in `configs/skills/pick_place.yaml` is:

`IDLE, HOVER, GRASP, CARRY, RELEASE`

The catalog mechanism is generic. A different training feature set may use a
different versioned catalog, graph, and complete handler registry. The default
`ActionPhase` and task extractor remain the specialized pick/place preset.

## Prediction validation

`GraphStatePostProcessor` ranks the full probability mapping deterministically
using catalog order for exact ties. It evaluates the highest-probability label
first. If that graph edge is unavailable, it may evaluate only the runner-up,
and only when the configured top-to-second score gap permits it. It never scans
lower-ranked labels. A transition must also pass the explicit confidence,
margin, persistence, and named guard checks.
Named guards fail closed when the runtime has not supplied an observation-based
guard evaluator.

Every call returns a `StateDecision` containing the previous and accepted skill,
top two labels, selected rank, confidence, graph edge, reason, pending count,
source, frame/time, and catalog fingerprint. A rejected prediction holds the
current skill; it does not invent a graph path.

Missing detection is separate from classifier uncertainty. After the configured
timeout, execution is suspended in `IDLE`; no gripper-open action is generated.
The previous skill is retained as `suspended_from`, allowing a subsequent valid
prediction to resume against the correct graph context. All five post-state
thresholds are deliberately `null` in the committed template and must be chosen
from validation data.

## Default relationship graph

The canonical complete episode is:

`IDLE -> HOVER -> GRASP -> CARRY -> RELEASE -> HOVER -> IDLE`

All skills have explicit self-edges for persistence. Contextual edges define:

- `IDLE -> HOVER` (`TO_GRASP`): move to the task's grasp-hover pose.
- `RELEASE -> HOVER` (`TO_HOME`): move to the saved home joint configuration.
- `GRASP -> HOVER` (`TO_HOME_ABORT`): return home only when `grasp_empty` passes.
- `IDLE`: emit no arm or gripper action, preserving the previous measured state.

Other guards (`grasp_confirmed`, `transport_complete`, and `home_reached`) are
injected by the runtime because they depend on simulation observations. The
postprocessor validates graph intent but does not execute robot motion.

## Skill execution modules

`SkillRegistry` binds every catalog handler ID exactly once. Repeated self-state
decisions do not restart composite actions. The default handlers in
`pick_place_skills.py` emit small low-level action records:

- Cartesian tool-pose motion with a gripper request;
- named full-arm joint-preset motion with a gripper request.

The default handlers expand GRASP into descend/close and CARRY into lift plus
the already processed path. Adding a new composite skill requires a new catalog
version, graph edges, and handler registration; it does not require edits to the
postprocessor.

`SkillRuntime` composes the postprocessor and registry into one call returning
both the auditable decision and its action plan. `RobotController.prepare_action`
dispatches each action by shape, so the catalog and graph remain independent of
Mink, Ruckig, Panda names, and MuJoCo.

`home` is resolved against the active `RobotProfile`, either from a MuJoCo
keyframe or an explicit mapping containing every named arm joint exactly once.
Gripper and object coordinates are excluded. Return-home uses the same Ruckig
velocity, acceleration, jerk, measured-speed, joint-limit, and tracking-error
checks as Cartesian execution, but bypasses IK. Measured per-joint arrival
tolerances remain explicit configuration.

## Deliberately unresolved values

The following are model/design settings and stay `null` in committed templates:

- minimum accepted confidence;
- minimum transition margin over the current state;
- maximum score gap that permits choosing the runner-up;
- required consecutive observations;
- missing-detection timeout;
- saved-preset measured joint-arrival tolerances.

They should be selected from held-out temporal predictions and simulated
execution traces. Controller gains, Panda limits, gripper ranges, and the
existing control rates were not changed for this feature.
