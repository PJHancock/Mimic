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
margin, and persistence checks. Named guards carry an explicit scope. The
postprocessor defaults to failing named guards closed when no observation-based
evaluator is supplied; offline classifier export deliberately defers `runtime`
guards to robot execution and records that policy in artifact provenance.

Every call returns a `StateDecision` containing the previous and accepted skill,
top two labels, selected rank, confidence, graph edge, reason, pending count,
source, frame/time, and catalog fingerprint. A rejected prediction holds the
current skill; it does not invent a graph path.

Missing detection is separate from classifier uncertainty. After the configured
timeout, execution is suspended in `IDLE`; no gripper-open action is generated.
The previous skill is retained as `suspended_from`, allowing a subsequent valid
prediction to resume against the correct graph context. The committed
pick/place preset supplies explicit post-state defaults; these remain
model/design settings and are not execution feedback.

## Default relationship graph

The canonical complete episode is:

`IDLE -> HOVER -> GRASP -> CARRY -> RELEASE -> HOVER -> IDLE`

`IDLE` is also a legal terminal transition from every state. A successfully
observed pick/place may therefore end `RELEASE -> IDLE` when the classifier has
no terminal HOVER observation. Earlier transitions to IDLE terminate/abort the
state stream and do not turn an incomplete manipulation into an executable
task.

After one complete episode, playback also supports the demonstrated continuation
`RELEASE -> IDLE -> GRASP -> CARRY -> RELEASE`. The graph exposes
`IDLE -> GRASP` as `CONTINUATION_REGRASP`, while task extraction scopes the
hoverless start to later episodes; the first episode still requires HOVER.

All skills have explicit self-edges for persistence. Contextual edges define:

- `IDLE -> HOVER` (`TO_GRASP`): move to the task's grasp-hover pose.
- `IDLE -> GRASP` (`CONTINUATION_REGRASP`): begin a re-grasp only when task
  extraction has already accepted a preceding complete episode. This edge
  records a missed classifier HOVER; the grasp handler still emits the
  approach pose before descend/close rather than starting at the grasp height.
- `RELEASE -> HOVER` (`TO_HOME`): move to the saved home joint configuration.
- `GRASP -> HOVER` (`TO_HOME_ABORT`): return home only when `grasp_empty` passes.
- `IDLE`: emit no arm or gripper action, preserving the previous measured state.

Playback through `SkillExecutor` always expands each episode with a hover
primitive generated from that episode's approach waypoint, including a
hoverless continuation. The hover pose is realized on a home-seeded IK branch
so consecutive episodes can keep the live arm and object state without a
return-to-home detour. Classifier HOVER remains the observed approach label
when the model supplies it.

Runtime-scoped guards (`grasp_confirmed`, `transport_complete`, and
`grasp_empty`) depend on simulation observations. Offline export preserves the
semantic transition and the deterministic executor independently fails closed
if its measured grasp, contact, transport, or arrival criteria are not met. The
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

## Explicit experiment values

The following remain model/design settings even when a committed experiment
configuration supplies explicit values:

- minimum accepted confidence;
- minimum transition margin over the current state;
- maximum score gap that permits choosing the runner-up;
- required consecutive observations;
- missing-detection timeout;
- saved-preset measured joint-arrival tolerances.

They should be selected from held-out temporal predictions and simulated
execution traces. Controller gains, Panda limits, gripper ranges, and the
existing control rates were not changed for this feature.

## Inference and robot result boundary

The action classifier exposes the complete softmax matrix. The inference runner
maps its columns through the active `SkillCatalog` into `SkillPrediction`
records, then calls `GraphStatePostProcessor` in timestamp order. Graph logic is
not embedded in the neural classifier.

The classifier-only runner can produce two deliberately different JSON contracts:

- `mimic.skill_scores.v2` is a diagnostic artifact. Every valid frame contains
  `frame_idx`, `timestamp_s`, `detection_valid`, and a complete `state_scores`
  mapping. It records the catalog and checkpoint fingerprints.
- `mimic.robot_actions.v1` is the narrow action-only artifact. Every
  frame contains exactly one post-processed `phase`, its corresponding model
  confidence when available, and the decision source. It contains no competing
  scores or raw classifier label.

The full video pipeline embeds that same narrow frame structure as
`resolved_actions` inside `mimic.demo_task_input.v1`, alongside an independently
sampled `object_tracks` stream and provenance stored once. This consolidated
artifact is the persisted input accepted by waypoint generation and simulation;
the adapter never exposes score distributions to the robot framework.

`mimic.integration.load_robot_actions` rejects diagnostic score files and old
top-one-only result formats. `load_task_actions` applies that same narrow adapter
to the consolidated artifact. This makes it impossible for task extraction to
accidentally bypass graph post-processing. Existing legacy JSON cannot be
converted to the score schema because its unreported probability mass is lost;
inference must be rerun.

Classifier checkpoint v2 embeds the ordered labels and catalog fingerprint.
Robot-facing inference requires that metadata and rejects a checkpoint trained
against another catalog. The robot artifact also fingerprints the graph and
post-state settings. Offline inference records
`defer_runtime_guards_to_execution`; direct/runtime postprocessor construction
retains fail-closed behavior unless an explicit observation evaluator is
provided.
