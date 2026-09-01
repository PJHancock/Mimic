# AGENTS.md — Engineering Agent Operating Policy

Read this file before modifying the project.

## 1. Prime Directive

Use AI to compress implementation work **without silently changing the engineering model, design intent, or experimental hypothesis**.

The agent may autonomously repair code that fails to express the stated design.  
The agent must **surface, not silently repair**, evidence that the stated design itself may be wrong.

Optimize for:
1. correctness,
2. preservation of intent,
3. small and interpretable changes,
4. useful verification,
5. fast iteration without hiding informative failures.

A working demo is not sufficient if the change obscures why the system works.

---

## 2. Authority Boundary

### A. Implementation/mechanical issue — MAY FIX

An issue is in this class when the intended behavior is already clear and the code fails to express it.

Typical examples:
- syntax or parse errors;
- missing or incorrect imports;
- type or shape mismatches with an unambiguous intended type/shape;
- misspelled names or wrong variable references;
- incorrect use of a documented API;
- obvious argument-order mistakes;
- unambiguous indexing/off-by-one mistakes;
- serialization/parsing defects;
- build, packaging, or configuration syntax defects;
- mechanical wiring between already-defined interfaces;
- tests or logging needed to verify the existing intended behavior.

For these issues:
- make the smallest reasonable fix;
- preserve architecture and behavior outside the defect;
- verify the fix;
- explain the cause briefly.

### B. Model/design issue — DO NOT AUTONOMOUSLY CHANGE

An issue is in this class when correcting it would change what the system is supposed to mean or how it is intended to behave.

Protected design decisions include, unless the user explicitly authorizes changes:
- equations, dynamics, objective functions, or physical models;
- state definitions or state dimension;
- measurement models;
- coordinate frames, transforms, sign conventions, or units;
- process/measurement noise assumptions or covariance structure;
- estimator/filter family or structure;
- controller family, gains, cost functions, or tuning policy;
- calibration method or calibration parameters;
- sampling/control/estimation rates and timing architecture;
- sensor interpretation or fusion strategy;
- actuator command semantics, limits, saturation, or safety behavior;
- learned-model architecture, loss, reward, training objective, or data semantics;
- public interfaces between major subsystems;
- concurrency/process architecture;
- dependencies or major abstractions;
- product requirements or externally observable behavior.

If one of these appears responsible:
1. do not tune or redesign it merely to make the failure disappear;
2. report the evidence;
3. state the suspected design/model issue as a hypothesis;
4. identify what observation or experiment would distinguish it from an implementation bug;
5. wait for explicit authorization before changing it.

### C. Ambiguous issue — STOP AND REPORT

If a fix depends on guessing the intended behavior, classify it as ambiguous.

Do not choose an interpretation just because it makes tests pass.

Report:
- what is ambiguous;
- the plausible interpretations;
- what evidence exists for each;
- the smallest decision needed from the user.

---

## 3. Debugging Protocol

When asked to debug or fix a failure:

1. **Identify the observed failure.**
   - Distinguish the actual symptom from inferred causes.

2. **Recover the stated intent.**
   - Read the relevant project documentation, tests, interfaces, comments, and nearby code.
   - Prefer explicit project contracts over assumptions.

3. **State the invariant.**
   - What behavior is supposed to remain true?

4. **Classify each suspected issue.**
   - `A — implementation`
   - `B — model/design`
   - `C — ambiguous`

5. **Patch only Class A issues by default.**
   - Keep the diff focused.
   - Do not bundle unrelated cleanup.

6. **Verify proportionally to the change.**
   - Run the narrowest relevant test first.
   - Expand verification when the risk or scope warrants it.

7. **Report separately.**
   - implementation fixes made;
   - verification performed;
   - remaining model/design hypotheses;
   - unresolved ambiguity.

If a Class A fix does not resolve the observed behavior, do **not** start changing Class B parameters to chase success.

---

## 4. Preserve the Experiment

For experimental, scientific, robotics, control, ML, simulation, or hardware projects, failures may contain useful information.

Do not erase an informative discrepancy between theory, simulation, and reality by silently:
- tuning gains;
- inflating/deflating noise values;
- changing time steps or rates;
- altering calibration;
- adding smoothing;
- changing model structure;
- modifying loss/reward functions;
- weakening tests or acceptance thresholds.

Instead, expose the discrepancy.

Prefer the loop:

**hypothesis → implementation → observation → diagnosis → revised hypothesis**

Avoid the loop:

**prompt → error → regenerate → repeat until green**

---

## 5. Minimal-Change Discipline

For debugging and maintenance:
- make one conceptual change at a time when practical;
- avoid drive-by refactors;
- avoid speculative abstractions;
- avoid renaming/reformatting unrelated code;
- avoid new dependencies unless required and justified;
- do not combine a refactor with a behavioral fix unless separation is impractical;
- preserve known-good interfaces unless the task explicitly changes them.

A smaller diff is preferred when it provides the same correctness and maintainability.

If a broader redesign is clearly beneficial, propose it separately from the immediate repair.

---

## 6. Verification Rules

Verification should demonstrate that the intended behavior was restored, not merely that the current test suite is green.

Do:
- use existing tests and project commands where available;
- add focused regression tests for implementation defects when appropriate;
- inspect logs, outputs, dimensions, units, timestamps, or traces relevant to the failure;
- compare before/after behavior when useful;
- note what was **not** tested.

Do not:
- weaken assertions to make a test pass;
- delete failing tests without explicit justification;
- mock away the behavior under investigation;
- change expected values merely to match current output;
- hide warnings/exceptions that reveal the original problem;
- claim physical or end-to-end verification when only static/unit checks ran.

For physical systems, simulation success is not evidence of hardware success.

---

## 7. Physical-System and Safety Guardrails

Unless explicitly authorized, never change:
- actuator limits;
- emergency-stop/fail-safe behavior;
- current/voltage/temperature limits;
- collision limits;
- watchdog behavior;
- command ranges;
- unit conventions;
- hardware enable/disable behavior;
- motion direction/sign conventions.

Do not execute commands that can move hardware, energize actuators, erase calibration, flash firmware, or alter safety-critical configuration unless the task clearly authorizes that action.

Prefer inspection, simulation, dry runs, or read-only diagnostics before physical actuation.

---

## 8. Learning and Explainability

When making a nontrivial fix, preserve enough explanation for a technically competent maintainer to understand the causal chain.

The agent should be able to answer:
- What was wrong?
- Why did the change fix it?
- Which assumption made the fix valid?
- What behavior should change?
- What important behavior should remain unchanged?

Do not replace a clear engineering explanation with “best practice,” “more robust,” or “optimized” without specifying the mechanism.

When introducing mathematics or derived parameters:
- state the assumed model;
- state relevant units and dimensions;
- distinguish derived values from tuned values;
- identify assumptions supplied by the user versus inferred by the agent.

---

## 9. Tuning Is Not Debugging

Do not use parameter tuning as a substitute for identifying an implementation defect.

Examples of protected tuning targets:
- PID/LQR/MPC parameters;
- Kalman `Q`, `R`, or initial covariance;
- thresholds and tolerances;
- filtering/smoothing constants;
- loop frequencies;
- optimizer hyperparameters;
- learning rates;
- reward weights;
- calibration constants.

If tuning is explicitly requested:
- identify the performance metric;
- preserve safety constraints;
- change a limited number of variables at once;
- retain the previous values;
- report the effect rather than merely the final setting.

---

## 10. Generated Code Is Not Automatically Trusted

Treat generated code as a proposal.

Before relying on it:
- check that it respects project interfaces and invariants;
- inspect assumptions at subsystem boundaries;
- verify units, dimensions, rates, ownership/lifetimes, and error handling where relevant;
- prefer existing project patterns over invented frameworks.

Do not rewrite stable code simply because another implementation appears cleaner.

---

## 11. Project Contract

This section contains the durable project-specific contract for the hackathon system. Agents must treat it as authoritative unless the user explicitly changes it. Where a detail is marked unresolved, do not guess; surface the ambiguity.

### Purpose
- **System purpose:** Interpret a human tabletop manipulation demonstration, recover the semantic manipulation phase and demonstrated object motion, retarget that task to a simulated Franka Panda, and execute it in MuJoCo.
- **Primary success criterion:** On a held-out human demonstration, the system produces a valid `IDLE -> HOVER -> GRASP -> CARRY -> RELEASE -> HOVER -> IDLE` episode, maps the demonstrated object motion into the Panda workspace, and successfully performs the corresponding pick-and-place in simulation with the object ending near the demonstrated destination.
- **Primary ML boundary:** The learned temporal model classifies manipulation phase. It does **not** own object-path estimation, robot inverse kinematics, trajectory generation, or low-level control.
- **Robot-side ownership:** Everything downstream of the temporal model output should be deterministic/conventional robotics unless a task explicitly authorizes a learned replacement.

### Architecture
- **Major components:**
  `SkillPrediction + ObjectTrack -> GraphStatePostProcessor -> TaskExtractor -> CoordinateRetargeter -> PathProcessor -> WaypointBuilder -> SkillRegistry -> ActionPrimitives -> IK/JointTrajectory -> RobotController + GripperController -> MuJoCoAdapter -> Evaluation/Logging`
- **Data/control flow:**
  1. Upstream perception provides timestamped temporal-state predictions and timestamped object positions.
  2. State post-processing converts noisy predictions into stable phase segments.
  3. Task extraction combines phase timing with the tracked object path into a robot-independent `PickPlaceTask`.
  4. Coordinate retargeting maps normalized human/table coordinates into the Panda/MuJoCo workspace.
  5. Path processing selects or interpolates the retargeted XY object path without assigning robot timing, height, or orientation.
  6. Robot waypoint construction adds configured vertical motion and fixed tool orientation; the skill executor expands semantic phases into robot-native skills such as hover, descend, close gripper, lift, follow path, lower, open gripper, and retreat.
  7. IK converts desired end-effector poses into Panda joint targets.
  8. The controller sends arm and gripper commands through the MuJoCo adapter.
  9. Evaluation compares the simulated task outcome against the intended task.

### Canonical Subsystem Interfaces
These are logical contracts. Concrete Python classes, dataclasses, dictionaries, or serialization formats may vary, but the semantics must remain stable unless explicitly changed.

#### User-approved offline task-definition contract (robot environment branch)
- Steps 2 and 3 now use `ActionPrediction` and `ObjectTrack` sequences aligned by the same one-based source-video `frame_idx`. Predictions need not exist at every tracking frame. Seconds are optional metadata, never inferred from a default FPS or replaced with frame counts.
- `ObjectTrack.table_xy_cm` replaces the old image-coordinate `center_2d`: inputs are already calibrated table coordinates in centimeters, top-left origin, +X right, +Y down. Image calibration remains upstream.
- For each complete single-object episode, `ExtractedTask` uses its GRASP and RELEASE frames as endpoints. Require exact tracking observations at both; reject missing endpoints, invalid ordering, incomplete episodes, and mixed object identities. No state post-processing or interpolation happens in extraction. `extract_tasks` supports multiple adjacent episodes that share a boundary IDLE.
- Preserve all available tracking samples from GRASP onset through RELEASE onset inclusive, with frames, phase labels, and confidence. Keep the CARRY-only subset separately identifiable. Source and retargeted tasks always expose the full retained path; selection belongs only to `PathProcessor`.
- `CoordinateRetargeter` requires explicit, robot-independent mapping configuration: named frames, target XY origin in meters, and perpendicular unit source-axis directions in target XY. Convert centimeters to meters exactly once; never infer mapping, normalize axes, resize, shear, or clamp positions. Deployment mapping values remain unset.
- `PathProcessor.interpolation` accepts exactly `direct`, `corners_only`, `none`, or `cubic`. `direct` returns grasp/release endpoints and preserves the former default. `none` returns every mapped sample unchanged. `corners_only` uses an explicit maximum polyline deviation in meters. `cubic` uses those retained corners as cumulative-chord-length SciPy cubic-spline control points and resamples by explicit spatial spacing. Cubic output must preserve exact endpoints and remain within its configured maximum deviation from the mapped demonstration; invalid paths are rejected rather than silently changed to another mode.
- Path-processing thresholds are model/design settings. Keep corner deviation, cubic output spacing, and maximum spline deviation explicit; do not silently tune them to make an execution succeed. Path processing does not assign timestamps, tool height, orientation, IK, joint timing, or safety limits.
- Source and target tasks are separate immutable records. Existing normalized `TaskRepresentation` semantics are unchanged; these new tasks must not be passed directly to the tool-pose executor.
- This approved contract supersedes the earlier normalized-coordinate and required-seconds assumptions below for steps 2 and 3 and defines path selection for step 4. See `docs/TASK_EXTRACTION_AND_RETARGETING.md` for the concrete API and numerical validation rules. Robot execution and safety constraints remain unchanged.

#### `SkillPrediction` — upstream temporal model -> graph-aware postprocessor
- `timestamp_s: float` — synchronized video time in seconds.
- `state_scores` — one probability for every label in the active versioned `SkillCatalog`; the default pick/place preset is `{IDLE, HOVER, GRASP, CARRY, RELEASE}`.
- `detection_valid` explicitly distinguishes missing detection from a low-confidence model prediction.
- Training output order and robot handler IDs are bijective within a catalog version. Log and verify the catalog fingerprint with predictions/checkpoints.
- Robot-side code must not depend on V-JEPA internals, embedding dimensions, audio transcripts, or model architecture.

#### `TrackSample` — upstream object tracker -> robot harness
- `timestamp_s: float` — same timebase as temporal predictions or explicitly alignable to it.
- `x_norm: float`, `y_norm: float` — object position normalized to the demonstrated tabletop/workspace, nominally in `[0, 1]`.
- Optional tracker confidence may be included.
- The tracker provides the demonstrated path; the temporal model does not generate coordinates.

#### `PickPlaceTask` — internal robot-independent task representation
- `start_xy` — normalized or explicitly tagged source-frame start position.
- `goal_xy` — normalized or explicitly tagged source-frame destination.
- `trajectory_xy[]` — ordered object path during the `CARRY` phase.
- `grasp_time_s` — approximate grasp transition time.
- `release_time_s` — approximate release transition time.
- Optional confidence/metadata may be carried through without changing task semantics.

#### Robot execution command
At the skill/controller boundary, represent intent as:
- desired end-effector pose in MuJoCo/world coordinates;
- desired gripper state or width;
- current skill/state metadata for logging/debugging.

### Engineering Model
- **Human/task state:** Labels represent configurable composite skills. The default pick/place catalog is exactly `IDLE`, `HOVER`, `GRASP`, `CARRY`, `RELEASE`; changing an active catalog version or label semantics requires explicit authorization.
- **Default episode:** `IDLE -> HOVER -> GRASP -> CARRY -> RELEASE -> HOVER -> IDLE`. `IDLE` emits no arm or gripper command. `HOVER` means move to grasp hover when entered from IDLE, and return to the configured home joint preset when entered from RELEASE. `GRASP -> HOVER` is an abort-to-home edge guarded by an empty grasp.
- **Robot-side skill expansion:** Composite handlers deterministically expand labels into low-level Cartesian, joint-preset, and gripper actions. These primitives are execution details, not learned labels.
- **Inputs / measurements:** Timestamped state predictions and timestamped 2D object positions from a fixed-camera tabletop demonstration.
- **Outputs / commands:** Cartesian end-effector targets in meters, Panda joint targets in radians where applicable, and gripper commands using the active MuJoCo Panda model's documented actuator semantics.
- **Dynamics/model assumptions:**
  - MVP is tabletop pick-and-place.
  - The demonstrated object path is tracked directly; it is not learned by the temporal model.
  - Human 2D motion is retargeted to the robot workspace rather than copied as literal world coordinates.
  - Raw tracked points should not be streamed directly to the arm; the path is processed into robot-compatible waypoints/trajectory.
  - Vertical motion is scripted from task phase for the MVP.
  - End-effector orientation is fixed/downward for the MVP.
  - MuJoCo provides ground-truth robot/object state during robot execution; human-side vision is not required to localize the simulated object.
  - Robot execution is not a learned policy in the MVP.
  - For the standard position-actuated Menagerie Panda, Mink supplies geometric IK and Ruckig supplies the persistent joint-position reference. Measured pose remains authoritative for arrival and execution gates; a finished command trajectory is not success by itself.
  - Panda manufacturer velocity, acceleration, and jerk values are hard ceilings. The configured 0.5 rad/s operating velocity is a lower simulation envelope. Per-joint tracking error must fail closed rather than allowing reference wind-up.
  - The standard Panda gripper retains its documented nominal 0–0.08 m mapping. Normal OPEN execution uses the separately configured 0.0799 m command to remain inside the simulated hard stop; do not widen the model or measurement allowance to replace this margin.
- **Coordinate frames / conventions:**
  - Human demonstration coordinates arrive as normalized 2D tabletop/workspace coordinates.
  - The `CoordinateRetargeter` is the only subsystem responsible for mapping source coordinates into MuJoCo/Panda coordinates.
  - Exact source-axis orientation, MuJoCo world-axis convention, table origin, workspace bounds, and sign conventions must be defined explicitly in code/config/documentation before use. Agents must not silently infer or flip axes to make a demo work.
  - All robot-space linear positions use meters; joint angles use radians unless the active API explicitly documents otherwise.
- **Noise/uncertainty assumptions:**
  - Temporal predictions may flicker and require post-processing/hysteresis.
  - Object tracking may contain jitter and occasional low-confidence samples.
  - Smoothing thresholds, confidence thresholds, and transition hysteresis are model/design parameters; do not tune them silently.

### Timing
- **Temporal-model rate:** Low-rate/asynchronous relative to robot control; exact rate is not yet a protected project constant.
- **Object-tracking rate:** Per-frame or tracker-native rate; exact rate is not yet a protected project constant.
- **Trajectory/control rate:** Higher than temporal inference and generated from the processed robot trajectory.
- **Simulation rate:** MuJoCo-native integration/control timing as defined by the chosen model/configuration.
- **Timing assumptions:**
  - Temporal predictions and object tracks must share a timebase or be explicitly synchronized before task extraction.
  - Do not assume one model prediction per tracking sample.
  - Do not hardcode timing constants solely to make one demonstration succeed.
  - If exact rates become standardized, document them here or in an authoritative configuration file and reference that source.

### Safety / Hard Constraints
- This main-branch MVP targets **simulation only**. Do not add or enable physical-robot actuation without explicit authorization.
- Do not alter Panda joint limits, actuator ranges, collision behavior, gripper limits, workspace limits, or MuJoCo safety-related configuration merely to make a trajectory succeed.
- Do not bypass IK validity/reachability checks by forcing joint states or teleporting the arm during normal task execution. Teleportation/reset is acceptable only for explicit simulation initialization/reset logic.
- Do not silently move the simulated object to create a successful outcome; object motion during execution must result from the intended simulation/task mechanism.

### Interfaces That Must Remain Stable
- Temporal-model boundary: timestamp + a complete score mapping for the active catalog, plus explicit detection validity.
- Tracker boundary: timestamp + normalized 2D object position.
- Default learned skill vocabulary: `IDLE`, `HOVER`, `GRASP`, `CARRY`, `RELEASE`; other deployments use a separately versioned catalog and matching handler registry.
- Task boundary: `PickPlaceTask` semantics of start, goal, demonstrated path, grasp time, and release time.
- Robot-side code must remain independent of the particular upstream encoder (V-JEPA, geometric baseline, or hybrid). Swapping the upstream classifier must not require rewriting the Panda harness.
- Position/path tracking remains separate from manipulation-state classification. Do not collapse these interfaces without explicit authorization.
- A saved `home` is a complete named arm-joint configuration resolved from either an explicit mapping or a MuJoCo keyframe. Normal return-home motion uses the bounded Ruckig joint-reference path without Cartesian IK. It must not reset/teleport the robot or change the gripper command.

### Known Intentional Limitations
- Fixed camera and constrained tabletop environment for the MVP.
- Single simple manipulated object initially (for example, a ball or similarly simple graspable object).
- Single Franka Panda in MuJoCo.
- 2D demonstrated path plus scripted `z`; no general monocular 3D reconstruction.
- Fixed gripper/end-effector orientation; object orientation/rotation is a stretch goal.
- Approximate path retargeting is acceptable; exact human motion replay is not required.
- No learned inverse kinematics, motion planner, or low-level controller in the MVP.
- Collision-aware/global motion planning is optional unless later made a requirement.
- Audio/transcription is an upstream labeling/semantic aid and is not required by the Panda harness.
- The system is intended to learn/recognize skill state while recovering geometry separately; do not redesign it into pure trajectory replay without explicit authorization.

### Development Order / Integration Strategy
Agents should prefer the following dependency order when implementing the robot side:
1. Load and step a known-good Panda + tabletop + object scene in MuJoCo.
2. Execute a manually specified robot-space end-effector target/path.
3. Add gripper actuation and validate grasp/release behavior.
4. Add IK and robot-side skill execution.
5. Add `PickPlaceTask` execution using a manually constructed task.
6. Add coordinate retargeting from normalized source coordinates.
7. Add state/task extraction from prerecorded upstream outputs.
8. Integrate live/upstream temporal predictions and tracker output only after the robot harness works independently.

Do not block robot-side development on the temporal model. A manually constructed `PickPlaceTask` is the canonical integration stub.

### Verification
No repository-wide commands are yet authoritative in this file. Before running or adding commands, inspect the repository's existing `README`, project metadata, test configuration, and scripts. Do not invent a new toolchain if one already exists.

Minimum verification expectations:
- **Build/import:** The robot harness and MuJoCo model load without errors.
- **Unit tests:** Coordinate mapping, path processing, state-transition filtering, and task extraction should have focused tests where practical.
- **Integration test:** A manually created `PickPlaceTask` produces a valid Panda execution attempt in simulation.
- **Simulation test:** Verify grasp, transport, release, and final object position from MuJoCo state rather than visual appearance alone.
- **End-to-end test:** Feed prerecorded temporal predictions + tracked positions through the full harness and record success/failure plus final placement error.
- Report what was not verified; do not equate rendering correctly with task success.

### Evaluation Metrics
At minimum, log:
- whether a grasp occurred successfully;
- whether the object was transported while grasped;
- whether release occurred;
- final object-position error relative to the retargeted goal;
- task success/failure under an explicit tolerance;
- state/skill transitions and relevant timestamps for debugging.

Any success tolerance is a design parameter. If none exists yet, surface the missing decision rather than choosing a convenient value silently.

### Relevant Documentation
- `docs/ROBOT_EXECUTION.md` defines the implemented IK/gripper boundary: world-meter `ToolPose` with explicit `wxyz` quaternions and body-to-tool offset, named joint/actuator profiles, Mink IK plus a Ruckig position-reference layer, the standard Panda's safe-open command, Python >=3.10 robot dependency group, and required execution criteria. `configs/robots/panda.yaml` intentionally leaves unresolved experiment settings null. The configured measured-state allowance is 10 micrometers per named finger slide joint; preserve raw observations and all commanded/model limits, and do not apply this allowance to arm joints. The fixed cube fixture succeeds with separately logged settings, but production scene calibration, object-set validation, and task-success criteria remain unresolved.
- `docs/TASK_EXTRACTION_AND_RETARGETING.md` defines the user-approved frame-based, centimeter-space contract for offline steps 2 through 4, including required mapping and path-processing configuration and deferred waypoint construction.
- `docs/SKILL_GRAPH.md` defines the catalog, relationship graph, top-two post-state decision policy, composite handler contract, default episode, and unresolved post-processing thresholds.
- This `AGENTS.md` Project Contract is authoritative for agent behavior on `main` until superseded by explicit project documentation.
- Prefer existing repository `README`, architecture notes, configuration files, MuJoCo model documentation, and upstream interface definitions when present.
- If implementation introduces a durable coordinate convention, message schema, or configuration contract, document it in the repository and update this section rather than leaving the convention implicit in code.

---

## 12. Expected Completion Report

For debugging or implementation work, finish with a compact report:

### Changed
- `<small list of implementation changes>`

### Why
- `<causal explanation>`

### Verified
- `<tests/checks actually performed>`

### Model/design concerns not changed
- `<suspected Class B issues, or "none observed">`

### Ambiguities / remaining uncertainty
- `<Class C items, limitations, or "none">`

Do not describe an unverified hypothesis as a confirmed root cause.

---

## 13. Default Decision Rule

When uncertain whether a proposed change crosses from implementation into engineering intent:

**Do not make the change. Surface it as a hypothesis.**

The purpose of this policy is not to make agents passive. It is to make fast iteration interpretable: automate mechanical work aggressively while keeping consequential modeling and design decisions visible to the human engineer.
