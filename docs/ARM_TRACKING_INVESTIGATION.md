# Panda arm tracking investigation

Date: 2026-08-31

## Scope and invariants

This investigation started from the fixed Panda diagnostic in
`scripts/verify_panda.py`. The diagnostic variants identified the failure before
the user authorized the control, gripper, and fixture-policy changes described
below. The checked-in Panda model, arm actuator limits, gains, damping, simulation
step, IK objectives, and task timeout remain unchanged.

The invariant under test is: a bounded differential-IK step must produce a
position-servo reference that advances the measured robot toward the requested
tool pose, while measured state remains the source of arrival and safety checks.

## Confirmed arm failure

`MinkIKSolver.solve()` is re-seeded from measured joint positions on every 10 ms
tick. With the configured 0.5 rad/s velocity bound, its position target is at
most 0.005 rad ahead of the same measurement. That target is passed directly to
the standard Menagerie Panda position servos.

This boundary matters because Mink documents `solve_ik()` as returning a joint
velocity and demonstrates advancing configuration with `integrate_inplace()`;
see the official [inverse-kinematics API](https://kevinzakka.github.io/mink/api/inverse_kinematics.html)
and [quickstart](https://kevinzakka.github.io/mink/tutorial/quickstart.html).

For the model's arm servos, actuator force has the form

`force = kp * (position_target - measured_position) - kv * measured_velocity`

and `kv = kp / 10`. Away from the target, a saturated IK step therefore gives,
ignoring other forces, a steady velocity of approximately

`(kp / kv) * 0.005 rad = 0.05 rad/s`.

This is one tenth of the 0.5 rad/s IK reference. A zero-gravity run measured
about 0.048 rad/s on each arm joint, confirming the mechanism. The one-step
position target behaves as a small velocity-producing offset instead of a
persistent position trajectory.

### Distinguishing experiments

| Diagnostic variant | Result after 10 s |
| --- | --- |
| Current measured-state re-seeding | 0.39538 m position error, 0.29052 rad orientation error |
| Gravity disabled | 0.41124 m, 0.22045 rad; failure remains |
| Panda collisions disabled | Identical to current behavior; only object/table contact occurred |
| Arm force limits disabled | Identical to current behavior; no actuator saturated in the baseline |
| Five-step look-ahead, diagnostic only | 0.00552 m, 0.01133 rad |
| Open-loop persistent joint reference | 0.00554 m, 0.01136 rad; static tracking error remains |
| Persistent reference with measured Cartesian feedback | Reached 0.000049 m and 0.000099 rad in 4.60 s |

The baseline's largest arm actuator load was 44.8% of its force range. Removing
force limits did not change the trace. Removing robot contacts also did not
change the trace. These observations exclude force saturation and collision as
causes of the initial HOVER timeout. Gravity affects the path but is not the
primary cause.

The successful feedback variant integrated each bounded IK delta into a
persistent command reference while still computing IK error and completion from
measured state. A purely open-loop reference moved at the expected rate but
retained about 5.5 mm of Cartesian error under load, so reference persistence
alone is insufficient for the fixture's 0.1 mm acceptance tolerance.

## Implemented arm solution

`RuckigPositionIK` is a model-independent position-reference layer between
differential IK and `RobotIO`:

1. Initialize its named joint reference from the measured joints on explicit
   controller reset.
2. It iterates Mink in private state to obtain the endpoint correction for a
   requested Cartesian pose, then applies that correction to the persistent
   commanded reference.
3. Ruckig advances that named reference with configured velocity, acceleration,
   and jerk limits. Finished trajectories are replanned from measured error so
   gravity/load error is corrected rather than accepted as arrival.
4. Only Ruckig's per-tick named reference is sent to position actuators.
5. Continue to use measured pose for `AT_TARGET`, skill transitions, joint-limit
   checks, and task evaluation.
6. Fail closed if the reference crosses a joint/actuator limit, measured speed
   crosses its allowed bound, or reference-to-measurement lag exceeds an explicit
   tracking-error limit. A stalled actuator must not permit integrator wind-up.
7. Reset the reference only during explicit reset/recovery, never to hide lag
   during normal motion.

This keeps Mink as the robot-independent differential-IK backend and makes the
actuation adapter explicit. A robot with velocity actuators can use a different
adapter; a position-actuated model supplies arrays matching its own named joints.
The Panda implementation monitors measured speed against manufacturer limits and
tracking lag against a configured per-joint fail-stop bound.

The implementation uses [Ruckig](https://docs.ruckig.com/index.html) to
time-parameterize joint references with explicit velocity, acceleration, and
jerk bounds. Franka's official
[FER control limits](https://frankarobotics.github.io/docs/robot_specifications.html#limits-for-franka-emika-robot-fer)
provide hardware maxima for all three. The Menagerie MJCF does not embed these
limits. The configured 0.5 rad/s velocity envelope remains deliberately below
the manufacturer maximum; measured feedback and replanning resolve steady load
error rather than asking Ruckig to act as a feedback controller.

DAQP's default feasibility tolerance admitted one 0.005000232 rad step against
the 0.005 rad interval bound during the first pose transition. Passing an explicit
`primal_tol=1e-9` aligns the QP solve with Mink's existing postcondition. The
operating velocity was not widened or clipped.

Increasing the skill timeout, multiplying the IK time step, disabling gravity,
removing force limits, or tuning Panda servo gains are not recommended fixes.
They either leave the interface error intact or change the engineering model to
hide it.

## Additional failure modes exposed

The successful hover variant exposed three separate gripper/task issues. These
are not causes of the original arm timeout.

### Open command at the hard stop

During the faster arm transient, commanding the nominal 0.0800 m full opening
produced a maximum finger excursion of 13.2 micrometers, exceeding the approved
10 micrometer observation allowance. A diagnostic command of 0.0799 m total
opening kept the excursion below 2.3 micrometers and still satisfied the existing
1 mm open-state tolerance.

Implemented solution: the Panda driver keeps the nominal 0–0.08 m mapping, while
normal OPEN actions use a separately configured 0.0799 m command. This stays
0.1 mm inside the hard stop without expanding the measured-state tolerance.

### Slip threshold versus fixture behavior

The 4 cm, 30 g cube moved 9.50 mm relative to the tool during transport and
12.19 mm during lowering. The current 10 mm slip threshold stopped execution
at the start of `LOWER` after a 10.063 mm displacement. This threshold and the
grasp/tool geometry are design parameters.

Implemented fixture policy: `verify_panda.py` records the former 10 mm threshold
and uses 15 mm. The final controlled run observed 2.22 mm during lift, 4.50 mm
during path following, and 10.07 mm during lowering. This value applies only to
the fixed cube fixture; an intended object set still needs calibration.

### Gripper movement timeout reactivates after success

After `CLOSE` achieved a bilateral candidate grasp, `HOLD` retained the original
close start time. When table support removed finger contact during lowering, the
status immediately became `TIMEOUT` because more than two seconds had elapsed
since the original close request. This conflates the duration of an old,
successfully completed movement with later contact monitoring.

Implemented fix: successful OPEN/CLOSE completion is latched and the movement
timeout covers only an uninterrupted attempt to reach the requested state.
During HOLD, contact loss remains governed by the executor's dedicated
contact-loss and slip checks. The original two-second timeout remains unchanged.

## Diagnostic end-to-end result

The checked-in combination of Mink, Ruckig, a 0.0799 m open command,
completion-aware gripper timeout, and the fixture-only 15 mm slip threshold
completed in 19.23 simulated seconds. It reported grasp, transport, release, and
retreat with 4.60 mm final object-position error against a 10 mm threshold.

The largest reference-to-measurement lag was 0.0591 rad against the 0.1 rad
fail-stop bound; largest measured joint speed was 0.5031 rad/s, below the sourced
Panda hardware maxima. No finger crossed its 0.04 m model limit. This result is
simulation evidence for this scene, object, path, and parameter snapshot only.

## Classification

- **A — implementation, fixed:** position-servo targets now persist through a
  Ruckig motion generator; DAQP feasibility matches the explicit velocity-step
  check; gripper completion no longer reactivates an old movement timeout.
- **B — model/design, authorized for the fixed fixture:** 0.5 rad/s operating
  speed, sourced Panda acceleration/jerk constraints, 0.1 rad tracking bound,
  0.0799 m OPEN command, and 15 mm fixture slip acceptance.
- **Still experiment-specific:** scene/tool geometry, object set, grasp evidence,
  and placement criteria are not generalized by this successful run.
- **Excluded for the original HOVER failure:** collision, force saturation, and
  gravity as the primary cause.
