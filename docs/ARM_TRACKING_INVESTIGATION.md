# Panda arm tracking investigation

Date: 2026-08-31

## Scope and invariants

This investigation starts from the fixed Panda diagnostic in
`scripts/verify_panda.py`. It does not change the checked-in Panda model, arm
actuator limits, gains, damping, simulation step, IK costs, task timeout, or
success thresholds. Variants described below were run in memory to distinguish
causes; none is a production calibration or an approved control policy.

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

## Recommended arm solution

Add a model-independent position-reference controller between differential IK
and `RobotIO`:

1. Initialize its named joint reference from the measured joints on explicit
   controller reset.
2. On each tick, run Mink against measured state and obtain a joint-space delta
   bounded by the configured velocity limits and real control interval.
3. Integrate that delta into the persistent reference; do not replace the
   reference with `measured + delta` every tick.
4. Send the persistent named reference to position actuators.
5. Continue to use measured pose for `AT_TARGET`, skill transitions, joint-limit
   checks, and task evaluation.
6. Fail closed if the reference crosses a joint/actuator limit, measured speed
   crosses its allowed bound, or reference-to-measurement lag exceeds an explicit
   tracking-error limit. A stalled actuator must not permit integrator wind-up.
7. Reset the reference only during explicit reset/recovery, never to hide lag
   during normal motion.

This keeps Mink as the robot-independent differential-IK backend and makes the
actuation adapter explicit. A robot with velocity actuators can use a velocity
adapter; the standard Panda model uses the position-reference adapter. The
diagnostic reference caused a peak measured speed of 0.527 rad/s against the
0.5 rad/s reference, so measured-speed/acceleration policy must be decided before
this controller is adopted.

For a production trajectory layer, [Ruckig](https://docs.ruckig.com/index.html)
is an appropriate existing library to time-parameterize joint references with
explicit velocity, acceleration, and jerk bounds. Franka's official
[FER control limits](https://frankarobotics.github.io/docs/robot_specifications.html#limits-for-franka-emika-robot-fer)
provide hardware maxima for all three. The Menagerie MJCF does not embed the
velocity, acceleration, or jerk limits, and a lower operating envelope for this
simulation remains a project decision. Ruckig does not replace measured feedback
or resolve steady load error by itself.

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

Recommended solution: configure an explicit robot-profile open-command margin
inside the physical joint limit. Do not keep expanding the measured-state
tolerance. The 0.0799 m diagnostic value is evidence, not an approved production
setting.

### Slip threshold versus fixture behavior

The 4 cm, 30 g cube moved 9.50 mm relative to the tool during transport and
12.19 mm during lowering. The current 10 mm slip threshold stopped execution
at the start of `LOWER` after a 10.063 mm displacement. This threshold and the
grasp/tool geometry are design parameters.

Recommended solution: measure expected in-gripper settling across the intended
object set, verify the tool-center/grasp-height convention, and then choose a
slip policy that distinguishes stable settling from loss. Do not raise the
threshold solely to make this fixture pass.

### Gripper movement timeout reactivates after success

After `CLOSE` achieved a bilateral candidate grasp, `HOLD` retained the original
close start time. When table support removed finger contact during lowering, the
status immediately became `TIMEOUT` because more than two seconds had elapsed
since the original close request. This conflates the duration of an old,
successfully completed movement with later contact monitoring.

Recommended implementation fix: latch successful OPEN/CLOSE completion and make
the movement timeout cover only an uninterrupted attempt to reach the requested
state. During HOLD, contact loss should be handled by the executor's dedicated
contact-loss and slip checks. A diagnostic completion-aware timeout eliminated
this false failure with the original two-second setting.

## Diagnostic end-to-end result

An in-memory combination of persistent measured-feedback reference integration,
a 0.0799 m open command, completion-aware gripper timeout, and a diagnostic
15 mm slip threshold completed the fixture in 22.8 simulation seconds. It
reported grasp, transport, and release with 0.162 mm final object-position error.

This result proves the remaining stages can execute in the present scene. It
does not authorize the 15 mm slip threshold, the open-command margin, or the arm
control policy, and it is not hardware verification.

## Classification

- **A — implementation:** gripper movement timeout is applied again after the
  close operation has already succeeded.
- **B — model/design:** persistent position-reference policy, tracking-error
  limit, selected operating speed/acceleration/jerk envelope below published
  hardware maxima, gripper open margin, grasp geometry, and slip acceptance.
- **Excluded for the original HOVER failure:** collision, force saturation, and
  gravity as the primary cause.
