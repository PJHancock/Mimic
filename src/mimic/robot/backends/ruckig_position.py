"""Jerk-limited position references for position-actuated robot models."""

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Mapping, Optional

import numpy as np
import ruckig

from mimic.common.types import IKResult, IKStatus, RobotState, ToolPose
from mimic.robot.inverse_kinematics import IKSolver
from mimic.robot.model import ModelBindings


@dataclass(frozen=True)
class PositionTrajectorySettings:
    """Motion-generator constraints, ordered like the robot profile's arm joints."""

    hardware_velocity_limits: tuple[float, ...]
    acceleration_limits: tuple[float, ...]
    jerk_limits: tuple[float, ...]
    maximum_tracking_errors: tuple[float, ...]
    maximum_planning_steps: int

    def __post_init__(self):
        arrays = (
            self.hardware_velocity_limits,
            self.acceleration_limits,
            self.jerk_limits,
            self.maximum_tracking_errors,
        )
        if any(not values for values in arrays) or any(
            not np.all(np.isfinite(values)) or min(values) <= 0 for values in arrays
        ):
            raise ValueError("Trajectory limits must contain finite positive values")
        if (
            isinstance(self.maximum_planning_steps, bool)
            or not isinstance(self.maximum_planning_steps, int)
            or self.maximum_planning_steps <= 0
        ):
            raise ValueError("Maximum planning steps must be a positive integer")


class RuckigPositionIK:
    """Turn geometric IK results into persistent position-servo references.

    The wrapped solver remains responsible for measured pose error and arm/model
    validity. Ruckig owns commanded position, velocity and acceleration state.
    """

    def __init__(
        self,
        solver: IKSolver,
        bindings: ModelBindings,
        settings: PositionTrajectorySettings,
    ):
        self.solver, self.bindings, self.settings = solver, bindings, settings
        count = len(bindings.profile.arm_joints)
        arrays = (
            settings.hardware_velocity_limits,
            settings.acceleration_limits,
            settings.jerk_limits,
            settings.maximum_tracking_errors,
        )
        if any(len(values) != count for values in arrays):
            raise ValueError("Trajectory limits must match the arm joint count")
        if np.any(
            np.asarray(bindings.profile.velocity_limits)
            > np.asarray(settings.hardware_velocity_limits)
        ):
            raise ValueError("Operating velocity cannot exceed the sourced hardware limit")
        self.reset()

    def reset(self) -> None:
        self._dt_s = None
        self._ruckig = self._input = self._output = None
        self._target = None
        self._trajectory_finished = False
        self._last_observed_q = None
        self._last_observed_time = None

    def _result(self, source: IKResult, status: IKStatus, started: float, detail: str = ""):
        return replace(
            source,
            status=status,
            joint_targets={},
            solve_time_s=perf_counter() - started,
            detail=detail,
        )

    def _arm(self, state: RobotState) -> np.ndarray:
        return np.array(
            [state.joint_positions[name][0] for name in self.bindings.profile.arm_joints],
            dtype=float,
        )

    def _check_measured_velocity(self, state: RobotState, q: np.ndarray) -> Optional[str]:
        if self._last_observed_time is not None:
            elapsed = state.timestamp_s - self._last_observed_time
            if elapsed < 0:
                return "Measured arm time moved backwards"
            if elapsed == 0:
                if not np.array_equal(q, self._last_observed_q):
                    return "Measured arm position changed without elapsed simulation time"
            else:
                velocity = np.abs(q - self._last_observed_q) / elapsed
                if np.any(velocity > np.asarray(self.settings.hardware_velocity_limits) + 1e-8):
                    return "Measured arm velocity exceeds a sourced joint limit"
        self._last_observed_q = q.copy()
        self._last_observed_time = state.timestamp_s
        return None

    def _initialize(self, q: np.ndarray, dt_s: float) -> None:
        if self._dt_s is not None:
            if not np.isclose(dt_s, self._dt_s, rtol=0, atol=1e-12):
                raise ValueError("Ruckig position control requires a fixed control interval")
            return
        self._dt_s = dt_s
        count = len(q)
        self._ruckig = ruckig.Ruckig(count, dt_s)
        self._input = ruckig.InputParameter(count)
        self._output = ruckig.OutputParameter(count)
        self._input.current_position = q.tolist()
        self._input.current_velocity = [0.0] * count
        self._input.current_acceleration = [0.0] * count
        self._input.max_velocity = list(self.bindings.profile.velocity_limits)
        self._input.max_acceleration = list(self.settings.acceleration_limits)
        self._input.max_jerk = list(self.settings.jerk_limits)

    def _joint_state(self, state: RobotState, q: np.ndarray) -> RobotState:
        configuration = self.bindings.configuration(state.joint_positions)
        configuration[self.bindings.qpos_ids] = q
        joints = {
            name: tuple(map(float, configuration[section]))
            for name, section in self.bindings.joint_slices.items()
        }
        return replace(state, joint_positions=joints)

    def _stop_at_measured(self, target: ToolPose, measured_q: np.ndarray) -> None:
        """Stop a Cartesian reference once measured pose has reached its target.

        A planned joint endpoint can carry the nonlinear Cartesian path beyond a
        pose that the measured robot has already reached. Re-anchor all persistent
        Ruckig state at that measured configuration so the next skill starts from
        rest instead of inheriting the remainder of the old trajectory.
        """
        stopped = measured_q.tolist()
        zeros = [0.0] * len(stopped)
        self._input.current_position = stopped
        self._input.current_velocity = zeros
        self._input.current_acceleration = zeros
        self._input.target_position = stopped
        self._input.target_velocity = zeros
        self._input.target_acceleration = zeros
        self._target = target
        self._trajectory_finished = True

    def _plan(self, target: ToolPose, state: RobotState, measured_q: np.ndarray, dt_s: float):
        q = measured_q.copy()
        last = None
        for _ in range(self.settings.maximum_planning_steps):
            last = self.solver.solve(target, self._joint_state(state, q), dt_s)
            if not last.valid:
                return last
            if last.status == IKStatus.AT_TARGET:
                correction = q - measured_q
                reference = np.asarray(self._input.current_position)
                goal = reference + correction
                try:
                    self.bindings.validate_arm(goal)
                except ValueError as exc:
                    return replace(last, status=IKStatus.LIMIT_VIOLATION, detail=str(exc))
                self._input.target_position = goal.tolist()
                self._input.target_velocity = [0.0] * len(goal)
                self._input.target_acceleration = [0.0] * len(goal)
                self._trajectory_finished = False
                self._target = target
                return last
            q = np.array([last.joint_targets[name] for name in self.bindings.profile.arm_joints])
        return replace(
            last,
            status=IKStatus.SOLVER_FAILED,
            joint_targets={},
            detail="IK planning did not converge within the configured step bound",
        )

    def solve(self, target: ToolPose, state: RobotState, dt_s: float) -> IKResult:
        return self._solve_cartesian(
            target,
            state,
            dt_s,
            stop_on_measured_arrival=False,
        )

    def solve_stopping_at_measured_arrival(
        self, target: ToolPose, state: RobotState, dt_s: float
    ) -> IKResult:
        """Solve a manipulation-boundary target and stop on measured arrival."""
        return self._solve_cartesian(
            target,
            state,
            dt_s,
            stop_on_measured_arrival=True,
        )

    def _solve_cartesian(
        self,
        target: ToolPose,
        state: RobotState,
        dt_s: float,
        *,
        stop_on_measured_arrival: bool,
    ) -> IKResult:
        started = perf_counter()
        measured = self.solver.solve(target, state, dt_s)
        if not measured.valid:
            return replace(measured, solve_time_s=perf_counter() - started)
        measured_q = self._arm(state)
        velocity_error = self._check_measured_velocity(state, measured_q)
        if velocity_error:
            return self._result(measured, IKStatus.LIMIT_VIOLATION, started, velocity_error)
        try:
            self._initialize(measured_q, dt_s)
        except ValueError as exc:
            return self._result(measured, IKStatus.INVALID_INPUT, started, str(exc))

        reference = np.asarray(self._input.current_position)
        if np.any(
            np.abs(reference - measured_q) > np.asarray(self.settings.maximum_tracking_errors)
        ):
            return self._result(
                measured,
                IKStatus.LIMIT_VIOLATION,
                started,
                "Position reference exceeds the configured tracking-error bound",
            )
        if measured.status == IKStatus.AT_TARGET and (
            self._trajectory_finished or stop_on_measured_arrival
        ):
            command = reference
            if stop_on_measured_arrival:
                self._stop_at_measured(target, measured_q)
                command = measured_q
            return replace(
                measured,
                joint_targets=dict(zip(self.bindings.profile.arm_joints, map(float, command))),
                solve_time_s=perf_counter() - started,
            )

        if target != self._target or self._trajectory_finished:
            planned = self._plan(target, state, measured_q, dt_s)
            if not planned.valid:
                return replace(planned, solve_time_s=perf_counter() - started)

        trajectory_result = self._ruckig.update(self._input, self._output)
        if int(trajectory_result) < 0:
            return self._result(
                measured,
                IKStatus.SOLVER_FAILED,
                started,
                f"Ruckig failed with {trajectory_result}",
            )
        command = np.asarray(self._output.new_position)
        try:
            self.bindings.validate_arm(command)
        except ValueError as exc:
            return self._result(measured, IKStatus.LIMIT_VIOLATION, started, str(exc))
        if np.any(np.abs(command - measured_q) > np.asarray(self.settings.maximum_tracking_errors)):
            return self._result(
                measured,
                IKStatus.LIMIT_VIOLATION,
                started,
                "New position reference exceeds the configured tracking-error bound",
            )
        self._output.pass_to_input(self._input)
        self._trajectory_finished = trajectory_result == ruckig.Result.Finished
        return replace(
            measured,
            status=IKStatus.VALID_STEP,
            joint_targets=dict(zip(self.bindings.profile.arm_joints, map(float, command))),
            solve_time_s=perf_counter() - started,
        )

    def solve_joint_target(
        self,
        target: Mapping[str, float],
        tolerances: Mapping[str, float],
        state: RobotState,
        dt_s: float,
    ) -> IKResult:
        """Generate a persistent reference to an explicit saved arm configuration.

        This path deliberately bypasses Cartesian IK while retaining the same Ruckig
        limits, measured-velocity checks, reference tracking bound, and joint limits.
        """
        started = perf_counter()
        names = self.bindings.profile.arm_joints
        if set(target) != set(names) or set(tolerances) != set(names):
            return IKResult(
                IKStatus.INVALID_INPUT,
                {},
                None,
                None,
                perf_counter() - started,
                "Joint target and tolerances must exactly match the configured arm joints",
            )
        goal = np.array([target[name] for name in names], dtype=float)
        tolerance = np.array([tolerances[name] for name in names], dtype=float)
        if (
            not np.all(np.isfinite(goal))
            or not np.all(np.isfinite(tolerance))
            or np.any(tolerance <= 0)
        ):
            return IKResult(
                IKStatus.INVALID_INPUT,
                {},
                None,
                None,
                perf_counter() - started,
                "Joint targets must be finite and tolerances must be finite and positive",
            )
        try:
            self.bindings.validate_arm(goal)
        except ValueError as exc:
            return IKResult(
                IKStatus.LIMIT_VIOLATION,
                {},
                None,
                None,
                perf_counter() - started,
                str(exc),
            )

        measured_q = self._arm(state)
        arrived = bool(np.all(np.abs(goal - measured_q) <= tolerance))
        measured = IKResult(
            IKStatus.AT_TARGET if arrived else IKStatus.VALID_STEP,
            {},
            None,
            None,
            0,
        )
        velocity_error = self._check_measured_velocity(state, measured_q)
        if velocity_error:
            return self._result(measured, IKStatus.LIMIT_VIOLATION, started, velocity_error)
        try:
            self._initialize(measured_q, dt_s)
        except ValueError as exc:
            return self._result(measured, IKStatus.INVALID_INPUT, started, str(exc))

        reference = np.asarray(self._input.current_position)
        if np.any(
            np.abs(reference - measured_q) > np.asarray(self.settings.maximum_tracking_errors)
        ):
            return self._result(
                measured,
                IKStatus.LIMIT_VIOLATION,
                started,
                "Position reference exceeds the configured tracking-error bound",
            )
        signature = ("joint", *map(float, goal))
        if arrived and self._trajectory_finished and self._target == signature:
            return replace(
                measured,
                joint_targets=dict(zip(names, map(float, reference))),
                solve_time_s=perf_counter() - started,
            )
        if self._target != signature or self._trajectory_finished:
            self._input.target_position = goal.tolist()
            self._input.target_velocity = [0.0] * len(goal)
            self._input.target_acceleration = [0.0] * len(goal)
            self._trajectory_finished = False
            self._target = signature

        trajectory_result = self._ruckig.update(self._input, self._output)
        if int(trajectory_result) < 0:
            return self._result(
                measured,
                IKStatus.SOLVER_FAILED,
                started,
                f"Ruckig failed with {trajectory_result}",
            )
        command = np.asarray(self._output.new_position)
        try:
            self.bindings.validate_arm(command)
        except ValueError as exc:
            return self._result(measured, IKStatus.LIMIT_VIOLATION, started, str(exc))
        if np.any(np.abs(command - measured_q) > np.asarray(self.settings.maximum_tracking_errors)):
            return self._result(
                measured,
                IKStatus.LIMIT_VIOLATION,
                started,
                "New position reference exceeds the configured tracking-error bound",
            )
        self._output.pass_to_input(self._input)
        self._trajectory_finished = trajectory_result == ruckig.Result.Finished
        return replace(
            measured,
            status=IKStatus.VALID_STEP,
            joint_targets=dict(zip(names, map(float, command))),
            solve_time_s=perf_counter() - started,
        )
