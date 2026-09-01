"""Coordinate one arm tick and a separately scheduled gripper update."""

from dataclasses import dataclass

import numpy as np

from mimic.common.types import IKResult, RobotState, ToolPose
from mimic.robot.gripper import GripperAction, GripperLogic, GripperResult
from mimic.robot.inverse_kinematics import IKSolver
from mimic.robot.simulation import RobotIO


class ExecutionFailure(RuntimeError):
    """A failed execution attempt: no automatic retry, release, or model changes."""


@dataclass(frozen=True)
class ControlSample:
    state: RobotState
    ik: IKResult
    gripper: GripperResult


class RobotController:
    def __init__(
        self,
        io: RobotIO,
        ik: IKSolver,
        gripper: GripperLogic,
        arm_control_hz: float,
        gripper_control_hz: float,
    ):
        if (
            not np.all(np.isfinite([arm_control_hz, gripper_control_hz]))
            or not 0 < gripper_control_hz <= arm_control_hz
        ):
            raise ValueError("Require finite rates with 0 < gripper Hz <= arm Hz")
        ratio = arm_control_hz / gripper_control_hz
        if not np.isclose(ratio, round(ratio), rtol=0, atol=1e-8):
            raise ValueError("Arm rate must be an integer multiple of gripper rate")
        self.io, self.ik, self.gripper = io, ik, gripper
        self.dt_s = 1 / arm_control_hz
        self._gripper_ticks = round(ratio)
        self.reset()

    def reset(self):
        self._ticks = 0
        self._gripper_result = None
        self._pending = None
        self._failed = False
        reset_ik = getattr(self.ik, "reset", None)
        if reset_ik is not None:
            reset_ik()
        self.gripper.reset()

    def prepare(self, target: ToolPose, action: GripperAction) -> ControlSample:
        """Observe/compute only. The executor can reject this sample before actuating."""
        if self._failed:
            raise ExecutionFailure("Controller is stopped after failure; explicit reset required")
        self._pending = None
        state = self.io.read()
        result = self.ik.solve(target, state, self.dt_s)
        if not result.valid:
            self._failed = True
            raise ExecutionFailure(f"IK {result.status.value}: {result.detail}")
        if self._ticks % self._gripper_ticks == 0:
            self._gripper_result = self.gripper.update(action, state.timestamp_s, state.gripper)
        self._pending = ControlSample(state, result, self._gripper_result)
        return self._pending

    def commit(self, sample: ControlSample) -> None:
        """Apply one validated tick; advance physics only after all execution gates pass."""
        if sample is not self._pending or sample.state.timestamp_s != self.io.read().timestamp_s:
            raise ExecutionFailure("Refusing a stale or already committed control sample")
        self._pending = None
        if not sample.ik.valid:
            raise ExecutionFailure("Refusing failed IK result")
        try:
            self.io.apply(sample.ik.joint_targets, sample.gripper.actuator_targets)
            self.io.advance(self.dt_s)
        except (ValueError, RuntimeError):
            self._failed = True
            raise
        self._ticks += 1
