"""MuJoCo state ownership and atomic control writes; simulation only."""

from typing import Callable, Mapping, Optional, Protocol, Tuple

import mujoco
import numpy as np

from mimic.common.types import GripperFeedback, RobotState, ToolPose
from mimic.robot.model import ModelBindings


class RobotIO(Protocol):
    def read(self) -> RobotState: ...

    def apply(
        self, arm_targets: Mapping[str, float], gripper_targets: Mapping[str, float]
    ) -> None: ...
    def advance(self, duration_s: float) -> None: ...


class MuJoCoAdapter:
    def __init__(
        self,
        bindings: ModelBindings,
        gripper_observer: Callable[[mujoco.MjData, Optional[int]], GripperFeedback],
        gripper_actuators: Tuple[str, ...],
        object_body: Optional[str] = None,
    ):
        self.bindings, self.model = bindings, bindings.model
        self.data = mujoco.MjData(self.model)
        self._observe_gripper = gripper_observer
        self._gripper_ids = {name: self.model.actuator(name).id for name in gripper_actuators}
        if len(self._gripper_ids) != len(gripper_actuators):
            raise ValueError("Duplicate gripper actuator names")
        if set(self._gripper_ids.values()) & set(bindings.actuator_ids):
            raise ValueError("Arm and gripper actuator ownership must be disjoint")
        self._object_id = self.model.body(object_body).id if object_body else None
        self._tool_position = np.empty(3)
        self._tool_quaternion = np.empty(4)
        mujoco.mj_forward(self.model, self.data)

    def reset(self, keyframe: str) -> RobotState:
        """Only explicit initialization resets joint/object coordinates."""
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key(keyframe).id)
        mujoco.mj_forward(self.model, self.data)
        return self.read()

    def read(self) -> RobotState:
        # Refresh position-dependent observations after mj_step; do not expose stale xpos.
        mujoco.mj_forward(self.model, self.data)
        body = self.bindings.body_id
        offset = self.bindings.profile.tool_offset
        mujoco.mju_mulPose(
            self._tool_position,
            self._tool_quaternion,
            self.data.xpos[body],
            self.data.xquat[body],
            np.array(offset.position),
            np.array(offset.quaternion_wxyz),
        )
        return RobotState(
            float(self.data.time),
            {
                name: tuple(map(float, self.data.qpos[section]))
                for name, section in self.bindings.joint_slices.items()
            },
            ToolPose(tuple(self._tool_position), tuple(self._tool_quaternion)),
            self._observe_gripper(self.data, self._object_id),
            tuple(self.data.xpos[self._object_id]) if self._object_id is not None else None,
        )

    def apply(self, arm_targets: Mapping[str, float], gripper_targets: Mapping[str, float]) -> None:
        profile = self.bindings.profile
        if set(arm_targets) != set(profile.arm_joints):
            raise ValueError("Command must contain exactly the bound arm joints")
        values = np.array([arm_targets[name] for name in profile.arm_joints])
        self.bindings.validate_arm(values)
        if set(gripper_targets) != set(self._gripper_ids):
            raise ValueError("Command must contain exactly the bound gripper actuators")
        controls = self.data.ctrl.copy()
        controls[self.bindings.actuator_ids] = values
        for name, actuator_id in self._gripper_ids.items():
            value = gripper_targets[name]
            lo, hi = self.model.actuator_ctrlrange[actuator_id]
            if (
                not self.model.actuator_ctrllimited[actuator_id]
                or not np.isfinite(value)
                or not lo <= value <= hi
            ):
                raise ValueError(f"Invalid control for {name}")
            controls[actuator_id] = value
        # All validation precedes the only live actuator write.
        self.data.ctrl[:] = controls

    def advance(self, duration_s: float) -> None:
        count = duration_s / self.model.opt.timestep
        if (
            not np.isfinite(count)
            or count < 1
            or not np.isclose(count, round(count), rtol=0, atol=1e-8)
        ):
            raise ValueError("Control interval must contain an integer number of MuJoCo steps")
        warnings_before = self.data.warning.number.copy()
        for _ in range(round(count)):
            mujoco.mj_step(self.model, self.data)
            if not np.array_equal(self.data.warning.number, warnings_before):
                raise RuntimeError("MuJoCo emitted a warning; execution stopped without recovery")
