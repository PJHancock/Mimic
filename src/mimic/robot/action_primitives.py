"""Robot-neutral action shapes emitted by composite skill handlers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Union

import numpy as np

from mimic.common.types import ToolPose
from mimic.robot.gripper import GripperAction


@dataclass(frozen=True)
class CartesianMotion:
    primitive_id: str
    target: ToolPose
    gripper_action: GripperAction


@dataclass(frozen=True)
class JointPresetMotion:
    primitive_id: str
    preset_id: str
    joint_positions: Mapping[str, float]
    gripper_action: GripperAction = GripperAction.HOLD

    def __post_init__(self) -> None:
        positions = dict(self.joint_positions)
        if not positions or any(
            not isinstance(name, str)
            or not name
            or isinstance(value, bool)
            or not np.isfinite(value)
            for name, value in positions.items()
        ):
            raise ValueError("A joint preset motion requires finite named scalar positions")
        object.__setattr__(self, "joint_positions", positions)


RobotAction = Union[CartesianMotion, JointPresetMotion]
