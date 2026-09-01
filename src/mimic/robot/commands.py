"""Compatibility adapter for the existing Cartesian RobotCommand interface."""

from typing import Tuple

from mimic.common.types import RobotCommand, ToolPose
from mimic.robot.gripper import GripperAction


def command_target(
    command: RobotCommand,
    *,
    fixed_orientation_wxyz: Tuple[float, float, float, float],
    quaternion_order: str,
) -> tuple[ToolPose, GripperAction]:
    """Caller must declare the legacy quaternion order; no guessed frame conversion.

    target_position must already be retargeted into MuJoCo world meters. Duration
    and trajectory_points remain the caller's trajectory/scheduling responsibility.
    """
    if quaternion_order not in ("wxyz", "xyzw"):
        raise ValueError("Explicit quaternion_order must be wxyz or xyzw")
    q = command.target_orientation
    if q is None:
        q = fixed_orientation_wxyz
    elif quaternion_order == "xyzw":
        q = (q[3], *q[:3])
    pose = ToolPose(command.target_position, q)
    return pose, GripperAction.OPEN if command.gripper_open else GripperAction.CLOSE
