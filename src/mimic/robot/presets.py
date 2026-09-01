"""Robot-specific saved configurations resolved into named arm-joint presets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import mujoco
import numpy as np

from mimic.robot.model import ModelBindings


@dataclass(frozen=True)
class JointPreset:
    preset_id: str
    joint_positions: Mapping[str, float]

    def __post_init__(self) -> None:
        if not isinstance(self.preset_id, str) or not self.preset_id.strip():
            raise ValueError("preset_id must be a nonempty string")
        positions = dict(self.joint_positions)
        if not positions or any(
            not isinstance(name, str)
            or not name
            or isinstance(value, bool)
            or not np.isfinite(value)
            for name, value in positions.items()
        ):
            raise ValueError("A joint preset requires finite named scalar positions")
        object.__setattr__(self, "joint_positions", positions)


def resolve_joint_preset(
    bindings: ModelBindings,
    preset_id: str,
    *,
    keyframe: Optional[str] = None,
    joint_positions: Optional[Mapping[str, float]] = None,
) -> JointPreset:
    """Resolve exactly one configured source without including gripper/object coordinates."""
    if (keyframe is None) == (joint_positions is None):
        raise ValueError("A joint preset requires exactly one of keyframe or joint_positions")
    arm_names = bindings.profile.arm_joints
    if keyframe is not None:
        key_id = mujoco.mj_name2id(bindings.model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
        if key_id < 0:
            raise ValueError(f"Unknown MuJoCo keyframe: {keyframe}")
        values = np.asarray(bindings.model.key_qpos[key_id, bindings.qpos_ids], dtype=float)
        positions = dict(zip(arm_names, map(float, values)))
    else:
        positions = dict(joint_positions or {})
        if set(positions) != set(arm_names):
            missing = sorted(set(arm_names) - set(positions))
            extra = sorted(set(positions) - set(arm_names))
            raise ValueError(f"Joint preset/profile mismatch; missing={missing}, extra={extra}")
        values = np.array([positions[name] for name in arm_names], dtype=float)
    bindings.validate_arm(values)
    return JointPreset(preset_id, positions)
