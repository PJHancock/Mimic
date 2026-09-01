"""Explicit robot model bindings. No robot name or joint ordering is assumed."""

from dataclasses import dataclass
from typing import Mapping, Tuple

import mujoco
import numpy as np

from mimic.common.types import ToolPose


@dataclass(frozen=True)
class RobotProfile:
    arm_joints: Tuple[str, ...]
    arm_actuators: Tuple[str, ...]
    tool_body: str
    tool_offset: ToolPose  # transform from the body to the physical tool center
    velocity_limits: Tuple[float, ...]  # rad/s for hinges, m/s for slides
    workspace_min: Tuple[float, float, float]
    workspace_max: Tuple[float, float, float]

    def __post_init__(self):
        n = len(self.arm_joints)
        if not n or len(self.arm_actuators) != n or len(self.velocity_limits) != n:
            raise ValueError("Joint, actuator and velocity-limit lengths must match")
        if len(set(self.arm_joints)) != n or len(set(self.arm_actuators)) != n:
            raise ValueError("Arm bindings must be unique")
        if not np.all(np.isfinite(self.velocity_limits)) or min(self.velocity_limits) <= 0:
            raise ValueError("Velocity limits must be finite and positive")
        lo, hi = np.asarray(self.workspace_min), np.asarray(self.workspace_max)
        if lo.shape != (3,) or hi.shape != (3,) or not np.all(np.isfinite([lo, hi])):
            raise ValueError("Workspace requires two finite 3D corners")
        if np.any(lo >= hi):
            raise ValueError("Workspace minimum must be below maximum")

    def validate_target(self, pose: ToolPose) -> None:
        if np.any(np.asarray(pose.position) < self.workspace_min) or np.any(
            np.asarray(pose.position) > self.workspace_max
        ):
            raise ValueError("Target outside configured workspace; not clamped")


class ModelBindings:
    """Resolve names and validate direct position-servo actuator semantics once."""

    def __init__(self, model: mujoco.MjModel, profile: RobotProfile):
        self.model = model
        self.profile = profile
        self.body_id = model.body(profile.tool_body).id
        self.joint_ids = np.array([model.joint(name).id for name in profile.arm_joints])
        self.actuator_ids = np.array([model.actuator(name).id for name in profile.arm_actuators])
        self.qpos_ids = model.jnt_qposadr[self.joint_ids].copy()
        self.dof_ids = model.jnt_dofadr[self.joint_ids].copy()
        for joint_id, actuator_id in zip(self.joint_ids, self.actuator_ids):
            if int(model.jnt_type[joint_id]) not in (
                mujoco.mjtJoint.mjJNT_HINGE,
                mujoco.mjtJoint.mjJNT_SLIDE,
            ):
                raise ValueError("Arm joints must be scalar hinge or slide joints")
            if not model.jnt_limited[joint_id] or not model.actuator_ctrllimited[actuator_id]:
                raise ValueError("Explicit joint and actuator limits are required")
            if (
                model.actuator_trntype[actuator_id] != mujoco.mjtTrn.mjTRN_JOINT
                or model.actuator_trnid[actuator_id, 0] != joint_id
            ):
                raise ValueError("Arm actuator must transmit directly to the named joint")
            if not np.array_equal(model.actuator_gear[actuator_id], [1, 0, 0, 0, 0, 0]):
                raise ValueError("Only unit-gear arm position actuators are supported")
            gain = model.actuator_gainprm[actuator_id]
            bias = model.actuator_biasprm[actuator_id]
            if (
                model.actuator_dyntype[actuator_id] != mujoco.mjtDyn.mjDYN_NONE
                or model.actuator_gaintype[actuator_id] != mujoco.mjtGain.mjGAIN_FIXED
                or model.actuator_biastype[actuator_id] != mujoco.mjtBias.mjBIAS_AFFINE
                or gain[0] <= 0
                or bias[0] != 0
                or bias[1] != -gain[0]
                or bias[2] > 0
            ):
                raise ValueError("Unsupported arm actuator: expected a position servo")
        self.lower = np.maximum(
            model.jnt_range[self.joint_ids, 0], model.actuator_ctrlrange[self.actuator_ids, 0]
        )
        self.upper = np.minimum(
            model.jnt_range[self.joint_ids, 1], model.actuator_ctrlrange[self.actuator_ids, 1]
        )
        if np.any(self.lower >= self.upper):
            raise ValueError("Arm joint/actuator ranges do not intersect")
        self.joint_slices = {}
        for j in range(model.njnt):
            name = model.joint(j).name
            if not name:
                raise ValueError("Every movable scene joint must have a unique name")
            start = model.jnt_qposadr[j]
            end = model.jnt_qposadr[j + 1] if j + 1 < model.njnt else model.nq
            self.joint_slices[name] = slice(start, end)

    def configuration(self, joints: Mapping[str, Tuple[float, ...]]) -> np.ndarray:
        if set(joints) != set(self.joint_slices):
            raise ValueError("Observation joint names do not match the loaded model")
        q = np.empty(self.model.nq)
        for name, section in self.joint_slices.items():
            values = np.asarray(joints[name])
            if values.shape != (section.stop - section.start,) or not np.all(np.isfinite(values)):
                raise ValueError(f"Invalid coordinates for {name}")
            q[section] = values
        for j in range(self.model.njnt):
            kind = int(self.model.jnt_type[j])
            adr = self.model.jnt_qposadr[j]
            if kind == mujoco.mjtJoint.mjJNT_FREE:
                adr += 3
            if kind in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_BALL):
                if not np.isclose(np.linalg.norm(q[adr : adr + 4]), 1, rtol=0, atol=1e-8):
                    raise ValueError("Invalid ball/free-joint quaternion")
        return q

    def validate_arm(self, values: np.ndarray) -> None:
        if values.shape != self.lower.shape or not np.all(np.isfinite(values)):
            raise ValueError("Invalid arm targets")
        if np.any(values < self.lower) or np.any(values > self.upper):
            raise ValueError("Arm target violates joint/actuator limits")
