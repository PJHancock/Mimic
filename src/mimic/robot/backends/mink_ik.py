"""Mink differential IK, isolated from live MuJoCo data and from gripper control."""

from time import perf_counter
from typing import Mapping, Optional

import mink
import mujoco
import numpy as np
from qpsolvers.exceptions import SolverError

from mimic.common.types import IKResult, IKStatus, RobotState, ToolPose
from mimic.robot.inverse_kinematics import IKSettings
from mimic.robot.model import ModelBindings


def _transform(pose: ToolPose) -> mink.SE3:
    return mink.SE3.from_rotation_and_translation(
        mink.SO3(np.array(pose.quaternion_wxyz, dtype=float)), np.array(pose.position, dtype=float)
    )


class _MeasuredGripperConfiguration(mink.Configuration):
    """Keep raw coordinates; apply only explicitly named gripper observation allowances."""

    def __init__(self, bindings: ModelBindings, tolerances_m: Mapping[str, float]):
        model = bindings.model
        self.gripper_tolerances = {}
        for name, tolerance in tolerances_m.items():
            joint = model.joint(name).id
            if (
                joint in bindings.joint_ids
                or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_SLIDE
                or not model.jnt_limited[joint]
            ):
                raise ValueError(
                    "Measured gripper tolerance requires a limited non-arm slide joint"
                )
            if (
                isinstance(tolerance, bool)
                or not np.isfinite(tolerance)
                or tolerance <= 0
                or tolerance >= np.ptp(model.jnt_range[joint])
            ):
                raise ValueError(
                    "Measured gripper tolerance must be positive and smaller than travel"
                )
            # An observed finger must not move the tracked tool or any arm joint.
            protected_bodies = [bindings.body_id, *model.jnt_bodyid[bindings.joint_ids]]
            for body in protected_bodies:
                while body:
                    if body == model.jnt_bodyid[joint]:
                        raise ValueError(
                            "A tolerated gripper joint cannot be an ancestor of the arm/tool"
                        )
                    body = model.body_parentid[body]
            self.gripper_tolerances[joint] = float(tolerance)
        super().__init__(model)

    def check_limits(self, tol: float = 1e-6, safety_break: bool = True) -> None:
        # Mink calls this again inside solve_ik. Always fail closed; do not disable
        # its safety check globally or change its tolerance for unrelated joints.
        for joint in range(self.model.njnt):
            kind = int(self.model.jnt_type[joint])
            if not self.model.jnt_limited[joint] or kind == mujoco.mjtJoint.mjJNT_FREE:
                continue
            adr = self.model.jnt_qposadr[joint]
            allowance = self.gripper_tolerances.get(joint, tol)
            lower, upper = self.model.jnt_range[joint]
            if kind == mujoco.mjtJoint.mjJNT_BALL:
                quat = self.q[adr : adr + 4]
                value = 2 * np.arctan2(np.linalg.norm(quat[1:]), abs(quat[0]))
                lower = 0.0
            else:
                value = self.q[adr]
            if not lower - allowance <= value <= upper + allowance:
                raise mink.NotWithinConfigurationLimits(
                    joint_id=joint,
                    value=float(value),
                    lower=float(lower),
                    upper=float(upper),
                    model=self.model,
                )


class _FrozenGripperPositionLimit(mink.Limit):
    """Keep original limits except rows for observed-and-frozen gripper joints.

    A zero-velocity finger cannot simultaneously be forced back from a compliant
    excursion. Its measurement is checked separately; arm inequalities stay exact.
    """

    def __init__(self, position_limit: mink.ConfigurationLimit, dof_ids: list[int]):
        self.position_limit = position_limit
        self.dof_ids = dof_ids

    def compute_qp_inequalities(self, configuration, dt):
        constraint = self.position_limit.compute_qp_inequalities(configuration, dt)
        if constraint.inactive:
            return constraint
        keep = ~np.any(constraint.G[:, self.dof_ids] != 0, axis=1)
        return mink.Constraint(G=constraint.G[keep], h=constraint.h[keep])


class MinkIKSolver:
    def __init__(
        self,
        bindings: ModelBindings,
        settings: IKSettings,
        reference_posture: Mapping[str, float],
        *,
        measured_gripper_tolerances_m: Optional[Mapping[str, float]] = None,
    ):
        self.bindings, self.settings = bindings, settings
        model, profile = bindings.model, bindings.profile
        if set(reference_posture) != set(profile.arm_joints):
            raise ValueError("Reference posture must name exactly the arm joints")
        reference = np.array([reference_posture[name] for name in profile.arm_joints])
        bindings.validate_arm(reference)
        self.configuration = (
            _MeasuredGripperConfiguration(bindings, measured_gripper_tolerances_m)
            if measured_gripper_tolerances_m
            else mink.Configuration(model)
        )
        self.tool_offset = _transform(profile.tool_offset)
        self.frame_task = mink.FrameTask(
            profile.tool_body,
            "body",
            settings.position_cost,
            settings.orientation_cost,
            gain=settings.task_gain,
        )
        self.tasks = [self.frame_task]
        if settings.posture_cost:
            cost = np.zeros(model.nv)
            cost[bindings.dof_ids] = settings.posture_cost
            posture = mink.PostureTask(model, cost)
            q_ref = model.qpos0.copy()
            q_ref[bindings.qpos_ids] = reference
            posture.set_target(q_ref)
            self.tasks.append(posture)
        position_limit = mink.ConfigurationLimit(model, gain=1.0)
        # Respect tighter actuator ranges as well as the model's joint limits.
        position_limit.lower[bindings.qpos_ids] = bindings.lower
        position_limit.upper[bindings.qpos_ids] = bindings.upper
        measured_dofs = [
            int(model.jnt_dofadr[model.joint(name).id])
            for name in (measured_gripper_tolerances_m or {})
        ]
        self.limits = [
            (
                _FrozenGripperPositionLimit(position_limit, measured_dofs)
                if measured_dofs
                else position_limit
            ),
            mink.VelocityLimit(model, dict(zip(profile.arm_joints, profile.velocity_limits))),
        ]
        frozen = sorted(set(range(model.nv)) - set(bindings.dof_ids))
        self.constraints = [mink.DofFreezingTask(model, frozen)] if frozen else []

    def solve(self, target: ToolPose, state: RobotState, dt_s: float) -> IKResult:
        started = perf_counter()
        pos_error = ori_error = None

        def result(status, targets=None, detail=""):
            return IKResult(
                status, targets or {}, pos_error, ori_error, perf_counter() - started, detail
            )

        try:
            if not np.isfinite(dt_s) or dt_s <= 0 or not np.isfinite(state.timestamp_s):
                raise ValueError("State time and control interval must be finite; dt > 0")
            self.bindings.profile.validate_target(target)
            q = self.bindings.configuration(state.joint_positions)
        except (ValueError, TypeError) as exc:
            return result(IKStatus.INVALID_INPUT, detail=str(exc))
        try:
            self.bindings.validate_arm(q[self.bindings.qpos_ids])
        except ValueError as exc:
            return result(IKStatus.LIMIT_VIOLATION, detail=str(exc))
        self.configuration.update(q)
        try:
            self.configuration.check_limits(safety_break=True)
            desired = _transform(target)
            measured = (
                self.configuration.get_transform_frame_to_world(
                    self.bindings.profile.tool_body, "body"
                )
                @ self.tool_offset
            )
            pos_error = float(np.linalg.norm(desired.translation() - measured.translation()))
            rotation_error = np.empty(3)
            mujoco.mju_subQuat(rotation_error, desired.rotation().wxyz, measured.rotation().wxyz)
            ori_error = float(np.linalg.norm(rotation_error))
            self.frame_task.set_target(desired @ self.tool_offset.inverse())
            # Mink 1.3.0 SO3.log uses sign(w); exactly w=0 erases a pi rotation.
            # Use MuJoCo's error for acceptance and reject that degenerate linearization.
            frame_error = self.frame_task.compute_error(self.configuration)
            if (
                ori_error > self.settings.orientation_tolerance_rad
                and np.linalg.norm(frame_error[3:]) < 1e-12
            ):
                raise ValueError("Mink orientation linearization is degenerate (exact half-turn)")
            if (
                pos_error <= self.settings.position_tolerance_m
                and ori_error <= self.settings.orientation_tolerance_rad
            ):
                values = q[self.bindings.qpos_ids]
                status = IKStatus.AT_TARGET
            else:
                velocity = mink.solve_ik(
                    self.configuration,
                    self.tasks,
                    dt_s,
                    solver="daqp",
                    damping=self.settings.damping,
                    safety_break=True,
                    limits=self.limits,
                    constraints=self.constraints,
                )
                if not np.all(np.isfinite(velocity)):
                    raise ValueError("Solver returned nonfinite velocity")
                self.configuration.integrate_inplace(velocity, dt_s)
                values = self.configuration.q[self.bindings.qpos_ids]
                step = np.abs(values - q[self.bindings.qpos_ids])
                # 1e-9 is a numerical comparison allowance, not a widened model limit.
                if np.any(step > np.array(self.bindings.profile.velocity_limits) * dt_s + 1e-9):
                    raise ValueError("Solver step exceeds configured velocity limit")
                self.bindings.validate_arm(values)
                status = IKStatus.VALID_STEP
            return result(status, dict(zip(self.bindings.profile.arm_joints, map(float, values))))
        except mink.NotWithinConfigurationLimits as exc:
            return result(IKStatus.LIMIT_VIOLATION, detail=str(exc))
        except (mink.NoSolutionFound, SolverError, ValueError, np.linalg.LinAlgError) as exc:
            return result(IKStatus.SOLVER_FAILED, detail=str(exc))
