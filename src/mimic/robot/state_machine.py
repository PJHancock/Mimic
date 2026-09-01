"""Deterministic skill expansion with measured gates; no geometry generation or IK math."""

from dataclasses import asdict, dataclass, replace
from typing import Callable, Mapping, Optional, Tuple

import numpy as np

from mimic.common.types import ActionPhase, IKStatus, PickPlaceWaypoints, RobotState, ToolPose
from mimic.robot.controller import ExecutionFailure, RobotController
from mimic.robot.gripper import GripperAction, GripperStatus
from mimic.robot.presets import JointPreset


@dataclass(frozen=True)
class ExecutionSettings:
    step_timeout_s: float
    minimum_lift_m: float
    maximum_slip_m: float
    waypoint_handoff_radius_m: float
    contact_loss_timeout_s: float
    settle_time_s: float
    settled_speed_m_s: float
    placement_tolerance_m: float
    placement_approach_clearance_m: float
    placement_maximum_descent_speed_m_s: float
    placement_maximum_descent_acceleration_m_s2: float
    placement_contact_confirmation_s: float

    def __post_init__(self):
        if not np.all(np.isfinite(tuple(vars(self).values()))) or min(vars(self).values()) <= 0:
            raise ValueError("Execution criteria must be explicitly positive and finite")
        if self.placement_contact_confirmation_s >= self.step_timeout_s:
            raise ValueError("Placement contact confirmation must be shorter than step timeout")


@dataclass(frozen=True)
class SkillStep:
    phase: ActionPhase
    skill: str
    pose: ToolPose
    gripper_action: GripperAction
    waypoint_index: Optional[int] = None
    waypoint_count: Optional[int] = None

    @property
    def is_intermediate_waypoint(self) -> bool:
        return (
            self.skill == "FOLLOW_PATH"
            and self.waypoint_index is not None
            and self.waypoint_count is not None
            and self.waypoint_index < self.waypoint_count - 1
        )


def expand_skills(
    task: PickPlaceWaypoints, placement_approach_clearance_m: float
) -> tuple[SkillStep, ...]:
    """Subskills are execution metadata, never additional learned action labels."""
    lower_x, lower_y, lower_z = task.lower.position
    placement_approach = replace(
        task.lower,
        position=(lower_x, lower_y, lower_z + placement_approach_clearance_m),
    )
    return (
        SkillStep(ActionPhase.HOVER, "HOVER", task.approach, GripperAction.OPEN),
        SkillStep(ActionPhase.GRASP, "DESCEND", task.grasp, GripperAction.OPEN),
        SkillStep(ActionPhase.GRASP, "CLOSE", task.grasp, GripperAction.CLOSE),
        SkillStep(ActionPhase.CARRY, "LIFT", task.lift, GripperAction.HOLD),
        *(
            SkillStep(
                ActionPhase.CARRY,
                "FOLLOW_PATH",
                pose,
                GripperAction.HOLD,
                waypoint_index=index,
                waypoint_count=len(task.path),
            )
            for index, pose in enumerate(task.path)
        ),
        SkillStep(
            ActionPhase.RELEASE,
            "PLACE_APPROACH",
            placement_approach,
            GripperAction.HOLD,
        ),
        SkillStep(ActionPhase.RELEASE, "LOWER", task.lower, GripperAction.HOLD),
        SkillStep(ActionPhase.RELEASE, "OPEN", task.lower, GripperAction.OPEN),
        SkillStep(ActionPhase.RELEASE, "RETREAT", task.retreat, GripperAction.HOLD),
    )


@dataclass(frozen=True)
class ExecutionReport:
    success: bool
    grasp_occurred: bool
    transported: bool
    released: bool
    final_position_error_m: Optional[float]
    failure: Optional[str]
    final_state: RobotState


class SkillExecutor:
    def __init__(
        self,
        controller: RobotController,
        settings: ExecutionSettings,
        record: Optional[Callable[[dict], None]] = None,
        home_preset: Optional[JointPreset] = None,
        preset_position_tolerances: Optional[Mapping[str, float]] = None,
        initialize_object: Optional[Callable[[Tuple[float, float, float]], RobotState]] = None,
        support_contact: Optional[Callable[[], bool]] = None,
    ):
        self.controller, self.settings = controller, settings
        self.record = record or (lambda event: None)
        self.home_preset = home_preset
        self.initialize_object = initialize_object
        self.support_contact = support_contact
        self.preset_position_tolerances = (
            dict(preset_position_tolerances) if preset_position_tolerances is not None else None
        )

    def return_home(self) -> RobotState:
        """Move to the configured full arm preset through the bounded joint trajectory."""
        if self.home_preset is None or self.preset_position_tolerances is None:
            raise ValueError("Home preset and measured joint tolerances must be configured")
        started = self.controller.io.read().timestamp_s
        self.record(
            {
                "event": "transition",
                "timestamp_s": started,
                "phase": ActionPhase.HOVER.value,
                "skill": "MOVE_TO_HOME",
                "preset_id": self.home_preset.preset_id,
            }
        )
        while True:
            sample = self.controller.prepare_joint(
                self.home_preset.joint_positions,
                self.preset_position_tolerances,
                GripperAction.HOLD,
            )
            if sample.state.timestamp_s - started >= self.settings.step_timeout_s:
                raise ExecutionFailure("MOVE_TO_HOME: target not achieved before timeout")
            if sample.ik.status == IKStatus.AT_TARGET:
                return sample.state
            self.controller.commit(sample)

    def run(self, task: PickPlaceWaypoints) -> ExecutionReport:
        """Run one simulation attempt. A failure returns without further stepping/release."""
        if self.support_contact is None:
            raise ValueError("Guarded placement requires a support-contact observer")
        self.controller.reset()
        if self.initialize_object is not None:
            initialized = self.initialize_object(task.grasp.position)
            self.record(
                {
                    "event": "object_initialized",
                    "timestamp_s": initialized.timestamp_s,
                    "position": initialized.object_position,
                }
            )
        initial = self.controller.io.read()
        if initial.object_position is None:
            raise ValueError("Pick-and-place verification requires a named simulated object")
        initial_z = initial.object_position[2]
        grasped = transported = released = False
        grasp_offset = None
        contact_lost_at = None
        failure = None
        self.record(
            {
                "event": "start",
                "timestamp_s": initial.timestamp_s,
                "execution_settings": asdict(self.settings),
            }
        )
        try:
            for step_index, step in enumerate(
                expand_skills(task, self.settings.placement_approach_clearance_m)
            ):
                started = self.controller.io.read().timestamp_s
                guarded_reference_z = (
                    step.pose.position[2] + self.settings.placement_approach_clearance_m
                    if step.skill == "LOWER"
                    else step.pose.position[2]
                )
                guarded_reference_speed_m_s = 0.0
                previous_lower_tool_z = None
                previous_lower_object_position = None
                previous_lower_timestamp = None
                support_hold_pose = None
                support_stable_since = None
                waypoint_metadata = (
                    {
                        "waypoint_index": step.waypoint_index,
                        "waypoint_count": step.waypoint_count,
                    }
                    if step.waypoint_index is not None
                    else {}
                )
                self.record(
                    {
                        "event": "transition",
                        "timestamp_s": started,
                        "phase": step.phase.value,
                        "skill": step.skill,
                        "step": step_index,
                        **waypoint_metadata,
                    }
                )
                while True:
                    target_pose = step.pose
                    support_observed = False
                    measured_descent_speed_m_s = 0.0
                    if step.skill == "LOWER":
                        observed = self.controller.io.read()
                        support_observed = self.support_contact()
                        if previous_lower_tool_z is not None:
                            elapsed = observed.timestamp_s - previous_lower_timestamp
                            if elapsed > 0:
                                measured_descent_speed_m_s = max(
                                    0.0,
                                    (previous_lower_tool_z - observed.tool_pose.position[2])
                                    / elapsed,
                                )
                                if (
                                    measured_descent_speed_m_s
                                    > self.settings.placement_maximum_descent_speed_m_s + 1e-9
                                ):
                                    raise ExecutionFailure(
                                        "LOWER: measured descent speed exceeds configured maximum"
                                    )
                        previous_lower_tool_z = observed.tool_pose.position[2]
                        previous_lower_timestamp = observed.timestamp_s
                        if support_observed:
                            if support_hold_pose is None:
                                support_hold_pose = observed.tool_pose
                                self.record(
                                    {
                                        "event": "placement_contact",
                                        "timestamp_s": observed.timestamp_s,
                                        "tool_position": observed.tool_pose.position,
                                        "object_position": observed.object_position,
                                        "measured_descent_speed_m_s": measured_descent_speed_m_s,
                                    }
                                )
                            target_pose = support_hold_pose
                        else:
                            if support_hold_pose is not None:
                                self.record(
                                    {
                                        "event": "placement_contact_lost",
                                        "timestamp_s": observed.timestamp_s,
                                    }
                                )
                            support_hold_pose = None
                            support_stable_since = None
                            guarded_reference_speed_m_s = min(
                                self.settings.placement_maximum_descent_speed_m_s,
                                guarded_reference_speed_m_s
                                + self.settings.placement_maximum_descent_acceleration_m_s2
                                * self.controller.dt_s,
                            )
                            descent_increment_speed_m_s = min(
                                guarded_reference_speed_m_s,
                                max(
                                    0.0,
                                    self.settings.placement_maximum_descent_speed_m_s
                                    - measured_descent_speed_m_s,
                                ),
                            )
                            # Keep the moving Cartesian target at most one descent
                            # increment below measurement. This prevents the endpoint
                            # joint trajectory from accumulating a surface-normal lead;
                            # measured downward speed consumes the available increment.
                            guarded_reference_z = max(
                                step.pose.position[2],
                                min(guarded_reference_z, observed.tool_pose.position[2])
                                - descent_increment_speed_m_s * self.controller.dt_s,
                            )
                            target_pose = replace(
                                step.pose,
                                position=(
                                    step.pose.position[0],
                                    step.pose.position[1],
                                    guarded_reference_z,
                                ),
                            )
                    sample = self.controller.prepare(target_pose, step.gripper_action)
                    state, grip = sample.state, sample.gripper
                    self.record(
                        {
                            "event": "sample",
                            "phase": step.phase.value,
                            "skill": step.skill,
                            "step": step_index,
                            **waypoint_metadata,
                            "target_pose": asdict(target_pose),
                            **(
                                {
                                    "support_contact": support_observed,
                                    "measured_descent_speed_m_s": measured_descent_speed_m_s,
                                }
                                if step.skill == "LOWER"
                                else {}
                            ),
                            **asdict(sample),
                        }
                    )
                    if state.timestamp_s - started >= self.settings.step_timeout_s:
                        raise ExecutionFailure(f"{step.skill}: target not achieved before timeout")
                    if grip.status in (GripperStatus.EMPTY, GripperStatus.TIMEOUT):
                        # On entry to OPEN, stale close feedback must not prevent the scheduled
                        # opening; failures while actively carrying still stop the attempt.
                        if not (step.skill == "OPEN" and grip.action == GripperAction.CLOSE):
                            raise ExecutionFailure(f"{step.skill}: gripper {grip.status.value}")
                    at_target = sample.ik.status == IKStatus.AT_TARGET
                    if state.object_position is None:
                        raise ExecutionFailure("Object observation disappeared")
                    object_position = np.array(state.object_position)
                    tool_position = np.array(state.tool_pose.position)
                    carrying = step.skill in (
                        "LIFT",
                        "FOLLOW_PATH",
                        "PLACE_APPROACH",
                        "LOWER",
                    )
                    if carrying:
                        forces = state.gripper.finger_contact_forces_n
                        contact = (
                            len(forces) >= 2
                            and min(forces) >= self.controller.gripper.settings.contact_force_n
                        )
                        # Support contact intentionally changes the airborne rigid-
                        # grasp offset. Free-space contact/slip gates end at support
                        # onset instead of classifying placement as a dropped object.
                        if step.skill == "LOWER" and support_observed:
                            contact_lost_at = None
                        else:
                            if contact:
                                contact_lost_at = None
                            elif contact_lost_at is None:
                                contact_lost_at = state.timestamp_s
                            elif (
                                state.timestamp_s - contact_lost_at
                                >= self.settings.contact_loss_timeout_s
                            ):
                                raise ExecutionFailure(f"{step.skill}: target-object contact lost")
                            if (
                                np.linalg.norm(object_position - tool_position - grasp_offset)
                                > self.settings.maximum_slip_m
                            ):
                                raise ExecutionFailure(
                                    f"{step.skill}: object slipped relative to tool"
                                )
                    complete = at_target
                    commit_handoff = False
                    if step.skill in ("HOVER", "DESCEND", "OPEN", "RETREAT"):
                        complete = complete and grip.status == GripperStatus.OPEN
                    elif step.skill == "CLOSE":
                        complete = complete and grip.status == GripperStatus.CANDIDATE_GRASP
                        if complete:
                            grasp_offset = object_position - tool_position
                            self.record(
                                {"event": "candidate_grasp", "timestamp_s": state.timestamp_s}
                            )
                    elif step.skill == "LIFT":
                        complete = (
                            complete
                            and contact
                            and object_position[2] - initial_z >= self.settings.minimum_lift_m
                        )
                        if complete:
                            grasped = True
                    elif step.skill == "FOLLOW_PATH":
                        waypoint_distance_m = float(
                            np.linalg.norm(tool_position - np.asarray(step.pose.position))
                        )
                        if step.is_intermediate_waypoint:
                            complete = (
                                waypoint_distance_m <= self.settings.waypoint_handoff_radius_m
                            )
                            if complete:
                                commit_handoff = True
                                self.record(
                                    {
                                        "event": "waypoint_handoff",
                                        "timestamp_s": state.timestamp_s,
                                        "phase": step.phase.value,
                                        "skill": step.skill,
                                        "step": step_index,
                                        **waypoint_metadata,
                                        "distance_m": waypoint_distance_m,
                                        "radius_m": self.settings.waypoint_handoff_radius_m,
                                        "ik_status": sample.ik.status.value,
                                    }
                                )
                        elif complete:
                            transported = grasped
                    elif step.skill == "LOWER":
                        complete = False
                        object_speed_m_s = float("inf")
                        if previous_lower_object_position is not None:
                            elapsed = state.timestamp_s - previous_lower_object_position[0]
                            if elapsed > 0:
                                object_speed_m_s = float(
                                    np.linalg.norm(
                                        object_position - previous_lower_object_position[1]
                                    )
                                    / elapsed
                                )
                        previous_lower_object_position = (
                            state.timestamp_s,
                            object_position.copy(),
                        )
                        stable_support = (
                            support_observed and object_speed_m_s <= self.settings.settled_speed_m_s
                        )
                        support_stable_since = (
                            (
                                support_stable_since
                                if support_stable_since is not None
                                else state.timestamp_s
                            )
                            if stable_support
                            else None
                        )
                        if (
                            support_stable_since is not None
                            and state.timestamp_s - support_stable_since
                            >= self.settings.placement_contact_confirmation_s
                        ):
                            placement_error_m = float(
                                np.linalg.norm(object_position - np.asarray(task.goal_position))
                            )
                            if placement_error_m > self.settings.placement_tolerance_m:
                                raise ExecutionFailure(
                                    "LOWER: supported object outside placement tolerance"
                                )
                            complete = True
                            self.record(
                                {
                                    "event": "placement_supported",
                                    "timestamp_s": state.timestamp_s,
                                    "object_speed_m_s": object_speed_m_s,
                                    "placement_error_m": placement_error_m,
                                }
                            )
                    # Never advance on a computed target alone: completion uses measured state.
                    if complete:
                        # Proximity handoff can occur while Ruckig still has a moving
                        # reference. Apply that tick so its persistent state stays
                        # synchronized with the measured arm before target replacement.
                        if commit_handoff:
                            self.controller.commit(sample)
                        break
                    self.controller.commit(sample)

            # Continue physical stepping at the retreat pose until the object is settled.
            settling_started = self.controller.io.read().timestamp_s
            stable_since = None
            previous = self.controller.io.read()
            while True:
                sample = self.controller.prepare(task.retreat, GripperAction.HOLD)
                state = sample.state
                elapsed = state.timestamp_s - previous.timestamp_s
                speed = (
                    np.linalg.norm(np.array(state.object_position) - previous.object_position)
                    / elapsed
                    if elapsed > 0
                    else float("inf")
                )
                detached = all(
                    f < self.controller.gripper.settings.contact_force_n
                    for f in state.gripper.finger_contact_forces_n
                )
                stable = (
                    speed <= self.settings.settled_speed_m_s
                    and detached
                    and sample.gripper.status == GripperStatus.OPEN
                )
                stable_since = (
                    (stable_since if stable_since is not None else state.timestamp_s)
                    if stable
                    else None
                )
                if (
                    stable_since is not None
                    and state.timestamp_s - stable_since >= self.settings.settle_time_s
                ):
                    released = True
                    break
                if state.timestamp_s - settling_started >= self.settings.step_timeout_s:
                    raise ExecutionFailure("Object did not detach and settle after release")
                previous = state
                self.controller.commit(sample)
        except (ExecutionFailure, RuntimeError, ValueError) as exc:
            failure = str(exc)
            self.record({"event": "failure", "reason": failure})
        final = self.controller.io.read()
        error = float(np.linalg.norm(np.array(final.object_position) - task.goal_position))
        success = (
            failure is None
            and grasped
            and transported
            and released
            and error <= self.settings.placement_tolerance_m
        )
        if failure is None and not success:
            failure = "Final placement did not meet the configured acceptance criteria"
        report = ExecutionReport(success, grasped, transported, released, error, failure, final)
        self.record({"event": "result", **asdict(report)})
        return report
