"""Deterministic skill expansion with measured gates; no geometry generation or IK math."""

from dataclasses import asdict, dataclass
from typing import Callable, Optional

import numpy as np

from mimic.common.types import ActionPhase, IKStatus, PickPlaceWaypoints, RobotState, ToolPose
from mimic.robot.controller import ExecutionFailure, RobotController
from mimic.robot.gripper import GripperAction, GripperStatus


@dataclass(frozen=True)
class ExecutionSettings:
    step_timeout_s: float
    minimum_lift_m: float
    maximum_slip_m: float
    contact_loss_timeout_s: float
    settle_time_s: float
    settled_speed_m_s: float
    placement_tolerance_m: float

    def __post_init__(self):
        if not np.all(np.isfinite(tuple(vars(self).values()))) or min(vars(self).values()) <= 0:
            raise ValueError("Execution criteria must be explicitly positive and finite")


@dataclass(frozen=True)
class SkillStep:
    phase: ActionPhase
    skill: str
    pose: ToolPose
    gripper_action: GripperAction


def expand_skills(task: PickPlaceWaypoints) -> tuple[SkillStep, ...]:
    """Subskills are execution metadata, never additional learned action labels."""
    return (
        SkillStep(ActionPhase.APPROACH, "HOVER", task.approach, GripperAction.OPEN),
        SkillStep(ActionPhase.GRASP, "DESCEND", task.grasp, GripperAction.OPEN),
        SkillStep(ActionPhase.GRASP, "CLOSE", task.grasp, GripperAction.CLOSE),
        SkillStep(ActionPhase.MOVE, "LIFT", task.lift, GripperAction.HOLD),
        *(
            SkillStep(ActionPhase.MOVE, "FOLLOW_PATH", pose, GripperAction.HOLD)
            for pose in task.path
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
    ):
        self.controller, self.settings = controller, settings
        self.record = record or (lambda event: None)

    def run(self, task: PickPlaceWaypoints) -> ExecutionReport:
        """Run one simulation attempt. A failure returns without further stepping/release."""
        self.controller.reset()
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
            for step_index, step in enumerate(expand_skills(task)):
                started = self.controller.io.read().timestamp_s
                self.record(
                    {
                        "event": "transition",
                        "timestamp_s": started,
                        "phase": step.phase.value,
                        "skill": step.skill,
                        "step": step_index,
                    }
                )
                while True:
                    sample = self.controller.prepare(step.pose, step.gripper_action)
                    state, grip = sample.state, sample.gripper
                    self.record(
                        {
                            "event": "sample",
                            "phase": step.phase.value,
                            "skill": step.skill,
                            "step": step_index,
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
                    carrying = step.skill in ("LIFT", "FOLLOW_PATH", "LOWER")
                    if carrying:
                        forces = state.gripper.finger_contact_forces_n
                        contact = (
                            len(forces) >= 2
                            and min(forces) >= self.controller.gripper.settings.contact_force_n
                        )
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
                            raise ExecutionFailure(f"{step.skill}: object slipped relative to tool")
                    complete = at_target
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
                    elif step.skill == "FOLLOW_PATH" and complete:
                        transported = grasped
                    # Never advance on a computed target alone: completion uses measured state.
                    if complete:
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
