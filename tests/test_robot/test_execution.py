"""Sequencing fakes test contracts only; they are not evidence of simulated grasp."""

from dataclasses import replace

import pytest

from mimic.common.types import (
    ActionPhase,
    GripperFeedback,
    IKResult,
    IKStatus,
    PickPlaceWaypoints,
    RobotCommand,
    RobotState,
    ToolPose,
)
from mimic.robot.commands import command_target
from mimic.robot.controller import ExecutionFailure, RobotController
from mimic.robot.action_primitives import CartesianMotion
from mimic.robot.factory import build_executor
from mimic.robot.gripper import GripperAction, GripperLogic, GripperSettings
from mimic.robot.state_machine import ExecutionSettings, SkillExecutor


def pose(x=0, z=0):
    return ToolPose((x, 0, z), (1, 0, 0, 0))


def task():
    return PickPlaceWaypoints(
        pose(z=0.1), pose(), pose(z=0.1), (pose(0.2, 0.1),), pose(0.2), pose(0.2, 0.1), (0.2, 0, 0)
    )


class Driver:
    open_width_m, open_command_width_m, closed_width_m = 0.1, 0.1, 0
    actuator_names = ("grip",)

    def controls(self, width_m):
        return {"grip": width_m}


class ScriptedIO:
    def __init__(self, empty=False):
        self.state = RobotState(
            0, {"x": (0,), "z": (0.1,)}, pose(z=0.1), GripperFeedback(0.1, 0, (0, 0)), (0, 0, 0)
        )
        self.empty = empty
        self.grasping = False
        self.advance_calls = 0

    def read(self):
        return self.state

    def apply(self, arm_targets, gripper_targets):
        next_pose = pose(arm_targets["x"], arm_targets["z"])
        self.grasping = gripper_targets["grip"] == 0 and not self.empty
        feedback = (
            GripperFeedback(0.04, 0, (1, 1))
            if self.grasping
            else GripperFeedback(gripper_targets["grip"], 0, (0, 0))
        )
        self.state = replace(
            self.state,
            tool_pose=next_pose,
            gripper=feedback,
            object_position=next_pose.position if self.grasping else self.state.object_position,
        )

    def advance(self, duration_s):
        self.advance_calls += 1
        self.state = replace(self.state, timestamp_s=self.state.timestamp_s + duration_s)


class ScriptedIK:
    def solve(self, target, state, dt_s):
        return IKResult(
            IKStatus.AT_TARGET if target == state.tool_pose else IKStatus.VALID_STEP,
            {"x": target.position[0], "z": target.position[2]},
            0,
            0,
            0,
        )


def executor(io, ik=None):
    events = []
    controller = RobotController(
        io,
        ik or ScriptedIK(),
        GripperLogic(Driver(), GripperSettings(0.001, 0.002, 0.1, 0.01, 1)),
        100,
        10,
    )
    return (
        SkillExecutor(
            controller, ExecutionSettings(2, 0.05, 0.01, 0.1, 0.03, 0.01, 0.01), events.append
        ),
        events,
    )


def test_all_phases_have_feedback_gated_subskills():
    io = ScriptedIO()
    runner, events = executor(io)
    result = runner.run(task())
    assert result.success and result.grasp_occurred and result.transported and result.released
    transitions = [e["skill"] for e in events if e["event"] == "transition"]
    assert transitions == [
        "HOVER",
        "DESCEND",
        "CLOSE",
        "LIFT",
        "FOLLOW_PATH",
        "LOWER",
        "OPEN",
        "RETREAT",
    ]
    assert {e["phase"] for e in events if e["event"] == "transition"} == {
        p.value for p in ActionPhase if p != ActionPhase.IDLE
    }


def test_empty_grasp_stops_before_lift_and_does_not_auto_release():
    io = ScriptedIO(empty=True)
    runner, events = executor(io)
    result = runner.run(task())
    assert not result.success and not result.grasp_occurred
    assert "EMPTY" in result.failure
    assert [e["skill"] for e in events if e["event"] == "transition"] == [
        "HOVER",
        "DESCEND",
        "CLOSE",
    ]
    assert io.read().gripper.width_m == 0


def test_failed_ik_never_writes_or_steps():
    class FailingIK:
        def solve(self, target, state, dt_s):
            return IKResult(IKStatus.SOLVER_FAILED, {}, None, None, 0, "deliberate failure")

    io = ScriptedIO()
    runner, _ = executor(io, FailingIK())
    result = runner.run(task())
    assert "deliberate failure" in result.failure
    assert io.advance_calls == 0 and io.read().gripper.width_m == 0.1


def test_no_progress_times_out_without_skipping_waypoint():
    class StalledIK:
        def solve(self, target, state, dt_s):
            return IKResult(IKStatus.VALID_STEP, {"x": 0, "z": 0.1}, 1, 0, 0)

    io = ScriptedIO()
    runner, events = executor(io, StalledIK())
    result = runner.run(task())
    assert "timeout" in result.failure.lower()
    assert [e["skill"] for e in events if e["event"] == "transition"] == ["HOVER"]


def test_gripper_has_own_rate_even_across_skill_transitions():
    runner, events = executor(ScriptedIO())
    result = runner.run(task())
    assert result.success
    commands = [e for e in events if e["event"] == "sample"]
    previous = commands[0]["gripper"]["actuator_targets"]
    for sample in commands[1:]:
        targets = sample["gripper"]["actuator_targets"]
        if targets != previous:
            ticks = sample["state"]["timestamp_s"] * 10
            assert ticks == pytest.approx(round(ticks))
        previous = targets


def test_existing_command_requires_declared_quaternion_order():
    command = RobotCommand(ActionPhase.GRASP, (0.5, 0, 0.1), (0, 0, 0, 1), False)
    target, action = command_target(
        command, fixed_orientation_wxyz=(0, 1, 0, 0), quaternion_order="xyzw"
    )
    assert target.quaternion_wxyz == (1, 0, 0, 0) and action.value == "CLOSE"
    command.target_orientation = None
    target, _ = command_target(
        command, fixed_orientation_wxyz=(0, 1, 0, 0), quaternion_order="xyzw"
    )
    assert target.quaternion_wxyz == (0, 1, 0, 0)


def test_configuration_does_not_guess_unresolved_parameters():
    from pathlib import Path

    config = Path(__file__).resolve().parents[2] / "configs/robots/panda.yaml"
    with pytest.raises(ValueError, match="Unresolved setting"):
        build_executor(config)


def test_invalid_pose_and_orientation_changes_rejected():
    with pytest.raises(ValueError, match="unit"):
        ToolPose((0, 0, 0), (2, 0, 0, 0))
    with pytest.raises(ValueError, match="fixed"):
        replace(task(), lower=ToolPose((0.2, 0, 0), (0, 1, 0, 0)))


def test_control_sample_cannot_be_reused():
    runner, _ = executor(ScriptedIO())
    controller = runner.controller

    sample = controller.prepare(pose(z=0.1), GripperAction.OPEN)
    controller.commit(sample)
    with pytest.raises(ExecutionFailure, match="stale"):
        controller.commit(sample)


def test_controller_dispatches_registry_cartesian_action():
    runner, _ = executor(ScriptedIO())
    action = CartesianMotion("TEST", pose(z=0.1), GripperAction.OPEN)
    sample = runner.controller.prepare_action(action)
    assert sample.ik.status == IKStatus.AT_TARGET


def test_dropped_object_fails_transport_without_releasing():
    class DropIO(ScriptedIO):
        def apply(self, arm_targets, gripper_targets):
            super().apply(arm_targets, gripper_targets)
            if arm_targets["x"] > 0:
                self.state = replace(self.state, object_position=(0, 0, 0))

    runner, _ = executor(DropIO())
    result = runner.run(task())
    assert not result.success and result.grasp_occurred and not result.released
    assert "slipped" in result.failure


def test_open_timeout_does_not_count_as_release():
    class StuckClosedIO(ScriptedIO):
        def apply(self, arm_targets, gripper_targets):
            if self.grasping and gripper_targets["grip"] > 0:
                return
            super().apply(arm_targets, gripper_targets)

    runner, _ = executor(StuckClosedIO())
    result = runner.run(task())
    assert result.grasp_occurred and not result.released and not result.success
    assert result.failure == "OPEN: gripper TIMEOUT"
