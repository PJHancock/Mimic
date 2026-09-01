"""The position-reference layer is independent of Panda joint names and count."""

from dataclasses import replace

import numpy as np
import pytest

from mimic.common.types import IKStatus, ToolPose
from mimic.robot.backends.ruckig_position import (
    PositionTrajectorySettings,
    RuckigPositionIK,
)

pytest_plugins = ("tests.test_robot.execution_fixtures",)


def settings(*, hardware=(1.0, 1.0), tracking=(0.2, 0.2)):
    return PositionTrajectorySettings(hardware, (1.0, 1.0), (10.0, 10.0), tracking, 1000)


def test_trajectory_reaches_target_through_position_servo(small_robot):
    bindings, io, differential_ik = small_robot
    solver = RuckigPositionIK(differential_ik, bindings, settings())
    target = ToolPose((0.1, 0, 0.1), (1, 0, 0, 0))
    commands = [np.zeros(2)]

    for _ in range(1000):
        result = solver.solve(target, io.read(), 0.01)
        assert result.valid, result.detail
        command = np.array([result.joint_targets[name] for name in bindings.profile.arm_joints])
        commands.append(command)
        io.apply(result.joint_targets, {"finger_motor": 0})
        io.advance(0.01)
        if result.status == IKStatus.AT_TARGET:
            break

    assert result.status == IKStatus.AT_TARGET
    assert np.linalg.norm(np.asarray(io.read().tool_pose.position) - target.position) < 1e-3
    velocity = np.diff(commands, axis=0) / 0.01
    acceleration = np.diff(np.vstack([np.zeros(2), velocity]), axis=0) / 0.01
    jerk = np.diff(np.vstack([np.zeros(2), acceleration]), axis=0) / 0.01
    assert np.max(np.abs(velocity)) <= 0.5 + 1e-8
    assert np.max(np.abs(acceleration)) <= 1.0 + 1e-8
    assert np.max(np.abs(jerk)) <= 10.0 + 1e-6


def test_saved_joint_target_uses_same_bounded_position_trajectory(small_robot):
    bindings, io, differential_ik = small_robot
    solver = RuckigPositionIK(differential_ik, bindings, settings())
    target = {"slide_x": 0.1, "slide_z": -0.05}
    tolerances = {"slide_x": 1e-3, "slide_z": 1e-3}
    for _ in range(1000):
        result = solver.solve_joint_target(target, tolerances, io.read(), 0.01)
        assert result.valid, result.detail
        io.apply(result.joint_targets, {"finger_motor": 0})
        io.advance(0.01)
        if result.status == IKStatus.AT_TARGET:
            break
    assert result.status == IKStatus.AT_TARGET
    measured = io.read().joint_positions
    assert measured["slide_x"][0] == pytest.approx(0.1, abs=1e-3)
    assert measured["slide_z"][0] == pytest.approx(-0.05, abs=1e-3)


def test_saved_joint_target_requires_exact_named_arm_contract(small_robot):
    bindings, io, differential_ik = small_robot
    solver = RuckigPositionIK(differential_ik, bindings, settings())
    result = solver.solve_joint_target({"slide_x": 0.1}, {"slide_x": 1e-3}, io.read(), 0.01)
    assert result.status == IKStatus.INVALID_INPUT
    assert not result.joint_targets


def test_stalled_servo_trips_per_joint_tracking_bound(small_robot):
    bindings, io, differential_ik = small_robot
    solver = RuckigPositionIK(differential_ik, bindings, settings(tracking=(0.001, 0.001)))
    target = ToolPose((0.1, 0, 0.1), (1, 0, 0, 0))
    state = io.read()
    for tick in range(1000):
        result = solver.solve(target, replace(state, timestamp_s=tick * 0.01), 0.01)
        if not result.valid:
            break
    assert result.status == IKStatus.LIMIT_VIOLATION
    assert "tracking-error bound" in result.detail


def test_measured_hardware_velocity_violation_fails_closed(small_robot):
    bindings, io, differential_ik = small_robot
    solver = RuckigPositionIK(differential_ik, bindings, settings())
    state = io.read()
    solver.solve(state.tool_pose, state, 0.01)
    joints = {**state.joint_positions, "slide_x": (0.02,)}
    moved = replace(state, timestamp_s=0.01, joint_positions=joints)
    result = solver.solve(state.tool_pose, moved, 0.01)
    assert result.status == IKStatus.LIMIT_VIOLATION
    assert "sourced joint limit" in result.detail


def test_fixed_interval_and_sourced_ceiling_are_enforced(small_robot):
    bindings, io, differential_ik = small_robot
    with pytest.raises(ValueError, match="Operating velocity"):
        RuckigPositionIK(differential_ik, bindings, settings(hardware=(0.4, 0.4)))

    solver = RuckigPositionIK(differential_ik, bindings, settings())
    state = io.read()
    assert solver.solve(state.tool_pose, state, 0.01).valid
    result = solver.solve(state.tool_pose, state, 0.02)
    assert result.status == IKStatus.INVALID_INPUT
    assert "fixed control interval" in result.detail


@pytest.mark.parametrize(
    "bad",
    [
        PositionTrajectorySettings((1,), (1,), (1,), (0.1,), 1),
        PositionTrajectorySettings((1, 1), (1, 1), (1, 1), (0.1,), 1),
    ],
)
def test_all_per_joint_settings_must_match_robot(small_robot, bad):
    bindings, _, differential_ik = small_robot
    with pytest.raises(ValueError, match="arm joint count"):
        RuckigPositionIK(differential_ik, bindings, bad)
