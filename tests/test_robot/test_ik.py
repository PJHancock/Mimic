from dataclasses import replace

import mujoco
import numpy as np
import pytest

from mimic.common.types import IKStatus, ToolPose
from mimic.robot.model import ModelBindings

pytest_plugins = ("tests.test_robot.execution_fixtures",)


def test_named_targets_and_nonarm_isolation(small_robot):
    bindings, io, solver = small_robot
    state = io.read()
    original = io.data.qpos.copy()
    result = solver.solve(ToolPose((0.1, 0, 0.1), (1, 0, 0, 0)), state, 0.01)
    assert result.status == IKStatus.VALID_STEP
    assert set(result.joint_targets) == {"slide_x", "slide_z"}
    assert all(0 < v <= 0.005 + 1e-9 for v in result.joint_targets.values())
    np.testing.assert_array_equal(io.data.qpos, original)
    nonarm = np.setdiff1d(np.arange(bindings.model.nq), bindings.qpos_ids)
    np.testing.assert_allclose(solver.configuration.q[nonarm], original[nonarm], atol=1e-12)


def test_reachable_target_moves_through_physics(small_robot):
    _, io, solver = small_robot
    target = ToolPose((0.1, 0, 0.1), (1, 0, 0, 0))
    for _ in range(400):
        result = solver.solve(target, io.read(), 0.01)
        assert result.valid, result.detail
        io.apply(result.joint_targets, {"finger_motor": 0})
        io.advance(0.01)
    assert np.linalg.norm(np.array(io.read().tool_pose.position) - target.position) < 1e-3


@pytest.mark.parametrize("dt", [0, -1, float("nan")])
def test_invalid_interval_returns_no_targets(small_robot, dt):
    _, io, solver = small_robot
    result = solver.solve(io.read().tool_pose, io.read(), dt)
    assert result.status == IKStatus.INVALID_INPUT and not result.joint_targets


def test_out_of_workspace_rejected_without_clamping(small_robot):
    _, io, solver = small_robot
    result = solver.solve(ToolPose((3, 0, 0), (1, 0, 0, 0)), io.read(), 0.01)
    assert result.status == IKStatus.INVALID_INPUT


def test_unreachable_orientation_is_not_reported_as_reached(small_robot):
    _, io, solver = small_robot
    result = solver.solve(ToolPose((0, 0, 0), (0, 1, 0, 0)), io.read(), 0.01)
    # A local least-squares solve may return a zero-progress step; it must not
    # confuse that with arrival. The executor bounds repeated attempts by time.
    assert result.status != IKStatus.AT_TARGET
    assert result.orientation_error_rad == pytest.approx(np.pi)


def test_joint_limit_violation_does_not_command(small_robot):
    _, io, solver = small_robot
    state = io.read()
    broken = replace(state, joint_positions={**state.joint_positions, "slide_x": (1.1,)})
    result = solver.solve(state.tool_pose, broken, 0.01)
    assert result.status == IKStatus.LIMIT_VIOLATION and not result.joint_targets


def test_torque_actuator_rejected(small_robot):
    bindings, _, _ = small_robot
    bindings.model.actuator_biastype[bindings.actuator_ids[0]] = mujoco.mjtBias.mjBIAS_NONE
    with pytest.raises(ValueError, match="position servo"):
        ModelBindings(bindings.model, bindings.profile)


def test_arm_and_gripper_commands_validate_atomically(small_robot):
    _, io, _ = small_robot
    previous = io.data.ctrl.copy()
    with pytest.raises(ValueError):
        io.apply({"slide_x": 0.1, "slide_z": 0.1}, {"finger_motor": 1})
    np.testing.assert_array_equal(previous, io.data.ctrl)
    io.apply({"slide_x": 0.1, "slide_z": 0.2}, {"finger_motor": 0.03})
    np.testing.assert_allclose(io.data.ctrl, [0.03, 0.2, 0.1])


def test_panda_fk_target_and_private_solve(panda):
    bindings, io, solver, _ = panda
    current = io.read()
    desired_q = io.data.qpos.copy()
    desired_q[bindings.qpos_ids[0]] += 0.08
    scratch = mujoco.MjData(bindings.model)
    scratch.qpos[:] = desired_q
    mujoco.mj_forward(bindings.model, scratch)
    target = ToolPose(tuple(scratch.xpos[bindings.body_id]), tuple(scratch.xquat[bindings.body_id]))
    q = io.data.qpos.copy()
    for _ in range(100):
        positions = {n: tuple(q[s]) for n, s in bindings.joint_slices.items()}
        result = solver.solve(target, replace(current, joint_positions=positions), 0.01)
        assert result.valid, result.detail
        if result.status == IKStatus.AT_TARGET:
            break
        q[bindings.qpos_ids] = [result.joint_targets[n] for n in bindings.profile.arm_joints]
    assert result.status == IKStatus.AT_TARGET
    assert result.position_error_m < 1e-4
    np.testing.assert_array_equal(io.data.qpos, bindings.model.key_qpos[0])


def test_observation_reflects_post_step_state(small_robot):
    _, io, _ = small_robot
    io.apply({"slide_x": 0.1, "slide_z": 0.1}, {"finger_motor": 0})
    io.advance(0.01)
    observed = io.read()
    assert observed.tool_pose.position[0] == pytest.approx(observed.joint_positions["slide_x"][0])
