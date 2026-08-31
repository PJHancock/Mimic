"""Observed compliance must never enlarge commands or overwrite measured coordinates."""

from dataclasses import replace

import numpy as np
import pytest

from mimic.common.types import IKStatus, ToolPose
from mimic.robot.backends.mink_ik import MinkIKSolver

pytest_plugins = ("tests.test_robot.execution_fixtures",)


def tolerant_solver(bindings, settings, reference, tolerances):
    return MinkIKSolver(bindings, settings, reference, measured_gripper_tolerances_m=tolerances)


@pytest.mark.parametrize("finger_position", [-1e-5, -3.33e-6, 0.1 + 3.33e-6, 0.1 + 1e-5])
def test_accepted_observation_remains_raw_and_frozen(small_robot, ik_settings, finger_position):
    bindings, io, original_solver = small_robot
    state = replace(
        io.read(), joint_positions={**io.read().joint_positions, "finger": (finger_position,)}
    )
    model_ranges = bindings.model.jnt_range.copy()
    actuator_ranges = bindings.model.actuator_ctrlrange.copy()
    live_q = io.data.qpos.copy()
    solver = tolerant_solver(bindings, ik_settings, {"slide_x": 0, "slide_z": 0}, {"finger": 1e-5})
    target = ToolPose((0.1, 0, 0.1), (1, 0, 0, 0))
    assert original_solver.solve(target, state, 0.01).status == IKStatus.LIMIT_VIOLATION
    result = solver.solve(target, state, 0.01)
    assert result.status == IKStatus.VALID_STEP, result.detail
    finger_id = bindings.model.joint("finger").qposadr[0]
    assert solver.configuration.q[finger_id] == pytest.approx(finger_position, abs=1e-12)
    assert state.joint_positions["finger"] == (finger_position,)
    np.testing.assert_array_equal(io.data.qpos, live_q)
    np.testing.assert_array_equal(bindings.model.jnt_range, model_ranges)
    np.testing.assert_array_equal(bindings.model.actuator_ctrlrange, actuator_ranges)
    assert set(result.joint_targets) == {"slide_x", "slide_z"}
    assert all(0 < value <= 0.005 + 1e-9 for value in result.joint_targets.values())


@pytest.mark.parametrize("finger_position", [-1.0001e-5, 0.1 + 1.0001e-5])
def test_beyond_allowance_is_rejected_at_target_and_during_motion(
    small_robot, ik_settings, finger_position
):
    bindings, io, _ = small_robot
    state = replace(
        io.read(), joint_positions={**io.read().joint_positions, "finger": (finger_position,)}
    )
    solver = tolerant_solver(bindings, ik_settings, {"slide_x": 0, "slide_z": 0}, {"finger": 1e-5})
    for pose in (state.tool_pose, ToolPose((0.1, 0, 0.1), (1, 0, 0, 0))):
        result = solver.solve(pose, state, 0.01)
        assert result.status == IKStatus.LIMIT_VIOLATION
        assert not result.joint_targets


def test_arm_and_actuator_limits_remain_strict(small_robot, ik_settings):
    bindings, io, _ = small_robot
    solver = tolerant_solver(bindings, ik_settings, {"slide_x": 0, "slide_z": 0}, {"finger": 1e-5})
    state = io.read()
    arm_violation = replace(
        state, joint_positions={**state.joint_positions, "slide_x": (1 + 1e-8,)}
    )
    assert solver.solve(state.tool_pose, arm_violation, 0.01).status == IKStatus.LIMIT_VIOLATION
    near_limit = replace(state, joint_positions={**state.joint_positions, "slide_x": (0.999,)})
    result = solver.solve(ToolPose((1.1, 0, 0), (1, 0, 0, 0)), near_limit, 0.01)
    assert result.valid, result.detail
    assert result.joint_targets["slide_x"] <= 1
    before = io.data.ctrl.copy()
    with pytest.raises(ValueError, match="Invalid control"):
        io.apply({"slide_x": 0, "slide_z": 0}, {"finger_motor": 0.1 + 3.33e-6})
    np.testing.assert_array_equal(io.data.ctrl, before)


@pytest.mark.parametrize(
    "name,tolerance",
    [
        ("slide_x", 1e-5),
        ("object_free", 1e-5),
        ("finger", -1e-5),
        ("finger", 0),
        ("finger", float("nan")),
        ("finger", float("inf")),
        ("finger", True),
        ("finger", 0.1),
    ],
)
def test_invalid_tolerance_scope_or_value_rejected(small_robot, ik_settings, name, tolerance):
    bindings, _, _ = small_robot
    with pytest.raises(ValueError):
        tolerant_solver(bindings, ik_settings, {"slide_x": 0, "slide_z": 0}, {name: tolerance})


def test_panda_observed_excursion_is_accepted_without_weakening_the_other_finger(
    panda, ik_settings
):
    bindings, io, original_solver, driver = panda
    state = io.read()
    reference = {name: state.joint_positions[name][0] for name in bindings.profile.arm_joints}
    solver = tolerant_solver(bindings, ik_settings, reference, {"finger_joint2": 1e-5})
    observed = replace(
        state, joint_positions={**state.joint_positions, "finger_joint2": (0.040003326573192846,)}
    )
    assert original_solver.solve(state.tool_pose, observed, 0.01).status == IKStatus.LIMIT_VIOLATION
    assert solver.solve(state.tool_pose, observed, 0.01).status == IKStatus.AT_TARGET
    wrong_finger = replace(
        state, joint_positions={**state.joint_positions, "finger_joint1": (0.040003326573192846,)}
    )
    assert solver.solve(state.tool_pose, wrong_finger, 0.01).status == IKStatus.LIMIT_VIOLATION
    assert driver.controls(0.08) == {"actuator8": 255}
    with pytest.raises(ValueError):
        driver.controls(0.08 + 1e-8)
