import numpy as np
import pytest

from mimic.common.types import GripperFeedback
from mimic.robot.backends.panda_gripper import PandaGripperDriver
from mimic.robot.gripper import GripperAction, GripperLogic, GripperSettings, GripperStatus

pytest_plugins = ("tests.test_robot.execution_fixtures",)


class OtherGripper:
    """Independent driver fixture: different opening, actuator name, and control scale."""

    open_width_m, closed_width_m = 0.1, 0
    actuator_names = ("other_motor",)

    def controls(self, width_m):
        return {"other_motor": width_m * 10}


@pytest.fixture
def logic():
    return GripperLogic(OtherGripper(), GripperSettings(0.001, 0.002, 0.1, 0.1, 1))


def test_close_command_does_not_mean_grasp(logic):
    out = logic.update(GripperAction.CLOSE, 0, GripperFeedback(0.1, 0, (0, 0)))
    assert out.status == GripperStatus.MOVING
    assert out.actuator_targets == {"other_motor": 0}
    assert (
        logic.update(GripperAction.CLOSE, 0.2, GripperFeedback(0, 0, (0, 0))).status
        == GripperStatus.EMPTY
    )


def test_requires_bilateral_sustained_target_contact(logic):
    logic.update(GripperAction.CLOSE, 0, GripperFeedback(0.04, 0, (1, 0)))
    assert (
        logic.update(GripperAction.CLOSE, 0.1, GripperFeedback(0.04, 0, (1, 1))).status
        == GripperStatus.MOVING
    )
    assert (
        logic.update(GripperAction.HOLD, 0.21, GripperFeedback(0.04, 0, (1, 1))).status
        == GripperStatus.CANDIDATE_GRASP
    )
    assert (
        logic.update(GripperAction.HOLD, 0.3, GripperFeedback(0.04, 0, (0, 1))).status
        == GripperStatus.MOVING
    )


def test_repeated_requests_do_not_reset_timeout(logic):
    feedback = GripperFeedback(0.05, 0, (0, 0))
    logic.update(GripperAction.CLOSE, 0, feedback)
    logic.update(GripperAction.CLOSE, 0.9, feedback)
    assert logic.update(GripperAction.CLOSE, 1.1, feedback).status == GripperStatus.TIMEOUT


def test_hold_requires_command_and_reset_clears_lifecycle(logic):
    with pytest.raises(ValueError, match="earlier"):
        logic.update(GripperAction.HOLD, 0, GripperFeedback(0.1, 0, (0, 0)))
    logic.update(GripperAction.OPEN, 2, GripperFeedback(0.1, 0, (0, 0)))
    with pytest.raises(ValueError, match="backwards"):
        logic.update(GripperAction.OPEN, 1, GripperFeedback(0.1, 0, (0, 0)))
    logic.reset()
    assert (
        logic.update(GripperAction.OPEN, 0, GripperFeedback(0.1, 0, (0, 0))).status
        == GripperStatus.OPEN
    )


def test_panda_mapping_and_rejection(panda):
    bindings, _, _, driver = panda
    for width, expected in [(0, 0), (0.04, 127.5), (0.08, 255)]:
        assert driver.controls(width)["actuator8"] == pytest.approx(expected)
    for width in [-0.01, 0.09, float("nan")]:
        with pytest.raises(ValueError):
            driver.controls(width)
    bindings.model.actuator_ctrlrange[bindings.model.actuator("actuator8").id] = [0, 0.04]
    with pytest.raises(ValueError, match="mapping mismatch"):
        PandaGripperDriver(bindings.model)


def test_panda_actual_finger_motion(panda):
    bindings, io, _, driver = panda
    arm = {n: io.read().joint_positions[n][0] for n in bindings.profile.arm_joints}
    io.apply(arm, driver.controls(0))
    io.advance(0.5)
    closed = io.read().gripper
    assert closed.width_m < 0.005
    assert closed.finger_contact_forces_n == (0, 0)
    io.apply(arm, driver.controls(0.08))
    io.advance(0.5)
    opened = io.read().gripper
    assert opened.width_m > 0.075
    np.testing.assert_allclose(io.data.ctrl[bindings.actuator_ids], list(arm.values()))
