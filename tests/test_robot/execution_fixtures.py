"""Numerical/physics fixtures, separate from concurrently developed task fixtures."""

from pathlib import Path

import mujoco
import pytest

from mimic.common.types import GripperFeedback, ToolPose
from mimic.robot.backends.mink_ik import MinkIKSolver
from mimic.robot.backends.panda_gripper import PandaGripperDriver
from mimic.robot.inverse_kinematics import IKSettings
from mimic.robot.model import ModelBindings, RobotProfile
from mimic.robot.simulation import MuJoCoAdapter

PANDA_XML = Path(__file__).resolve().parents[2] / "models/franka_emika_panda/upstream/panda.xml"


@pytest.fixture
def ik_settings():
    return IKSettings(1e-4, 1e-4, 1, 1, 0, 1e-4, 1)


@pytest.fixture
def small_robot(ik_settings):
    # Different joint count, reordered actuators, object before the arm, non-arm slider.
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><option timestep="0.002" gravity="0 0 0"/>
      <worldbody>
        <body name="object" pos="0 1 0"><freejoint name="object_free"/>
          <geom type="sphere" size="0.02" mass="0.1"/></body>
        <body name="carriage"><joint name="slide_x" type="slide" axis="1 0 0" range="-1 1"/>
          <geom type="sphere" size="0.03" mass="1"/>
          <body name="tool"><joint name="slide_z" type="slide" axis="0 0 1" range="-1 1"/>
            <geom type="sphere" size="0.02" mass="1"/></body></body>
        <body name="finger" pos="0 -1 0"><joint name="finger" type="slide" range="0 0.1"/>
          <geom type="sphere" size="0.01" mass="0.1"/></body>
      </worldbody>
      <actuator><position name="finger_motor" joint="finger" kp="100" ctrlrange="0 0.1"/>
        <position name="z_motor" joint="slide_z" kp="1000" kv="100" ctrlrange="-1 1"/>
        <position name="x_motor" joint="slide_x" kp="1000" kv="100" ctrlrange="-1 1"/></actuator>
    </mujoco>""")
    profile = RobotProfile(
        ("slide_x", "slide_z"),
        ("x_motor", "z_motor"),
        "tool",
        ToolPose((0, 0, 0), (1, 0, 0, 0)),
        (0.5, 0.5),
        (-2, -2, -2),
        (2, 2, 2),
    )
    bindings = ModelBindings(model, profile)
    io = MuJoCoAdapter(
        bindings, lambda data, obj: GripperFeedback(0.1, 0, (0, 0)), ("finger_motor",), "object"
    )
    return bindings, io, MinkIKSolver(bindings, ik_settings, {"slide_x": 0, "slide_z": 0})


@pytest.fixture
def panda(ik_settings):
    if not PANDA_XML.exists():
        pytest.skip("Run uv run python scripts/fetch_panda_model.py for Panda integration tests")
    model = mujoco.MjModel.from_xml_path(str(PANDA_XML))
    # Hand origin suffices for a kinematic test; no grasp-center claim is made.
    profile = RobotProfile(
        tuple(f"joint{i}" for i in range(1, 8)),
        tuple(f"actuator{i}" for i in range(1, 8)),
        "hand",
        ToolPose((0, 0, 0), (1, 0, 0, 0)),
        (0.5,) * 7,
        (-1, -1, -1),
        (1, 1, 1),
    )
    bindings = ModelBindings(model, profile)
    driver = PandaGripperDriver(model)
    io = MuJoCoAdapter(bindings, driver.observe, driver.actuator_names)
    state = io.reset("home")
    solver = MinkIKSolver(
        bindings, ik_settings, {n: state.joint_positions[n][0] for n in profile.arm_joints}
    )
    return bindings, io, solver, driver
