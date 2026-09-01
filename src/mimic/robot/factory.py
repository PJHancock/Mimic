"""Explicit application wiring; all experiment-specific settings are required."""

import hashlib
from pathlib import Path
from typing import Callable, Optional

import mujoco

from mimic.common.types import PickPlaceWaypoints, ToolPose
from mimic.config import Config
from mimic.robot.backends.mink_ik import MinkIKSolver
from mimic.robot.backends.panda_gripper import PandaGripperDriver
from mimic.robot.backends.ruckig_position import PositionTrajectorySettings, RuckigPositionIK
from mimic.robot.controller import RobotController
from mimic.robot.gripper import GripperLogic, GripperSettings
from mimic.robot.inverse_kinematics import IKSettings
from mimic.robot.model import ModelBindings, RobotProfile
from mimic.robot.presets import resolve_joint_preset
from mimic.robot.simulation import MuJoCoAdapter
from mimic.robot.state_machine import ExecutionSettings, SkillExecutor


def read_pose(value: dict) -> ToolPose:
    if not isinstance(value, dict):
        raise ValueError("An explicit pose with position and quaternion_wxyz is required")
    return ToolPose(tuple(value["position"]), tuple(value["quaternion_wxyz"]))


def read_waypoints(value: dict) -> PickPlaceWaypoints:
    return PickPlaceWaypoints(
        **{
            name: read_pose(value[name])
            for name in ("approach", "grasp", "lift", "lower", "retreat")
        },
        path=tuple(read_pose(pose) for pose in value["path"]),
        goal_position=tuple(value["goal_position"]),
    )


def build_executor(
    config_path: Path, record: Optional[Callable[[dict], None]] = None
) -> SkillExecutor:
    cfg = Config(str(config_path)).get("robot_execution")
    if not isinstance(cfg, dict):
        raise ValueError("robot_execution configuration is required")

    # Nulls document unresolved decisions; never replace them with numerical defaults.
    def require_complete(value, path="robot_execution"):
        if value is None:
            raise ValueError(f"Unresolved setting: {path}; see docs/ROBOT_EXECUTION.md")
        if isinstance(value, dict):
            for key, entry in value.items():
                require_complete(entry, f"{path}.{key}")
        elif isinstance(value, list):
            for index, entry in enumerate(value):
                require_complete(entry, f"{path}[{index}]")

    require_complete(cfg)
    path = Path(cfg["model_path"])
    if not path.is_absolute():
        path = config_path.resolve().parent / path
    if record is not None:
        record(
            {
                "event": "configuration",
                "effective_robot_execution": cfg,
                "model_path": str(path.resolve()),
                "scene_xml_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    model = mujoco.MjModel.from_xml_path(str(path))
    spec = cfg["profile"]
    profile = RobotProfile(
        arm_joints=tuple(spec["arm_joints"]),
        arm_actuators=tuple(spec["arm_actuators"]),
        tool_body=spec["tool_body"],
        tool_offset=read_pose(spec["tool_offset"]),
        velocity_limits=tuple(spec["velocity_limits"]),
        workspace_min=tuple(spec["workspace_min"]),
        workspace_max=tuple(spec["workspace_max"]),
    )
    bindings = ModelBindings(model, profile)
    if cfg["gripper_driver"] != "panda_standard_tendon":
        raise ValueError("Unsupported gripper driver; inject a compatible driver explicitly")
    gripper_config = dict(cfg["gripper"])
    driver = PandaGripperDriver(
        model, open_command_width_m=gripper_config.pop("open_command_width_m")
    )
    io = MuJoCoAdapter(bindings, driver.observe, driver.actuator_names, cfg["object_body"])
    support_contact = io.support_contact_observer(cfg["support_geom"])
    state = io.reset(cfg["home_keyframe"])
    reference = {name: state.joint_positions[name][0] for name in profile.arm_joints}
    differential_ik = MinkIKSolver(
        bindings,
        IKSettings(**cfg["ik"]),
        reference,
        measured_gripper_tolerances_m={
            model.joint(joint).name: cfg["measured_gripper_joint_tolerance_m"]
            for joint in driver.joint_ids
        },
    )
    solver = RuckigPositionIK(
        differential_ik,
        bindings,
        PositionTrajectorySettings(**cfg["trajectory"]),
    )
    controller = RobotController(
        io,
        solver,
        GripperLogic(driver, GripperSettings(**gripper_config)),
        cfg["arm_control_hz"],
        cfg["gripper_control_hz"],
    )
    # Reject a timing mismatch before the first command can be written.
    interval = controller.dt_s / model.opt.timestep
    if abs(interval - round(interval)) > 1e-8 or interval < 1:
        raise ValueError("Configured arm interval is not an integer number of simulation steps")
    home_spec = cfg["presets"]["home"]
    source = home_spec.get("source")
    if source == "keyframe":
        home = resolve_joint_preset(bindings, "home", keyframe=home_spec.get("keyframe"))
    elif source == "joint_positions":
        home = resolve_joint_preset(
            bindings, "home", joint_positions=home_spec.get("joint_positions")
        )
    else:
        raise ValueError("Home preset source must be keyframe or joint_positions")
    tolerances = cfg["preset_position_tolerances"]
    if len(tolerances) != len(profile.arm_joints):
        raise ValueError("Preset position tolerances must match the arm joint count")
    execution_settings = ExecutionSettings(**cfg["execution"])
    if execution_settings.waypoint_handoff_radius_m <= cfg["ik"]["position_tolerance_m"]:
        raise ValueError("execution.waypoint_handoff_radius_m must exceed ik.position_tolerance_m")
    return SkillExecutor(
        controller,
        execution_settings,
        record,
        home,
        dict(zip(profile.arm_joints, tolerances)),
        io.initialize_object_position,
        support_contact,
    )
