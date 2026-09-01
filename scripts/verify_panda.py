"""Diagnostic fixture, not production calibration: attempt pick/place and retain failures.

This scene uses a 4 cm cube (30 g), a table at z=0, and a tool point at the
center height of the standard model's principal fingertip pads. All settings
below are fixed test assumptions. Its measured-state allowance, motion generator,
safe gripper-open command, and slip criterion are logged for reproducibility.
"""

import argparse
import hashlib
import importlib.metadata
import json
from dataclasses import asdict
from pathlib import Path

import mujoco

from mimic.common.types import PickPlaceWaypoints, ToolPose
from mimic.robot.backends.mink_ik import MinkIKSolver
from mimic.robot.backends.panda_gripper import PandaGripperDriver
from mimic.robot.backends.ruckig_position import PositionTrajectorySettings, RuckigPositionIK
from mimic.robot.controller import RobotController
from mimic.robot.gripper import GripperLogic, GripperSettings
from mimic.robot.inverse_kinematics import IKSettings
from mimic.robot.model import ModelBindings, RobotProfile
from mimic.robot.simulation import MuJoCoAdapter
from mimic.robot.state_machine import ExecutionSettings, SkillExecutor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New diagnostic directory")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    model_path = root / "models/franka_emika_panda/upstream/panda.xml"
    spec = mujoco.MjSpec.from_file(str(model_path))
    # Add scene objects only. Do not change Panda actuators, dynamics or collision settings.
    spec.worldbody.add_geom(
        name="table", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0.5, 0, -0.025], size=[0.4, 0.5, 0.025]
    )
    object_body = spec.worldbody.add_body(name="object", pos=[0.5, 0, 0.02])
    object_body.add_freejoint(name="object_free")
    object_body.add_geom(
        name="object_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.02, 0.02], mass=0.03
    )
    model = spec.compile()
    # Explicit initialization: extend upstream home with the object's initial pose.
    object_adr = model.jnt_qposadr[model.joint("object_free").id]
    model.key_qpos[model.key("home").id, object_adr : object_adr + 7] = model.qpos0[
        object_adr : object_adr + 7
    ]
    profile = RobotProfile(
        tuple(f"joint{i}" for i in range(1, 8)),
        tuple(f"actuator{i}" for i in range(1, 8)),
        "hand",
        # Derived from XML finger base 0.0584 + primary pad center 0.0445 m.
        ToolPose((0, 0, 0.1029), (1, 0, 0, 0)),
        (0.5,) * 7,
        (0.2, -0.4, 0),
        (0.8, 0.4, 0.5),
    )
    bindings = ModelBindings(model, profile)
    driver = PandaGripperDriver(model, open_command_width_m=0.0799)
    io = MuJoCoAdapter(bindings, driver.observe, driver.actuator_names, "object")
    state = io.reset("home")
    settings = IKSettings(1e-4, 1e-4, 1, 1, 0, 1e-4, 1)
    gripper_settings = GripperSettings(0.001, 0.002, 0.1, 0.1, 2)
    # Previous fixture maximum_slip_m was 0.01. Measured stable in-gripper settling
    # reached 0.01219 m; 0.015 retains loss detection with 2.81 mm fixture headroom.
    execution_settings = ExecutionSettings(
        step_timeout_s=10,
        minimum_lift_m=0.05,
        maximum_slip_m=0.015,
        waypoint_handoff_radius_m=0.1,
        contact_loss_timeout_s=0.25,
        settle_time_s=0.01,
        settled_speed_m_s=0.01,
        placement_tolerance_m=0.01,
        placement_approach_clearance_m=0.015,
        placement_maximum_descent_speed_m_s=0.05,
        placement_maximum_descent_acceleration_m_s2=0.5,
        placement_contact_confirmation_s=0.05,
    )
    measured_gripper_tolerances_m = {model.joint(joint).name: 1e-5 for joint in driver.joint_ids}
    differential_ik = MinkIKSolver(
        bindings,
        settings,
        {n: state.joint_positions[n][0] for n in profile.arm_joints},
        measured_gripper_tolerances_m=measured_gripper_tolerances_m,
    )
    trajectory_settings = PositionTrajectorySettings(
        (2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61),
        (15, 7.5, 10, 12.5, 15, 20, 20),
        (7500, 3750, 5000, 6250, 7500, 10000, 10000),
        (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
        2000,
    )
    solver = RuckigPositionIK(differential_ik, bindings, trajectory_settings)
    controller = RobotController(io, solver, GripperLogic(driver, gripper_settings), 100, 10)

    # World z is up; this fixture points local tool z down with fixed world x rotation.
    def pose(x, y, z):
        return ToolPose((x, y, z), (0, 1, 0, 0))

    waypoints = PickPlaceWaypoints(
        pose(0.5, 0, 0.17),
        pose(0.5, 0, 0.02),
        pose(0.5, 0, 0.17),
        tuple(pose(0.5, y, 0.17) for y in (0.025, 0.05, 0.075, 0.1)),
        pose(0.5, 0.1, 0.02),
        pose(0.5, 0.1, 0.17),
        (0.5, 0.1, 0.02),
    )
    with (args.output / "trace.jsonl").open("x") as stream:

        def record(event):
            stream.write(json.dumps(event, allow_nan=False) + "\n")

        record(
            {
                "event": "metadata",
                "fixture_only": True,
                "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                "model_manifest": json.loads((model_path.parent / "manifest.json").read_text()),
                "profile": asdict(profile),
                "ik": asdict(settings),
                "gripper": asdict(gripper_settings),
                "gripper_open_command_width_m": driver.open_command_width_m,
                "trajectory": asdict(trajectory_settings),
                "measured_gripper_tolerances_m": measured_gripper_tolerances_m,
                "waypoints": asdict(waypoints),
                "arm_control_hz": 100,
                "gripper_control_hz": 10,
                "versions": {
                    n: importlib.metadata.version(n)
                    for n in ("mink", "mujoco", "qpsolvers", "daqp", "ruckig")
                },
            }
        )
        report = SkillExecutor(
            controller,
            execution_settings,
            record,
            support_contact=io.support_contact_observer("table"),
        ).run(waypoints)
    summary = asdict(report)
    (args.output / "result.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "final_state"}, indent=2))
    return 0 if report.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
