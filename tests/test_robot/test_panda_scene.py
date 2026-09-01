"""Configured Panda scene preserves the approved table frame and reset object pose."""

from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
import pytest
import yaml

from mimic.common.constants import ROBOT_TABLE_SETBACK_M, TABLE_HEIGHT_M, TABLE_WIDTH_M

ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / "models" / "panda_pick_place_scene.xml"
UPSTREAM = ROOT / "models" / "franka_emika_panda" / "upstream" / "panda.xml"


def test_robot_config_references_scene_specific_complete_keyframe() -> None:
    config = yaml.safe_load((ROOT / "configs" / "robots" / "panda_complete.yaml").read_text())[
        "robot_execution"
    ]
    scene_root = ET.parse(SCENE).getroot()
    key_names = {key.attrib["name"] for key in scene_root.findall("./keyframe/key")}

    assert (ROOT / "configs" / "robots" / config["model_path"]).resolve() == SCENE
    assert config["home_keyframe"] == "pick_place_home"
    assert config["presets"]["home"]["keyframe"] == "pick_place_home"
    assert config["ik"]["task_gain"] == pytest.approx(1.0)
    assert config["support_geom"] == "tabletop_clone"
    assert config["execution"]["waypoint_handoff_radius_m"] == pytest.approx(0.03)
    assert config["execution"]["placement_approach_clearance_m"] == pytest.approx(0.015)
    assert config["execution"]["placement_maximum_descent_speed_m_s"] == pytest.approx(0.05)
    assert config["gripper"] == {
        "open_command_width_m": pytest.approx(0.0799),
        "width_tolerance_m": pytest.approx(0.001),
        "empty_width_m": pytest.approx(0.002),
        "contact_force_n": pytest.approx(0.1),
        "contact_duration_s": pytest.approx(0.1),
        "movement_timeout_s": pytest.approx(2.0),
    }
    assert "pick_place_home" in key_names


@pytest.mark.skipif(not UPSTREAM.is_file(), reason="run scripts/fetch_panda_model.py first")
def test_scene_loads_with_table_clone_and_cube_reset_pose() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    key_id = model.key("pick_place_home").id
    mujoco.mj_resetDataKeyframe(model, data, key_id)

    table = model.geom("tabletop_clone")
    base = model.site("robot_base_frame")
    cube_joint = model.joint("cube_free")
    cube_qpos_address = model.jnt_qposadr[cube_joint.id]
    cube_qpos_end = cube_qpos_address + 7
    cube_qpos = data.qpos[cube_qpos_address:cube_qpos_end]

    np.testing.assert_allclose(table.size, (TABLE_WIDTH_M / 2, TABLE_HEIGHT_M / 2, 0.005))
    np.testing.assert_allclose(table.pos, (ROBOT_TABLE_SETBACK_M + TABLE_WIDTH_M / 2, 0.0, -0.005))
    np.testing.assert_allclose(base.pos, (0.0, 0.0, 0.0))
    np.testing.assert_allclose(cube_qpos, (0.30, 0.0, 0.02, 1.0, 0.0, 0.0, 0.0))
    assert model.body("cube").id >= 0
