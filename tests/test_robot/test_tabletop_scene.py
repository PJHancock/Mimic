"""MuJoCo tabletop clone dimensions and left-edge frame placement."""

from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from mimic.common.constants import ROBOT_TABLE_SETBACK_M, TABLE_HEIGHT_M, TABLE_WIDTH_M
from mimic.robot import TabletopCloneSettings, add_tabletop_clone


def _settings():
    root = Path(__file__).resolve().parents[2]
    return yaml.safe_load((root / "configs" / "retargeting.yaml").read_text())["tabletop_clone"]


def test_clone_matches_physical_footprint_and_left_edge_origin():
    spec = mujoco.MjSpec()
    settings = add_tabletop_clone(spec, _settings())
    model = spec.compile()

    table = model.geom("tabletop_clone")
    base = model.site("robot_base_frame")
    np.testing.assert_allclose(table.size, (TABLE_WIDTH_M / 2, TABLE_HEIGHT_M / 2, 0.005))
    np.testing.assert_allclose(table.pos, (ROBOT_TABLE_SETBACK_M + TABLE_WIDTH_M / 2, 0.0, -0.005))
    np.testing.assert_allclose(base.pos, (0.0, 0.0, 0.0))
    assert settings.robot_edge == "left"


@pytest.mark.parametrize(
    "field,value",
    [
        ("width_m", 0.0),
        ("depth_m", -1.0),
        ("thickness_m", np.nan),
        ("surface_z_m", np.inf),
        ("robot_base_xy_m", [0.0]),
        ("robot_setback_m", -0.1),
        ("robot_edge", "right"),
    ],
)
def test_invalid_clone_configuration_fails_before_scene_build(field, value):
    settings = _settings()
    settings[field] = value
    with pytest.raises(ValidationError):
        TabletopCloneSettings.model_validate(settings)
