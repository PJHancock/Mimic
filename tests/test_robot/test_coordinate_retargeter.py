"""Known mappings, configuration failures, and source-path preservation."""

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from mimic.common.constants import TABLE_HEIGHT_M, TABLE_WIDTH_M
from mimic.robot import CoordinateRetargeter, MappingConfig, retarget_task


def test_known_origin_rotation_in_meters(extracted_task, mapping_config_values):
    result = retarget_task(extracted_task, mapping_config_values)
    assert result.target_frame == "synthetic_world"
    assert result.source_task is extracted_task
    np.testing.assert_allclose(result.start_xy_m, (0.8, -1.9), rtol=0, atol=1e-14)
    np.testing.assert_allclose(result.goal_xy_m, (0.7, -1.4), rtol=0, atol=1e-14)
    assert len(result.path_xy_m) == 7
    np.testing.assert_allclose(
        result.carry_trajectory_xy_m, ((0.5, -1.7), (0.4, -1.6), (0.6, -1.5))
    )


def test_explicit_downward_y_mapping(extracted_task, mapping_config_values):
    mapping_config_values.update(
        table_origin_target_xy_m=[0, 0],
        table_x_axis_target_xy=[1, 0],
        table_y_axis_target_xy=[0, -1],
    )
    result = retarget_task(extracted_task, mapping_config_values)
    np.testing.assert_allclose(result.start_xy_m, (0.1, -0.2))
    np.testing.assert_allclose(result.goal_xy_m, (0.6, -0.3))


def test_configured_arbitrary_rotation_round_trip(extracted_task, mapping_config_values):
    angle = 0.37
    mapping_config_values.update(
        table_x_axis_target_xy=[np.cos(angle), np.sin(angle)],
        table_y_axis_target_xy=[-np.sin(angle), np.cos(angle)],
    )
    config = MappingConfig.model_validate(mapping_config_values)
    result = CoordinateRetargeter(config).retarget(extracted_task)
    axes = np.column_stack((config.table_x_axis_target_xy, config.table_y_axis_target_xy))
    recovered = (np.asarray(result.path_xy_m) - config.table_origin_target_xy_m) @ axes
    np.testing.assert_allclose(recovered, extracted_task.path_xy_m, rtol=0, atol=1e-12)


def test_mapping_keeps_path_frames_phases_and_source_unchanged(
    extracted_task, mapping_config_values
):
    original = extracted_task.path_xy_m
    result = retarget_task(extracted_task, mapping_config_values)
    full = result.path_xy_m
    for _ in range(3):
        assert result.path_xy_m == full
        assert extracted_task.path_xy_m == original
    assert result.source_task.phase_boundaries == extracted_task.phase_boundaries


@pytest.mark.parametrize(
    "field",
    [
        "source_frame",
        "target_frame",
        "table_origin_target_xy_m",
        "table_x_axis_target_xy",
        "table_y_axis_target_xy",
    ],
)
def test_every_configuration_value_is_required(mapping_config_values, field):
    del mapping_config_values[field]
    with pytest.raises(ValidationError):
        CoordinateRetargeter(mapping_config_values)


@pytest.mark.parametrize("values", [None, {}])
def test_no_default_mapping(values):
    with pytest.raises(ValueError):
        CoordinateRetargeter(values)


def test_deployment_template_maps_left_edge_table_clone():
    root = Path(__file__).resolve().parents[2]
    data = yaml.safe_load((root / "configs" / "retargeting.yaml").read_text())
    defaults = yaml.safe_load((root / "configs" / "default.yaml").read_text())
    config = CoordinateRetargeter(data["retargeting"]).mapping_config
    assert defaults["tracking"]["table_width_m"] == TABLE_WIDTH_M
    assert defaults["tracking"]["table_height_m"] == TABLE_HEIGHT_M
    assert config.target_frame == "mujoco_world"
    assert config.table_origin_target_xy_m == (0.0, TABLE_HEIGHT_M / 2)
    assert config.table_x_axis_target_xy == (1.0, 0.0)
    assert config.table_y_axis_target_xy == (0.0, -1.0)
    assert data["tabletop_clone"] == {
        "width_m": TABLE_WIDTH_M,
        "depth_m": TABLE_HEIGHT_M,
        "thickness_m": 0.01,
        "surface_z_m": 0.0,
        "robot_edge": "left",
        "robot_base_xy_m": [0.0, 0.0],
    }


def test_left_edge_mapping_places_robot_at_origin_and_clones_table(extracted_task):
    root = Path(__file__).resolve().parents[2]
    data = yaml.safe_load((root / "configs" / "retargeting.yaml").read_text())
    table_points = (
        (0.0, 0.0),
        (TABLE_WIDTH_M, 0.0),
        (0.0, TABLE_HEIGHT_M),
        (TABLE_WIDTH_M, TABLE_HEIGHT_M),
        (0.0, TABLE_HEIGHT_M / 2),
        (TABLE_WIDTH_M, TABLE_HEIGHT_M / 2),
        (0.0, TABLE_HEIGHT_M / 2),
    )
    samples = tuple(
        replace(sample, table_xy_m=point)
        for sample, point in zip(extracted_task.demonstrated_path, table_points)
    )
    task = replace(extracted_task, demonstrated_path=samples)

    result = retarget_task(task, data["retargeting"])

    np.testing.assert_allclose(
        result.path_xy_m,
        (
            (0.0, TABLE_HEIGHT_M / 2),
            (TABLE_WIDTH_M, TABLE_HEIGHT_M / 2),
            (0.0, -TABLE_HEIGHT_M / 2),
            (TABLE_WIDTH_M, -TABLE_HEIGHT_M / 2),
            (0.0, 0.0),
            (TABLE_WIDTH_M, 0.0),
            (0.0, 0.0),
        ),
        rtol=0,
        atol=1e-14,
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("target_frame", ""),
        ("target_frame", "table"),
        ("target_frame", " world "),
        ("source_frame", "pixels"),
        ("table_origin_target_xy_m", [0, np.inf]),
        ("table_origin_target_xy_m", [0, np.nan]),
        ("table_origin_target_xy_m", [0, 0, 0]),
        ("table_origin_target_xy_m", ["0", 0]),
        ("table_origin_target_xy_m", [True, 0]),
        ("table_x_axis_target_xy", [2, 0]),
        ("table_x_axis_target_xy", [0, 0]),
        ("table_y_axis_target_xy", [0, 1]),  # parallel to configured X
        ("table_y_axis_target_xy", [1, 1]),
        ("table_y_axis_target_xy", [0, np.nan]),
    ],
)
def test_invalid_config_rejected_not_corrected(mapping_config_values, field, value):
    mapping_config_values[field] = value
    with pytest.raises(ValidationError):
        CoordinateRetargeter(mapping_config_values)


def test_unknown_fields_rejected(mapping_config_values):
    mapping_config_values["scale"] = 2
    with pytest.raises(ValidationError):
        CoordinateRetargeter(mapping_config_values)


def test_config_is_detached_and_frozen(extracted_task, mapping_config_values):
    mapper = CoordinateRetargeter(mapping_config_values)
    mapping_config_values["table_origin_target_xy_m"][0] = 999
    np.testing.assert_allclose(mapper.retarget(extracted_task).start_xy_m, (0.8, -1.9))
    with pytest.raises(ValidationError):
        mapper.mapping_config.target_frame = "changed"


def test_config_instances_are_revalidated(mapping_config_values):
    config = MappingConfig.model_validate(mapping_config_values)
    # Pydantic's model_copy(update=...) intentionally bypasses validation.
    invalid = config.model_copy(update={"table_origin_target_xy_m": (np.nan, 0)})
    with pytest.raises(ValidationError):
        CoordinateRetargeter(invalid)


def test_mapping_overflow_rejected(extracted_task, mapping_config_values):
    samples = tuple(
        replace(s, table_xy_m=(1e308, 1e308)) for s in extracted_task.demonstrated_path
    )
    task = replace(extracted_task, demonstrated_path=samples)
    mapping_config_values.update(
        table_origin_target_xy_m=[np.finfo(float).max, np.finfo(float).max],
        table_x_axis_target_xy=[1, 0],
        table_y_axis_target_xy=[0, 1],
    )
    with pytest.raises(ValueError, match="nonfinite"):
        retarget_task(task, mapping_config_values)


def test_retargeted_task_validates_sample_count_and_values(extracted_task, mapping_config_values):
    result = retarget_task(extracted_task, mapping_config_values)
    with pytest.raises(ValueError, match="every source"):
        replace(result, demonstrated_path_xy_m=result.demonstrated_path_xy_m[:-1])
    with pytest.raises(ValueError, match="finite"):
        replace(result, demonstrated_path_xy_m=((np.nan, 0),) * 7)


def test_geometry_imports_do_not_load_robot_or_model_backends():
    code = """
import sys
from mimic.common import ActionPrediction, ExtractedTask
from mimic.robot import (
    CoordinateRetargeter,
    MappingConfig,
    PathProcessor,
    TaskExtractor,
    WaypointBuilder,
)
assert not any(name.split('.')[0] in {'mujoco', 'mink', 'torch'} for name in sys.modules)
import mimic.robot
assert 'TaskExtractor' in dir(mimic.robot)
try:
    mimic.robot.nonexistent_export
except AttributeError:
    pass
else:
    raise AssertionError('Unknown exports must raise AttributeError')
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
