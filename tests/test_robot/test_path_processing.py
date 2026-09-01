"""Path selection/interpolation is explicit, deterministic, and non-destructive."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml
from pydantic import ValidationError
from scipy.interpolate import CubicSpline

from mimic.config import Config
from mimic.robot import (
    PathInterpolation,
    PathProcessingSettings,
    PathProcessor,
    process_path,
    retarget_task,
)
from mimic.robot.path_processing import _minimum_distances_to_polyline


def _retargeted(extracted_task, mapping_config_values, points=None):
    task = retarget_task(extracted_task, mapping_config_values)
    return replace(task, demonstrated_path_xy_m=tuple(points)) if points is not None else task


def test_direct_preserves_the_former_endpoint_only_behavior(extracted_task, mapping_config_values):
    task = _retargeted(extracted_task, mapping_config_values)
    result = process_path(task, {"interpolation": "direct"})
    assert result.interpolation == PathInterpolation.DIRECT
    assert result.xy_m == (task.start_xy_m, task.goal_xy_m)
    assert result.control_point_source_indices == (0, 6)
    assert result.source_task is task


def test_project_default_and_pipeline_yaml_use_cubic():
    root = Path(__file__).resolve().parents[2]
    default_settings = PathProcessingSettings.model_validate(Config().get("path_processing"))
    pipeline_settings = PathProcessingSettings.model_validate(
        yaml.safe_load((root / "configs" / "robot_pipeline.yaml").read_text())["robot_pipeline"][
            "path_processing"
        ]
    )
    assert default_settings == pipeline_settings
    assert default_settings.interpolation == PathInterpolation.CUBIC
    assert default_settings.corner_max_deviation_m == pytest.approx(0.04)
    assert default_settings.output_spacing_m == pytest.approx(0.05)
    assert default_settings.maximum_spline_deviation_m == pytest.approx(0.10)


def test_none_returns_every_coordinate_exactly_without_deduplication(
    extracted_task, mapping_config_values
):
    points = ((0.0, 0.0), (0.0, 0.0), (0.1, 0.0), (0.1, 0.1), (0.1, 0.1), (0.2, 0.1), (0.2, 0.2))
    task = _retargeted(extracted_task, mapping_config_values, points)
    result = process_path(task, {"interpolation": "none"})
    assert result.xy_m == points
    assert result.control_points_xy_m == points
    assert result.control_point_source_indices == tuple(range(len(points)))


def test_corners_only_retains_original_samples_and_respects_deviation(
    extracted_task, mapping_config_values
):
    points = ((0.0, 0.0), (0.1, 0.001), (0.2, 0.0), (0.3, 0.0), (0.3, 0.1), (0.3, 0.2), (0.4, 0.2))
    task = _retargeted(extracted_task, mapping_config_values, points)
    result = process_path(task, {"interpolation": "corners_only", "corner_max_deviation_m": 0.002})
    assert result.control_point_source_indices == (0, 3, 5, 6)
    assert result.xy_m == tuple(points[index] for index in result.control_point_source_indices)
    assert result.xy_m[0] == task.start_xy_m
    assert result.xy_m[-1] == task.goal_xy_m


def test_cubic_uses_scipy_spline_with_exact_endpoints_and_spatial_sampling(
    extracted_task, mapping_config_values
):
    points = (
        (0.0, 0.0),
        (0.1, 0.02),
        (0.2, 0.08),
        (0.3, 0.16),
        (0.4, 0.22),
        (0.5, 0.24),
        (0.6, 0.24),
    )
    task = _retargeted(extracted_task, mapping_config_values, points)
    result = process_path(
        task,
        {
            "interpolation": "cubic",
            "corner_max_deviation_m": 0.001,
            "output_spacing_m": 0.025,
            "maximum_spline_deviation_m": 0.02,
        },
    )
    assert result.interpolation == PathInterpolation.CUBIC
    assert result.xy_m[0] == points[0]
    assert result.xy_m[-1] == points[-1]
    segment_lengths = np.linalg.norm(np.diff(np.asarray(result.xy_m), axis=0), axis=1)
    assert np.max(segment_lengths) <= 0.025 + 1e-8
    assert len(result.xy_m) > len(points)


def test_cubic_samples_the_natural_spline_not_the_corner_polyline(
    extracted_task, mapping_config_values
):
    """A bent control polygon must leave the chords; linear corner interpolation would not."""
    points = (
        (0.0, 0.0),
        (0.1, 0.02),
        (0.2, 0.08),
        (0.3, 0.16),
        (0.4, 0.22),
        (0.5, 0.24),
        (0.6, 0.24),
    )
    task = _retargeted(extracted_task, mapping_config_values, points)
    cubic = process_path(
        task,
        {
            "interpolation": "cubic",
            "corner_max_deviation_m": 0.001,
            "output_spacing_m": 0.025,
            "maximum_spline_deviation_m": 0.02,
        },
    )
    direct = process_path(task, {"interpolation": "direct"})
    sampled = np.asarray(cubic.xy_m)
    controls = np.asarray(cubic.control_points_xy_m)
    assert len(controls) >= 3
    assert len(sampled) > len(direct.xy_m)

    chords = np.linalg.norm(np.diff(controls, axis=0), axis=1)
    parameters = np.concatenate(([0.0], np.cumsum(chords)))
    parameters /= parameters[-1]
    spline = CubicSpline(parameters, controls, axis=0, bc_type="natural", extrapolate=False)
    dense = spline(np.linspace(0.0, 1.0, 20001))
    spline_error = np.max(_minimum_distances_to_polyline(sampled, dense))
    polyline_error = np.max(_minimum_distances_to_polyline(sampled, controls))
    assert spline_error < 1e-8
    assert polyline_error > 1e-3


def test_cubic_keeps_a_straight_path_straight(extracted_task, mapping_config_values):
    points = tuple((index / 10, 0.0) for index in range(7))
    task = _retargeted(extracted_task, mapping_config_values, points)
    result = process_path(
        task,
        {
            "interpolation": "cubic",
            "corner_max_deviation_m": 0.001,
            "output_spacing_m": 0.04,
            "maximum_spline_deviation_m": 0.001,
        },
    )
    np.testing.assert_allclose(np.asarray(result.xy_m)[:, 1], 0.0, rtol=0, atol=1e-14)
    assert result.control_point_source_indices == (0, 6)


def test_cubic_rejects_excessive_departure_from_observed_polyline(
    extracted_task, mapping_config_values
):
    points = ((0.0, 0.0), (0.1, 0.2), (0.2, 0.4), (0.5, 1.0), (0.8, 0.4), (0.9, 0.2), (1.0, 0.0))
    task = _retargeted(extracted_task, mapping_config_values, points)
    with pytest.raises(ValueError, match="maximum_spline_deviation_m"):
        process_path(
            task,
            {
                "interpolation": "cubic",
                "corner_max_deviation_m": 2.0,
                "output_spacing_m": 0.05,
                "maximum_spline_deviation_m": 0.01,
            },
        )


def test_cubic_rejects_path_without_two_distinct_control_points(
    extracted_task, mapping_config_values
):
    task = _retargeted(extracted_task, mapping_config_values, ((0.0, 0.0),) * 7)
    with pytest.raises(ValueError, match="two distinct"):
        process_path(
            task,
            {
                "interpolation": "cubic",
                "corner_max_deviation_m": 0.01,
                "output_spacing_m": 0.01,
                "maximum_spline_deviation_m": 0.01,
            },
        )


@pytest.mark.parametrize("interpolation", ["DIRECT", "follow", "FOLLOW_PATH", "unknown"])
def test_unknown_or_legacy_options_are_rejected(interpolation):
    with pytest.raises(ValidationError):
        PathProcessingSettings(interpolation=interpolation)


@pytest.mark.parametrize(
    "values",
    [
        {"interpolation": "direct", "output_spacing_m": 0.01},
        {"interpolation": "none", "corner_max_deviation_m": 0.01},
        {"interpolation": "corners_only"},
        {"interpolation": "corners_only", "corner_max_deviation_m": 0.01, "output_spacing_m": 0.01},
        {"interpolation": "cubic", "corner_max_deviation_m": 0.01, "output_spacing_m": 0.01},
        {
            "interpolation": "cubic",
            "corner_max_deviation_m": True,
            "output_spacing_m": 0.01,
            "maximum_spline_deviation_m": 0.01,
        },
    ],
)
def test_mode_specific_configuration_is_strict(values):
    with pytest.raises(ValidationError):
        PathProcessingSettings.model_validate(values)


def test_processing_does_not_mutate_retained_geometry(extracted_task, mapping_config_values):
    task = _retargeted(extracted_task, mapping_config_values)
    before = task.path_xy_m
    processor = PathProcessor({"interpolation": "corners_only", "corner_max_deviation_m": 0.01})
    processor.process(task)
    assert task.path_xy_m == before
    assert task.source_task is extracted_task
