"""Processed geometry reaches the existing executor contract without guessed poses."""

import numpy as np
import pytest
from pydantic import ValidationError

from mimic.robot import build_waypoints, process_path, retarget_task
from mimic.robot.state_machine import expand_skills


def _settings():
    return {
        "approach_z_m": 0.2,
        "grasp_z_m": 0.03,
        "lift_z_m": 0.2,
        "transport_z_m": 0.2,
        "lower_z_m": 0.03,
        "retreat_z_m": 0.2,
        "object_goal_z_m": 0.02,
        "tool_quaternion_wxyz": (0.0, 1.0, 0.0, 0.0),
    }


@pytest.mark.parametrize(
    "path_settings",
    [
        {"interpolation": "direct"},
        {"interpolation": "none"},
        {"interpolation": "corners_only", "corner_max_deviation_m": 0.01},
        {
            "interpolation": "cubic",
            "corner_max_deviation_m": 0.01,
            "output_spacing_m": 0.02,
            "maximum_spline_deviation_m": 0.05,
        },
    ],
)
def test_all_path_modes_feed_the_existing_skill_contract(
    extracted_task, mapping_config_values, path_settings
):
    task = retarget_task(extracted_task, mapping_config_values)
    path = process_path(task, path_settings)
    waypoints = build_waypoints(path, _settings())
    path_skills = [step for step in expand_skills(waypoints, 0.015) if step.skill == "FOLLOW_PATH"]
    assert len(path_skills) == len(path.xy_m)
    assert tuple(step.pose.position[:2] for step in path_skills) == path.xy_m
    assert tuple(step.waypoint_index for step in path_skills) == tuple(range(len(path.xy_m)))
    assert all(step.waypoint_count == len(path.xy_m) for step in path_skills)
    assert all(step.pose.position[2] == 0.2 for step in path_skills)
    assert waypoints.approach.position[:2] == task.start_xy_m
    assert waypoints.lower.position[:2] == task.goal_xy_m
    assert waypoints.goal_position == (*task.goal_xy_m, 0.02)


def test_builder_preserves_one_fixed_orientation(extracted_task, mapping_config_values):
    path = process_path(
        retarget_task(extracted_task, mapping_config_values), {"interpolation": "direct"}
    )
    waypoints = build_waypoints(path, _settings())
    all_poses = (
        waypoints.approach,
        waypoints.grasp,
        waypoints.lift,
        *waypoints.path,
        waypoints.lower,
        waypoints.retreat,
    )
    assert all(pose.quaternion_wxyz == (0.0, 1.0, 0.0, 0.0) for pose in all_poses)


@pytest.mark.parametrize(
    "field,value",
    [
        ("approach_z_m", np.nan),
        ("grasp_z_m", True),
        ("transport_z_m", "0.2"),
        ("tool_quaternion_wxyz", (1.0, 1.0, 0.0, 0.0)),
        ("tool_quaternion_wxyz", (1.0, 0.0, 0.0)),
    ],
)
def test_invalid_or_implicit_pose_settings_are_rejected(
    extracted_task, mapping_config_values, field, value
):
    settings = _settings()
    settings[field] = value
    path = process_path(
        retarget_task(extracted_task, mapping_config_values), {"interpolation": "direct"}
    )
    with pytest.raises(ValidationError):
        build_waypoints(path, settings)
