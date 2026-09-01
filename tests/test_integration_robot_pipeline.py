"""Saved pixel results reach the executor's world-waypoint JSON contract."""

import json
from pathlib import Path

import numpy as np
import pytest

from mimic.integration.robot_pipeline import (
    build_robot_pipeline,
    load_calibrated_object_tracks,
    main,
    waypoint_payload,
)


def _actions() -> dict:
    phases = ("IDLE", "HOVER", "GRASP", "CARRY", "RELEASE", "HOVER", "IDLE")
    return {
        "schema": "mimic.robot_actions.v1",
        "catalog": {
            "schema_version": 1,
            "fingerprint": "a" * 64,
            "labels": ["IDLE", "HOVER", "GRASP", "CARRY", "RELEASE"],
        },
        "checkpoint_sha256": "b" * 64,
        "postprocessing": {
            "fingerprint": "c" * 64,
            "settings": {
                "minimum_confidence": 0.5,
                "minimum_transition_margin": 0.1,
                "maximum_second_choice_gap": 0.2,
                "required_consecutive_observations": 1,
                "missing_detection_timeout_s": 0.5,
            },
            "guard_policy": "fail_closed_without_runtime_observations",
        },
        "frames": [
            {
                "frame_idx": index,
                "timestamp_s": (index - 1) * 0.1,
                "phase": phase,
                "confidence": 0.9,
                "decision_source": "model",
            }
            for index, phase in enumerate(phases, 1)
        ],
    }


def _results() -> dict:
    return {
        "schema": "mimic.demo_results.v2",
        "metadata": {
            "catalog_fingerprint": "a" * 64,
            "postprocessing_fingerprint": "c" * 64,
            "tracking_coordinate_frame": "image_pixels",
        },
        "per_frame": [
            {
                "frame_idx": index,
                "position": {"x": x, "y": y, "confidence": 0.8},
            }
            for index, (x, y) in enumerate(
                ((10, 10), (20, 20), (100, 200), (200, 300), (300, 400), (40, 40), (50, 50)),
                1,
            )
        ],
    }


def _calibration() -> dict:
    return {
        "homography": [[0.001, 0, 0], [0, 0.001, 0], [0, 0, 1]],
        "table_width_m": 0.508,
        "table_height_m": 0.762,
    }


def _retargeting() -> dict:
    return {
        "retargeting": {
            "source_frame": "table",
            "target_frame": "mujoco_world",
            "table_origin_target_xy_m": [0.0, 0.381],
            "table_x_axis_target_xy": [1.0, 0.0],
            "table_y_axis_target_xy": [0.0, -1.0],
        },
        "tabletop_clone": {
            "width_m": 0.508,
            "depth_m": 0.762,
        },
    }


def _pipeline_config() -> dict:
    return {
        "robot_pipeline": {
            "path_processing": {"interpolation": "direct"},
            "waypoint_construction": {
                "approach_z_m": 0.2,
                "grasp_z_m": 0.03,
                "lift_z_m": 0.2,
                "transport_z_m": 0.2,
                "lower_z_m": 0.03,
                "retreat_z_m": 0.2,
                "object_goal_z_m": 0.02,
                "tool_quaternion_wxyz": [0.0, 1.0, 0.0, 0.0],
            },
        }
    }


def test_saved_pixels_are_calibrated_and_retargeted_into_world_waypoints() -> None:
    artifacts = build_robot_pipeline(
        actions_source=_actions(),
        results_source=_results(),
        calibration_source=_calibration(),
        retargeting_source=_retargeting(),
        pipeline_config_source=_pipeline_config(),
    )

    assert artifacts.episode_count == 1
    assert artifacts.selected_episode == 1
    assert artifacts.task.start_xy_m == pytest.approx((0.1, 0.2))
    assert artifacts.task.goal_xy_m == pytest.approx((0.3, 0.4))
    assert artifacts.retargeted_task.start_xy_m == pytest.approx((0.1, 0.181))
    assert artifacts.retargeted_task.goal_xy_m == pytest.approx((0.3, -0.019))
    assert artifacts.waypoints.grasp.position == pytest.approx((0.1, 0.181, 0.03))
    assert artifacts.waypoints.goal_position == pytest.approx((0.3, -0.019, 0.02))
    np.testing.assert_allclose(
        [pose.position[:2] for pose in artifacts.waypoints.path],
        [(0.1, 0.181), (0.3, -0.019)],
        rtol=0,
        atol=1e-7,
    )


def test_loader_skips_missing_detections_without_fabricating_coordinates() -> None:
    results = _results()
    results["per_frame"][0]["position"] = {"x": None, "y": None, "confidence": 0.0}
    tracks = load_calibrated_object_tracks(results, _calibration())
    assert [track.frame_idx for track in tracks] == [2, 3, 4, 5, 6, 7]


def test_loader_prefers_complete_tracker_stream_over_sparse_action_frames() -> None:
    results = _results()
    results["per_frame"] = results["per_frame"][::2]
    results["object_tracks"] = _results()["per_frame"]
    tracks = load_calibrated_object_tracks(results, _calibration())
    assert [track.frame_idx for track in tracks] == list(range(1, 8))


def test_calibration_must_match_the_tabletop_clone() -> None:
    retargeting = _retargeting()
    retargeting["tabletop_clone"]["width_m"] = 1.0
    with pytest.raises(ValueError, match="must match"):
        build_robot_pipeline(
            actions_source=_actions(),
            results_source=_results(),
            calibration_source=_calibration(),
            retargeting_source=retargeting,
            pipeline_config_source=_pipeline_config(),
        )


def test_pipeline_rejects_action_provenance_mismatch() -> None:
    results = _results()
    results["metadata"]["postprocessing_fingerprint"] = "d" * 64
    with pytest.raises(ValueError, match="post-processing"):
        build_robot_pipeline(
            actions_source=_actions(),
            results_source=results,
            calibration_source=_calibration(),
            retargeting_source=_retargeting(),
            pipeline_config_source=_pipeline_config(),
        )


def test_cli_writes_simulator_contract_without_importing_the_executor(tmp_path: Path) -> None:
    inputs = {
        "actions.json": _actions(),
        "results.json": _results(),
        "calibration.json": _calibration(),
    }
    for name, payload in inputs.items():
        (tmp_path / name).write_text(json.dumps(payload))
    import yaml

    (tmp_path / "retargeting.yaml").write_text(yaml.safe_dump(_retargeting()))
    (tmp_path / "pipeline.yaml").write_text(yaml.safe_dump(_pipeline_config()))
    output = tmp_path / "world_waypoints.json"

    result = main(
        (
            "--actions",
            str(tmp_path / "actions.json"),
            "--results",
            str(tmp_path / "results.json"),
            "--calibration",
            str(tmp_path / "calibration.json"),
            "--retargeting-config",
            str(tmp_path / "retargeting.yaml"),
            "--pipeline-config",
            str(tmp_path / "pipeline.yaml"),
            "--waypoints",
            str(output),
        )
    )

    assert result == 0
    payload = json.loads(output.read_text())
    assert payload == json.loads(
        json.dumps(
            waypoint_payload(
                build_robot_pipeline(
                    actions_source=_actions(),
                    results_source=_results(),
                    calibration_source=_calibration(),
                    retargeting_source=_retargeting(),
                    pipeline_config_source=_pipeline_config(),
                ).waypoints
            )
        )
    )
    assert set(payload) == {
        "approach",
        "grasp",
        "lift",
        "path",
        "lower",
        "retreat",
        "goal_position",
    }
