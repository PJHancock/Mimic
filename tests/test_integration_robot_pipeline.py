"""Saved pixel results reach the executor's world-waypoint JSON contract."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from mimic.integration.robot_pipeline import (
    build_robot_pipeline,
    build_robot_pipelines,
    load_calibrated_object_tracks,
    main,
    waypoint_payload,
    waypoint_sequence_payload,
)
from mimic.integration.task_input import load_demo_task_input
from mimic.robot import PathProcessingSettings, WaypointConstructionSettings


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


def _task_input() -> dict:
    actions = _actions()
    return {
        "schema": "mimic.demo_task_input.v1",
        "video": {
            "created_at": "2026-09-01T00:00:00-06:00",
            "fps": 10.0,
            "frame_count": 7,
            "duration_s": 0.7,
            "tracking_coordinate_frame": "image_pixels",
            "image_width_px": 1000,
            "image_height_px": 1000,
        },
        "catalog": actions["catalog"],
        "checkpoint_sha256": actions["checkpoint_sha256"],
        "postprocessing": actions["postprocessing"],
        "resolved_actions": actions["frames"],
        "object_tracks": [
            {
                "frame_idx": index,
                "timestamp_s": (index - 1) * 0.1,
                "position": {"x": x, "y": y, "confidence": 0.8},
            }
            for index, (x, y) in enumerate(
                ((10, 10), (20, 20), (100, 200), (200, 300), (300, 400), (40, 40), (50, 50)),
                1,
            )
        ],
    }


def _continuation_task_input() -> dict:
    task_input = _task_input()
    phases = (
        "IDLE",
        "HOVER",
        "GRASP",
        "CARRY",
        "RELEASE",
        "IDLE",
        "GRASP",
        "CARRY",
        "RELEASE",
        "IDLE",
    )
    task_input["video"].update(frame_count=len(phases), duration_s=1.0)
    task_input["resolved_actions"] = [
        {
            "frame_idx": index,
            "timestamp_s": (index - 1) * 0.1,
            "phase": phase,
            "confidence": 0.9,
            "decision_source": "model",
        }
        for index, phase in enumerate(phases, 1)
    ]
    positions = (
        (10, 10),
        (20, 20),
        (100, 200),
        (200, 300),
        (300, 400),
        (300, 400),
        (300, 400),
        (400, 300),
        (500, 200),
        (500, 200),
    )
    task_input["object_tracks"] = [
        {
            "frame_idx": index,
            "timestamp_s": (index - 1) * 0.1,
            "position": {"x": x, "y": y, "confidence": 0.8},
        }
        for index, (x, y) in enumerate(positions, 1)
    ]
    return task_input


def _calibration() -> dict:
    return {
        "homography": [[0.001, 0, 0], [0, 0.001, 0], [0, 0, 1]],
        "table_width_m": 0.508,
        "table_height_m": 0.762,
        "image_width_px": 1000,
        "image_height_px": 1000,
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
        task_input_source=load_demo_task_input(_task_input()),
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


def test_pipeline_builds_every_continuation_episode_in_source_order() -> None:
    artifacts = build_robot_pipelines(
        task_input_source=_continuation_task_input(),
        calibration_source=_calibration(),
        retargeting_source=_retargeting(),
        pipeline_config_source=_pipeline_config(),
    )

    assert [artifact.selected_episode for artifact in artifacts] == [1, 2]
    assert [artifact.task.grasp_frame for artifact in artifacts] == [3, 7]
    assert artifacts[0].waypoints.goal_position[:2] == pytest.approx(
        artifacts[1].waypoints.grasp.position[:2]
    )
    payload = waypoint_sequence_payload(tuple(artifact.waypoints for artifact in artifacts))
    assert payload["schema"] == "mimic.world_waypoint_sequence.v1"
    assert len(payload["episodes"]) == 2


def test_loader_skips_missing_detections_without_fabricating_coordinates() -> None:
    task_input = _task_input()
    task_input["object_tracks"][0]["position"] = None
    tracks = load_calibrated_object_tracks(task_input, _calibration())
    assert [track.frame_idx for track in tracks] == [2, 3, 4, 5, 6, 7]


def test_loader_rejects_pixels_outside_calibration_frame() -> None:
    task_input = _task_input()
    task_input["object_tracks"][2]["position"]["x"] = 1000
    with pytest.raises(ValueError, match="outside the declared image frame"):
        load_calibrated_object_tracks(task_input, _calibration())


def test_loader_rejects_calibration_resolution_mismatch() -> None:
    calibration = _calibration()
    calibration["image_width_px"] = 1920
    calibration["image_height_px"] = 1080
    with pytest.raises(ValueError, match="image dimensions differ"):
        load_calibrated_object_tracks(_task_input(), calibration)


def test_loader_preserves_tracker_stream_independent_of_sparse_action_frames() -> None:
    task_input = _task_input()
    task_input["resolved_actions"] = task_input["resolved_actions"][::2]
    tracks = load_calibrated_object_tracks(task_input, _calibration())
    assert [track.frame_idx for track in tracks] == list(range(1, 8))


def test_calibration_must_match_the_tabletop_clone() -> None:
    retargeting = _retargeting()
    retargeting["tabletop_clone"]["width_m"] = 1.0
    with pytest.raises(ValueError, match="must match"):
        build_robot_pipeline(
            task_input_source=_task_input(),
            calibration_source=_calibration(),
            retargeting_source=retargeting,
            pipeline_config_source=_pipeline_config(),
        )


def test_pipeline_rejects_action_outside_consolidated_catalog() -> None:
    task_input = _task_input()
    task_input["resolved_actions"][2]["phase"] = "UNKNOWN"
    with pytest.raises(ValueError, match="Input should be"):
        build_robot_pipeline(
            task_input_source=task_input,
            calibration_source=_calibration(),
            retargeting_source=_retargeting(),
            pipeline_config_source=_pipeline_config(),
        )


def test_cli_writes_simulator_contract_without_importing_the_executor(tmp_path: Path) -> None:
    inputs = {
        "task_input.json": _task_input(),
        "calibration.json": _calibration(),
    }
    for name, payload in inputs.items():
        (tmp_path / name).write_text(json.dumps(payload))
    (tmp_path / "retargeting.yaml").write_text(yaml.safe_dump(_retargeting()))
    (tmp_path / "pipeline.yaml").write_text(yaml.safe_dump(_pipeline_config()))
    output = tmp_path / "world_waypoints.json"

    result = main(
        (
            "--task-input",
            str(tmp_path / "task_input.json"),
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
                    task_input_source=_task_input(),
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


def test_cli_writes_all_complete_episodes_by_default(tmp_path: Path) -> None:
    inputs = {
        "task_input.json": _continuation_task_input(),
        "calibration.json": _calibration(),
    }
    for name, payload in inputs.items():
        (tmp_path / name).write_text(json.dumps(payload))
    (tmp_path / "retargeting.yaml").write_text(yaml.safe_dump(_retargeting()))
    (tmp_path / "pipeline.yaml").write_text(yaml.safe_dump(_pipeline_config()))
    output = tmp_path / "world_waypoints.json"

    result = main(
        (
            "--task-input",
            str(tmp_path / "task_input.json"),
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
    assert payload["schema"] == "mimic.world_waypoint_sequence.v1"
    assert len(payload["episodes"]) == 2


def test_committed_panda_pipeline_config_is_complete() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = yaml.safe_load((root / "configs" / "robot_pipeline.yaml").read_text())[
        "robot_pipeline"
    ]
    path = PathProcessingSettings.model_validate(payload["path_processing"])
    waypoints = WaypointConstructionSettings.model_validate(payload["waypoint_construction"])

    assert path.interpolation.value == "cubic"
    assert path.corner_max_deviation_m == pytest.approx(0.01)
    assert path.output_spacing_m == pytest.approx(0.05)
    assert path.maximum_spline_deviation_m == pytest.approx(0.10)
    assert waypoints.grasp_z_m == pytest.approx(0.02)
    assert waypoints.transport_z_m == pytest.approx(0.17)
    assert waypoints.tool_quaternion_wxyz == (0.0, 1.0, 0.0, 0.0)


@pytest.mark.parametrize("render_flag", ("--viewer", "--video-out"))
def test_cli_launches_rendered_simulation_with_mjpython_on_macos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, render_flag: str
) -> None:
    inputs = {
        "task_input.json": _task_input(),
        "calibration.json": _calibration(),
    }
    for name, payload in inputs.items():
        (tmp_path / name).write_text(json.dumps(payload))
    (tmp_path / "retargeting.yaml").write_text(yaml.safe_dump(_retargeting()))
    (tmp_path / "pipeline.yaml").write_text(yaml.safe_dump(_pipeline_config()))
    (tmp_path / "robot.yaml").write_text("robot_execution: {}\n")
    calls = []

    def fake_run(command, *, cwd, check):
        calls.append((command, cwd, check))
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr("mimic.integration.robot_pipeline.sys.platform", "darwin")
    monkeypatch.setattr("mimic.integration.robot_pipeline.shutil.which", lambda name: "/mjpython")
    monkeypatch.setattr("mimic.integration.robot_pipeline.subprocess.run", fake_run)

    result = main(
        (
            "--task-input",
            str(tmp_path / "task_input.json"),
            "--calibration",
            str(tmp_path / "calibration.json"),
            "--retargeting-config",
            str(tmp_path / "retargeting.yaml"),
            "--pipeline-config",
            str(tmp_path / "pipeline.yaml"),
            "--waypoints",
            str(tmp_path / "waypoints.json"),
            "--robot-config",
            str(tmp_path / "robot.yaml"),
            "--log",
            str(tmp_path / "execution.jsonl"),
            render_flag,
        )
    )

    assert result == 0
    command, cwd, check = calls[0]
    assert command[0] == "/mjpython"
    assert command[-1] == render_flag
    assert cwd == Path(__file__).resolve().parents[1]
    assert check is False
