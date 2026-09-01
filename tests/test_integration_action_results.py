"""Tests for the classifier-score to robot-action export boundary."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mimic.common import ActionPhase
from mimic.integration import (
    SkillSystemDefinition,
    build_action_inference_artifacts,
    build_demo_task_input,
    load_demo_task_input,
    load_robot_actions,
    load_skill_system,
    load_task_actions,
    predictions_from_probabilities,
    write_results,
)
from mimic.skills.types import SkillPrediction

EXPLICIT_SETTINGS = {
    "minimum_confidence": 0.3,
    "minimum_transition_margin": 0.05,
    "maximum_second_choice_gap": 0.05,
    "required_consecutive_observations": 1,
    "missing_detection_timeout_s": 0.2,
}


def configured_system() -> SkillSystemDefinition:
    root = Path(__file__).resolve().parents[1]
    template = load_skill_system(root / "configs" / "skills" / "pick_place.yaml")
    return replace(template, post_state=EXPLICIT_SETTINGS)


def test_complete_scores_are_preserved_before_single_state_export() -> None:
    system = configured_system()
    probabilities = np.array(
        [
            [0.70, 0.10, 0.05, 0.10, 0.05],
            [0.05, 0.42, 0.05, 0.43, 0.05],
        ]
    )
    predictions = predictions_from_probabilities(
        probabilities,
        [0.0, 0.1],
        system.catalog,
    )
    artifacts = build_action_inference_artifacts(predictions, system, "a" * 64)

    assert artifacts.scores.frames[1].state_scores == pytest.approx(
        dict(zip(system.catalog.labels, probabilities[1]))
    )
    # CARRY is illegal from IDLE, so the close runner-up HOVER is accepted.
    assert artifacts.decisions[1].top_skill == "CARRY"
    assert artifacts.decisions[1].second_skill == "HOVER"
    assert artifacts.robot_actions.frames[1].phase is ActionPhase.HOVER

    robot_payload = artifacts.robot_actions.model_dump(mode="json")
    assert set(robot_payload["frames"][1]) == {
        "frame_idx",
        "timestamp_s",
        "phase",
        "confidence",
        "decision_source",
    }
    assert "state_scores" not in robot_payload["frames"][1]


def test_offline_export_defers_runtime_guards_and_accepts_direct_terminal_idle() -> None:
    system = configured_system()
    winners = ("IDLE", "HOVER", "GRASP", "CARRY", "RELEASE", "IDLE")
    probabilities = np.array(
        [
            [0.8 if label == winner else 0.05 for label in system.catalog.labels]
            for winner in winners
        ]
    )
    predictions = predictions_from_probabilities(
        probabilities,
        np.arange(len(winners), dtype=float) / 10,
        system.catalog,
    )

    artifacts = build_action_inference_artifacts(predictions, system, "f" * 64)

    assert tuple(frame.phase.value for frame in artifacts.robot_actions.frames) == winners
    assert (
        artifacts.robot_actions.postprocessing.guard_policy == "defer_runtime_guards_to_execution"
    )


def test_robot_loader_returns_one_action_prediction_per_timestep(tmp_path: Path) -> None:
    system = configured_system()
    predictions = predictions_from_probabilities(
        np.array([[0.8, 0.05, 0.05, 0.05, 0.05]]),
        [0.0],
        system.catalog,
    )
    artifacts = build_action_inference_artifacts(predictions, system, "b" * 64)

    output = tmp_path / "robot_actions.json"
    write_results(output, artifacts.robot_actions)
    actions = load_robot_actions(
        output,
        expected_catalog_fingerprint=system.catalog.fingerprint,
    )
    assert len(actions) == 1
    assert actions[0].frame_idx == 1
    assert actions[0].phase is ActionPhase.IDLE
    assert actions[0].confidence == pytest.approx(0.8)


def test_consolidated_task_input_keeps_independent_streams_and_narrow_robot_view() -> None:
    system = configured_system()
    predictions = predictions_from_probabilities(
        np.array(
            [
                [0.8, 0.05, 0.05, 0.05, 0.05],
                [0.05, 0.8, 0.05, 0.05, 0.05],
            ]
        ),
        [0.0, 0.2],
        system.catalog,
        frame_indices=[1, 3],
    )
    artifacts = build_action_inference_artifacts(predictions, system, "9" * 64)
    task_input = build_demo_task_input(
        {
            "fps": 10.0,
            "duration": 0.3,
            "frame_count": 3,
            "frame_width_px": 640,
            "frame_height_px": 480,
            "positions": [
                {"frame": 0, "time": 0.0, "x": 10.0, "y": 20.0, "confidence": 0.8},
                {"frame": 1, "time": 0.1, "x": None, "y": None, "confidence": 0.0},
                {"frame": 2, "time": 0.2, "x": 30.0, "y": 40.0, "confidence": 0.7},
            ],
        },
        artifacts.robot_actions,
    )

    payload = task_input.model_dump(mode="json")
    assert payload["schema"] == "mimic.demo_task_input.v1"
    assert [frame["frame_idx"] for frame in payload["resolved_actions"]] == [1, 3]
    assert [frame["frame_idx"] for frame in payload["object_tracks"]] == [1, 2, 3]
    assert payload["object_tracks"][1]["position"] is None
    assert "state_scores" not in payload["resolved_actions"][0]
    assert "action_segments" not in payload
    assert "tracking_summary" not in payload

    loaded = load_demo_task_input(payload)
    robot_actions = load_task_actions(loaded.model_dump(mode="json"))
    assert len(robot_actions) == len(loaded.resolved_actions) == 2
    assert [action.phase for action in robot_actions] == [ActionPhase.IDLE, ActionPhase.HOVER]


def test_task_input_loader_rejects_superseded_split_artifacts() -> None:
    with pytest.raises(ValueError, match="mimic.demo_task_input.v1"):
        load_demo_task_input({"schema": "mimic.demo_results.v2"})


def test_robot_loader_rejects_string_frame_numbers() -> None:
    system = configured_system()
    predictions = predictions_from_probabilities(
        np.array([[0.8, 0.05, 0.05, 0.05, 0.05]]),
        [0.0],
        system.catalog,
    )
    artifacts = build_action_inference_artifacts(predictions, system, "e" * 64)
    payload = artifacts.robot_actions.model_dump(mode="json")
    payload["frames"][0]["frame_idx"] = "1"

    with pytest.raises(ValueError, match="one-based integer"):
        load_robot_actions(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"schema": "mimic.skill_scores.v2", "frames": []},
        {"per_frame": {"actions": ["IDLE"], "confidences": [0.9]}},
    ],
)
def test_robot_loader_rejects_raw_and_legacy_results(payload: dict) -> None:
    with pytest.raises(ValueError, match="raw scores and legacy top-one"):
        load_robot_actions(payload)


def test_missing_detection_still_exports_one_state_without_invented_confidence() -> None:
    system = configured_system()
    predictions = (
        SkillPrediction(
            frame_idx=1,
            timestamp_s=0.0,
            state_scores={
                "IDLE": 0.8,
                "HOVER": 0.05,
                "GRASP": 0.05,
                "CARRY": 0.05,
                "RELEASE": 0.05,
            },
        ),
        SkillPrediction(
            frame_idx=2,
            timestamp_s=0.3,
            state_scores={},
            detection_valid=False,
        ),
    )
    artifacts = build_action_inference_artifacts(predictions, system, "c" * 64)

    fallback = artifacts.robot_actions.frames[1]
    assert fallback.phase is ActionPhase.IDLE
    assert fallback.confidence is None
    assert fallback.decision_source.value == "no_detection_fallback"


def test_committed_post_state_settings_are_ready_for_export() -> None:
    root = Path(__file__).resolve().parents[1]
    system = load_skill_system(root / "configs" / "skills" / "pick_place.yaml")
    prediction = SkillPrediction(
        frame_idx=1,
        timestamp_s=0.0,
        state_scores={label: 0.2 for label in system.catalog.labels},
    )
    artifacts = build_action_inference_artifacts((prediction,), system, "d" * 64)
    assert artifacts.robot_actions.frames[0].phase is ActionPhase.IDLE


def test_probability_columns_must_match_catalog() -> None:
    system = configured_system()
    with pytest.raises(ValueError, match="shape"):
        predictions_from_probabilities(np.ones((2, 4)) / 4, [0.0, 0.1], system.catalog)
