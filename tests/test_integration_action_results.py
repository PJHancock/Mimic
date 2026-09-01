"""Tests for the classifier-score to robot-action export boundary."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mimic.common import ActionPhase
from mimic.integration import (
    SkillSystemDefinition,
    build_action_inference_artifacts,
    load_robot_actions,
    load_skill_system,
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


def test_committed_null_post_state_settings_fail_closed() -> None:
    root = Path(__file__).resolve().parents[1]
    system = load_skill_system(root / "configs" / "skills" / "pick_place.yaml")
    prediction = SkillPrediction(
        frame_idx=1,
        timestamp_s=0.0,
        state_scores={label: 0.2 for label in system.catalog.labels},
    )
    with pytest.raises(ValueError, match="Post-state settings are unresolved"):
        build_action_inference_artifacts((prediction,), system, "d" * 64)


def test_probability_columns_must_match_catalog() -> None:
    system = configured_system()
    with pytest.raises(ValueError, match="shape"):
        predictions_from_probabilities(np.ones((2, 4)) / 4, [0.0, 0.1], system.catalog)
