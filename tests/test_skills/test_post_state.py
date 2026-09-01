"""Tests for graph-aware state post-processing."""

from __future__ import annotations

import pytest

from mimic.skills.post_state import GraphStatePostProcessor, PostStateSettings
from mimic.skills.types import DecisionSource, SkillPrediction

LABELS = ("IDLE", "HOVER", "GRASP", "CARRY", "RELEASE")


def scores(**overrides: float) -> dict[str, float]:
    values = dict.fromkeys(LABELS, 0.0)
    values.update(overrides)
    unspecified = [label for label in LABELS if label not in overrides]
    remainder = 1.0 - sum(overrides.values())
    if unspecified:
        share = remainder / len(unspecified)
        for label in unspecified:
            values[label] = share
    return values


def prediction(timestamp_s: float, **overrides: float) -> SkillPrediction:
    return SkillPrediction(timestamp_s=timestamp_s, state_scores=scores(**overrides))


def test_accepts_legal_top_choice(post_processor: GraphStatePostProcessor) -> None:
    decision = post_processor.process(prediction(0.0, HOVER=0.8))
    assert decision.accepted_skill == "HOVER"
    assert decision.selected_rank == 1
    assert decision.transition is not None
    assert decision.transition.variant == "TO_GRASP"


def test_accepts_release_idle_regrasp_continuation(
    skill_catalog, skill_graph, post_state_settings
) -> None:
    post_processor = GraphStatePostProcessor(
        skill_catalog,
        skill_graph,
        post_state_settings,
        transition_guard=lambda _transition: True,
    )
    post_processor.process(prediction(0.0, HOVER=0.8))
    post_processor.process(prediction(0.1, GRASP=0.8))
    post_processor.process(prediction(0.2, CARRY=0.8))
    post_processor.process(prediction(0.3, RELEASE=0.8))
    idle = post_processor.process(prediction(0.4, IDLE=0.8))
    regrasp = post_processor.process(prediction(0.5, GRASP=0.8))

    assert idle.accepted_skill == "IDLE"
    assert regrasp.accepted_skill == "GRASP"
    assert regrasp.transition is not None
    assert regrasp.transition.variant == "CONTINUATION_REGRASP"


def test_uses_second_choice_when_top_choice_is_illegal(
    post_processor: GraphStatePostProcessor,
) -> None:
    post_processor.reset("RELEASE")
    decision = post_processor.process(
        SkillPrediction(
            timestamp_s=0.0,
            state_scores={
                "IDLE": 0.03,
                "HOVER": 0.44,
                "GRASP": 0.03,
                "CARRY": 0.46,
                "RELEASE": 0.04,
            },
        )
    )
    assert decision.top_skill == "CARRY"
    assert decision.second_skill == "HOVER"
    assert decision.accepted_skill == "HOVER"
    assert decision.selected_rank == 2


def test_holds_when_illegal_top_choice_is_too_dominant(
    post_processor: GraphStatePostProcessor,
) -> None:
    post_processor.reset("CARRY")
    decision = post_processor.process(prediction(0.0, HOVER=0.55, RELEASE=0.25))
    assert decision.accepted_skill == "CARRY"
    assert decision.selected_rank is None
    assert decision.reason == "no_legal_top_two_candidate"


def test_guard_rejection_holds_current_state(
    post_state_settings, skill_catalog, skill_graph
) -> None:
    processor = GraphStatePostProcessor(
        skill_catalog,
        skill_graph,
        post_state_settings,
        transition_guard=lambda transition: transition.guard != "grasp_confirmed",
    )
    processor.reset("GRASP")
    decision = processor.process(prediction(0.0, CARRY=0.8))
    assert decision.accepted_skill == "GRASP"
    assert decision.reason == "transition_guard_blocked"


def test_named_guard_fails_closed_without_runtime_evaluator(
    post_state_settings, skill_catalog, skill_graph
) -> None:
    processor = GraphStatePostProcessor(skill_catalog, skill_graph, post_state_settings)
    processor.reset("GRASP")
    decision = processor.process(prediction(0.0, CARRY=0.8))
    assert decision.accepted_skill == "GRASP"
    assert decision.reason == "transition_guard_blocked"


def test_requires_configured_observation_persistence(
    post_state_settings, skill_catalog, skill_graph
) -> None:
    settings = {**post_state_settings, "required_consecutive_observations": 2}
    processor = GraphStatePostProcessor(skill_catalog, skill_graph, settings)
    first = processor.process(prediction(0.0, HOVER=0.8))
    second = processor.process(prediction(0.1, HOVER=0.8))
    assert first.accepted_skill == "IDLE"
    assert first.pending_observations == 1
    assert second.accepted_skill == "HOVER"


def test_missing_detection_suspends_to_idle_then_resumes(
    post_processor: GraphStatePostProcessor,
) -> None:
    post_processor.reset("CARRY")
    post_processor.process(prediction(0.0, CARRY=0.8))
    pending = post_processor.process(
        SkillPrediction(timestamp_s=0.1, state_scores={}, detection_valid=False)
    )
    fallback = post_processor.process(
        SkillPrediction(timestamp_s=0.2, state_scores={}, detection_valid=False)
    )
    resumed = post_processor.process(prediction(0.3, CARRY=0.8))
    assert pending.accepted_skill == "CARRY"
    assert fallback.accepted_skill == "IDLE"
    assert fallback.source is DecisionSource.NO_DETECTION_FALLBACK
    assert fallback.suspended_from == "CARRY"
    assert resumed.accepted_skill == "CARRY"
    assert resumed.suspended_from == "CARRY"


def test_ties_use_catalog_order(post_processor: GraphStatePostProcessor) -> None:
    decision = post_processor.process(prediction(0.0, IDLE=0.4, HOVER=0.4))
    assert decision.top_skill == "IDLE"
    assert decision.accepted_skill == "IDLE"


def test_rejects_score_schema_mismatch(post_processor: GraphStatePostProcessor) -> None:
    with pytest.raises(ValueError, match="Prediction/catalog mismatch"):
        post_processor.process(SkillPrediction(timestamp_s=0.0, state_scores={"IDLE": 1.0}))


def test_rejects_non_monotonic_time(post_processor: GraphStatePostProcessor) -> None:
    post_processor.process(prediction(1.0, IDLE=0.8))
    with pytest.raises(ValueError, match="strictly increasing"):
        post_processor.process(prediction(0.9, IDLE=0.8))


def test_committed_template_has_explicit_validated_thresholds(skill_config: dict) -> None:
    settings = PostStateSettings.model_validate(skill_config["post_state"])
    assert settings.minimum_confidence == pytest.approx(0.5)
