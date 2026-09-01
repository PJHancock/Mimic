"""Composite labels bind to replaceable handlers and low-level action plans."""

from __future__ import annotations

from dataclasses import replace

import pytest

from mimic.common.types import PickPlaceWaypoints, ToolPose
from mimic.robot.action_primitives import CartesianMotion, JointPresetMotion
from mimic.robot.pick_place_skills import PickPlaceSkillContext, build_pick_place_skill_registry
from mimic.robot.presets import JointPreset
from mimic.skills import GraphStatePostProcessor, SkillPrediction, SkillRuntime


def pose(x: float = 0, z: float = 0) -> ToolPose:
    return ToolPose((x, 0, z), (1, 0, 0, 0))


@pytest.fixture
def skill_context() -> PickPlaceSkillContext:
    waypoints = PickPlaceWaypoints(
        pose(z=0.1),
        pose(),
        pose(z=0.1),
        (pose(0.1, 0.1), pose(0.2, 0.1)),
        pose(0.2),
        pose(0.2, 0.1),
        (0.2, 0, 0),
    )
    return PickPlaceSkillContext(waypoints, JointPreset("home", {"j1": 0.1, "j2": -0.2}))


def prediction(timestamp_s: float, winner: str) -> SkillPrediction:
    labels = ("IDLE", "HOVER", "GRASP", "CARRY", "RELEASE")
    return SkillPrediction(
        timestamp_s,
        {label: 0.8 if label == winner else 0.05 for label in labels},
    )


def test_hover_entry_from_idle_moves_to_grasp_hover(
    skill_catalog, skill_graph, post_state_settings, skill_context
) -> None:
    processor = GraphStatePostProcessor(skill_catalog, skill_graph, post_state_settings)
    registry = build_pick_place_skill_registry(skill_catalog)
    decision = processor.process(prediction(0, "HOVER"))
    actions = registry.plan(decision, skill_context)
    assert len(actions) == 1
    assert isinstance(actions[0], CartesianMotion)
    assert actions[0].primitive_id == "MOVE_TO_GRASP_HOVER"
    assert actions[0].target == skill_context.waypoints.approach


@pytest.mark.parametrize("source", ["RELEASE", "GRASP"])
def test_contextual_hover_returns_to_saved_joint_home(
    source, skill_catalog, skill_graph, post_state_settings, skill_context
) -> None:
    processor = GraphStatePostProcessor(
        skill_catalog,
        skill_graph,
        post_state_settings,
        transition_guard=lambda transition: True,
    )
    processor.reset(source)
    registry = build_pick_place_skill_registry(skill_catalog)
    decision = processor.process(prediction(0, "HOVER"))
    actions = registry.plan(decision, skill_context)
    assert len(actions) == 1
    assert isinstance(actions[0], JointPresetMotion)
    assert actions[0].preset_id == "home"
    assert actions[0].joint_positions == {"j1": 0.1, "j2": -0.2}


def test_repeated_label_does_not_restart_composite_skill(
    skill_catalog, skill_graph, post_state_settings, skill_context
) -> None:
    processor = GraphStatePostProcessor(skill_catalog, skill_graph, post_state_settings)
    registry = build_pick_place_skill_registry(skill_catalog)
    decision = processor.process(prediction(0, "IDLE"))
    assert registry.plan(decision, skill_context) == ()


def test_carry_handler_expands_to_lift_and_each_processed_waypoint(
    skill_catalog, skill_graph, post_state_settings, skill_context
) -> None:
    processor = GraphStatePostProcessor(
        skill_catalog,
        skill_graph,
        post_state_settings,
        transition_guard=lambda transition: True,
    )
    processor.reset("GRASP")
    registry = build_pick_place_skill_registry(skill_catalog)
    decision = processor.process(prediction(0, "CARRY"))
    actions = registry.plan(decision, skill_context)
    assert [action.primitive_id for action in actions] == [
        "LIFT",
        "FOLLOW_PATH",
        "FOLLOW_PATH",
    ]


def test_registry_rejects_decision_from_an_incompatible_catalog(
    skill_catalog, skill_graph, post_state_settings, skill_context
) -> None:
    processor = GraphStatePostProcessor(skill_catalog, skill_graph, post_state_settings)
    registry = build_pick_place_skill_registry(skill_catalog)
    decision = processor.process(prediction(0, "HOVER"))
    with pytest.raises(ValueError, match="fingerprints"):
        registry.plan(replace(decision, catalog_fingerprint="other"), skill_context)


def test_resumed_skill_does_not_restart_its_composite_plan(
    skill_catalog, skill_graph, post_state_settings, skill_context
) -> None:
    processor = GraphStatePostProcessor(skill_catalog, skill_graph, post_state_settings)
    processor.reset("CARRY")
    processor.process(prediction(0, "CARRY"))
    processor.process(SkillPrediction(0.2, {}, detection_valid=False))
    resumed = processor.process(prediction(0.3, "CARRY"))
    assert resumed.suspended_from == "CARRY"
    assert build_pick_place_skill_registry(skill_catalog).plan(resumed, skill_context) == ()


def test_runtime_composes_prediction_validation_and_handler_planning(
    skill_catalog, skill_graph, post_state_settings, skill_context
) -> None:
    processor = GraphStatePostProcessor(skill_catalog, skill_graph, post_state_settings)
    runtime = SkillRuntime(processor, build_pick_place_skill_registry(skill_catalog))
    result = runtime.process(prediction(0, "HOVER"), skill_context)
    assert result.decision.accepted_skill == "HOVER"
    assert [action.primitive_id for action in result.actions] == ["MOVE_TO_GRASP_HOVER"]
