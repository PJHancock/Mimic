"""Tests for the versioned skill catalog and relationship graph."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mimic.skills.catalog import SkillCatalog, SkillSpec
from mimic.skills.graph import SkillGraph, SkillTransition


def test_catalog_preserves_training_order_and_handler_contract(skill_catalog: SkillCatalog) -> None:
    assert skill_catalog.labels == ("IDLE", "HOVER", "GRASP", "CARRY", "RELEASE")
    assert skill_catalog.class_count == 5
    assert skill_catalog.index_for("CARRY") == 3
    assert skill_catalog.label_for(4) == "RELEASE"
    assert len(skill_catalog.fingerprint) == 64
    skill_catalog.require_handlers({"idle", "hover", "grasp", "carry", "release"})


@pytest.mark.parametrize("field", ["id", "training_index", "handler_id"])
def test_catalog_rejects_duplicate_identifiers(field: str) -> None:
    first = {"id": "A", "training_index": 0, "handler_id": "a"}
    second = {"id": "B", "training_index": 1, "handler_id": "b"}
    second[field] = first[field]
    with pytest.raises(ValidationError):
        SkillCatalog(schema_version=1, skills=(SkillSpec(**first), SkillSpec(**second)))


def test_catalog_fingerprint_changes_with_schema_version(skill_catalog: SkillCatalog) -> None:
    changed = skill_catalog.model_copy(update={"schema_version": 3})
    assert changed.fingerprint != skill_catalog.fingerprint


def test_catalog_rejects_handler_mismatch(skill_catalog: SkillCatalog) -> None:
    with pytest.raises(ValueError, match="Handler/catalog mismatch"):
        skill_catalog.require_handlers({"idle"})


def test_graph_exposes_contextual_hover_edges(skill_graph: SkillGraph) -> None:
    assert skill_graph.transition("IDLE", "HOVER").variant == "TO_GRASP"
    assert skill_graph.transition("RELEASE", "HOVER").variant == "TO_HOME"
    assert skill_graph.transition("GRASP", "HOVER").guard == "grasp_empty"
    assert skill_graph.transition("GRASP", "CARRY").guard_scope == "runtime"


def test_idle_is_a_legal_terminal_state_from_every_skill(skill_graph: SkillGraph) -> None:
    assert all(
        skill_graph.transition(source, "IDLE") is not None for source in skill_graph.catalog.labels
    )


def test_named_guard_requires_an_explicit_scope() -> None:
    with pytest.raises(ValueError, match="guard and guard_scope"):
        SkillTransition(source="GRASP", target="CARRY", guard="grasp_confirmed")


def test_graph_requires_self_edge_for_every_skill(skill_catalog: SkillCatalog) -> None:
    edges = tuple(
        edge
        for edge in SkillGraph.from_mapping(
            skill_catalog,
            {
                "start_skill": "IDLE",
                "transitions": [
                    {"source": label, "target": label} for label in skill_catalog.labels
                ],
            },
        ).transitions
        if edge.source != "CARRY"
    )
    with pytest.raises(ValueError, match="self-transition"):
        SkillGraph(catalog=skill_catalog, start_skill="IDLE", transitions=edges)


def test_graph_rejects_unknown_and_duplicate_edges(skill_catalog: SkillCatalog) -> None:
    self_edges = tuple(
        SkillTransition(source=label, target=label) for label in skill_catalog.labels
    )
    with pytest.raises(ValueError, match="transition endpoint"):
        SkillGraph(
            catalog=skill_catalog,
            start_skill="IDLE",
            transitions=self_edges + (SkillTransition(source="IDLE", target="UNKNOWN"),),
        )
    with pytest.raises(ValueError, match="Duplicate skill transition"):
        SkillGraph(
            catalog=skill_catalog,
            start_skill="IDLE",
            transitions=self_edges + (SkillTransition(source="IDLE", target="IDLE"),),
        )
