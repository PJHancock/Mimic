"""Validated directed relationships between classifier-visible skills."""

from typing import Dict, Literal, Mapping, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from .catalog import SkillCatalog


class SkillTransition(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    source: str
    target: str
    variant: Optional[str] = None
    guard: Optional[str] = None
    guard_scope: Optional[Literal["runtime"]] = None

    @field_validator("source", "target", "variant", "guard")
    @classmethod
    def validate_text(cls, value):
        if value is not None and (not value or value != value.strip()):
            raise ValueError("Transition identifiers must be nonempty without edge spaces")
        return value

    @model_validator(mode="after")
    def validate_guard_scope(self) -> "SkillTransition":
        if (self.guard is None) != (self.guard_scope is None):
            raise ValueError("guard and guard_scope must either both be set or both be omitted")
        return self


class SkillGraph:
    """Small explicit adjacency graph; cycles and self-edges are valid."""

    def __init__(
        self,
        catalog: SkillCatalog,
        start_skill: str,
        transitions: Sequence[SkillTransition],
    ) -> None:
        labels = set(catalog.labels)
        if start_skill not in labels:
            raise ValueError("start_skill must exist in the skill catalog")
        parsed = tuple(SkillTransition.model_validate(edge) for edge in transitions)
        edges: Dict[Tuple[str, str], SkillTransition] = {}
        for edge in parsed:
            if edge.source not in labels or edge.target not in labels:
                raise ValueError("Every transition endpoint must exist in the skill catalog")
            key = (edge.source, edge.target)
            if key in edges:
                raise ValueError(f"Duplicate skill transition: {edge.source} -> {edge.target}")
            edges[key] = edge
        missing_self = sorted(label for label in labels if (label, label) not in edges)
        if missing_self:
            raise ValueError(f"Every skill requires an explicit self-transition: {missing_self}")
        self.catalog = catalog
        self.start_skill = start_skill
        self.transitions = parsed
        self._edges = edges

    @classmethod
    def from_mapping(cls, catalog: SkillCatalog, value: Mapping[str, object]) -> "SkillGraph":
        if set(value) != {"start_skill", "transitions"}:
            raise ValueError("Skill graph requires exactly start_skill and transitions")
        start_skill = value["start_skill"]
        transitions = value["transitions"]
        if (
            not isinstance(start_skill, str)
            or not isinstance(transitions, Sequence)
            or isinstance(transitions, (str, bytes))
        ):
            raise ValueError("Skill graph start_skill and transitions have invalid types")
        parsed = tuple(SkillTransition.model_validate(edge) for edge in transitions)
        return cls(catalog, start_skill, parsed)

    def transition(self, source: str, target: str) -> Optional[SkillTransition]:
        return self._edges.get((source, target))

    def successors(self, source: str) -> Tuple[str, ...]:
        if source not in self.catalog.labels:
            raise KeyError(f"Unknown skill ID: {source}")
        return tuple(label for label in self.catalog.labels if (source, label) in self._edges)
