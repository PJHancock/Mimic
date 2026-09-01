"""Handler registry that keeps classifier labels separate from robot implementation."""

from __future__ import annotations

from typing import Generic, Mapping, Protocol, Tuple, TypeVar

from .catalog import SkillCatalog
from .types import StateDecision

ContextT = TypeVar("ContextT")
ActionT = TypeVar("ActionT")
HandlerContextT = TypeVar("HandlerContextT", contravariant=True)
HandlerActionT = TypeVar("HandlerActionT", covariant=True)


class SkillHandler(Protocol[HandlerContextT, HandlerActionT]):
    def __call__(
        self, decision: StateDecision, context: HandlerContextT
    ) -> Tuple[HandlerActionT, ...]: ...


class SkillRegistry(Generic[ContextT, ActionT]):
    """Resolve the catalog's handler IDs without hardcoding robot names or labels."""

    def __init__(
        self,
        catalog: SkillCatalog,
        handlers: Mapping[str, SkillHandler[ContextT, ActionT]],
    ) -> None:
        catalog.require_handlers(tuple(handlers))
        self.catalog = catalog
        self.handlers = dict(handlers)
        self._handler_by_skill = {
            skill.id: self.handlers[skill.handler_id] for skill in catalog.skills
        }

    def plan(self, decision: StateDecision, context: ContextT) -> Tuple[ActionT, ...]:
        if decision.catalog_fingerprint != self.catalog.fingerprint:
            raise ValueError("Decision and skill registry catalog fingerprints do not match")
        if decision.accepted_skill not in self._handler_by_skill:
            raise ValueError(f"Unknown accepted skill: {decision.accepted_skill}")
        # Stable repeated labels describe persistence, not a request to restart a composite skill.
        if decision.accepted_skill == decision.previous_skill:
            return ()
        # A resumed suspended skill continues its retained action cursor.
        if decision.suspended_from == decision.accepted_skill:
            return ()
        return self._handler_by_skill[decision.accepted_skill](decision, context)
