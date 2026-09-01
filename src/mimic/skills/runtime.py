"""Small composition root for prediction validation and composite-skill planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Tuple, TypeVar

from .post_state import GraphStatePostProcessor
from .registry import SkillRegistry
from .types import SkillPrediction, StateDecision

ContextT = TypeVar("ContextT")
ActionT = TypeVar("ActionT")


@dataclass(frozen=True)
class SkillRuntimeResult(Generic[ActionT]):
    decision: StateDecision
    actions: Tuple[ActionT, ...]


class SkillRuntime(Generic[ContextT, ActionT]):
    """Connect a generic graph resolver to a deployment-specific handler registry."""

    def __init__(
        self,
        postprocessor: GraphStatePostProcessor,
        registry: SkillRegistry[ContextT, ActionT],
    ) -> None:
        if postprocessor.catalog.fingerprint != registry.catalog.fingerprint:
            raise ValueError("Postprocessor and registry catalogs must match")
        self.postprocessor = postprocessor
        self.registry = registry

    def process(
        self, prediction: SkillPrediction, context: ContextT
    ) -> SkillRuntimeResult[ActionT]:
        decision = self.postprocessor.process(prediction)
        return SkillRuntimeResult(decision, self.registry.plan(decision, context))
