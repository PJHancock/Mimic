"""Graph-aware validation of top-one and second-place skill predictions."""

from numbers import Integral, Real
from typing import Callable, Mapping, Optional, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator

from .catalog import SkillCatalog
from .graph import SkillGraph, SkillTransition
from .types import DecisionSource, SkillPrediction, StateDecision

TransitionGuard = Callable[[SkillTransition], bool]


class PostStateSettings(BaseModel):
    model_config = ConfigDict(
        frozen=True, extra="forbid", allow_inf_nan=False, revalidate_instances="always"
    )

    minimum_confidence: float
    minimum_transition_margin: float
    maximum_second_choice_gap: float
    required_consecutive_observations: int
    missing_detection_timeout_s: float

    @field_validator(
        "minimum_confidence",
        "minimum_transition_margin",
        "maximum_second_choice_gap",
        mode="before",
    )
    @classmethod
    def validate_probability(cls, value):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Real)
            or not np.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError("Probability thresholds must be finite values in [0, 1]")
        return float(value)

    @field_validator("required_consecutive_observations", mode="before")
    @classmethod
    def validate_persistence(cls, value):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
            raise ValueError("required_consecutive_observations must be a positive integer")
        return int(value)

    @field_validator("missing_detection_timeout_s", mode="before")
    @classmethod
    def validate_timeout(cls, value):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Real)
            or not np.isfinite(value)
            or value < 0
        ):
            raise ValueError("missing_detection_timeout_s must be finite nonnegative seconds")
        return float(value)


class GraphStatePostProcessor:
    """Stateful resolver; it never executes a robot skill itself."""

    current_skill: str
    suspended_from: Optional[str]
    _pending_skill: Optional[str]
    _pending_count: int
    _last_timestamp: Optional[float]
    _last_valid_detection_timestamp: Optional[float]

    def __init__(
        self,
        catalog: SkillCatalog,
        graph: SkillGraph,
        settings: Mapping[str, object],
        transition_guard: Optional[TransitionGuard] = None,
    ) -> None:
        if graph.catalog.fingerprint != catalog.fingerprint:
            raise ValueError("Skill graph and postprocessor catalogs must match")
        self.catalog = catalog
        self.graph = graph
        self.settings = PostStateSettings.model_validate(settings)
        # Named observation guards fail closed unless the runtime supplies an evaluator.
        self.transition_guard = transition_guard or (lambda transition: transition.guard is None)
        self.reset()

    def reset(self, initial_skill: Optional[str] = None) -> None:
        current_skill = initial_skill or self.graph.start_skill
        if current_skill not in self.catalog.labels:
            raise ValueError(f"Unknown initial skill: {current_skill}")
        self.current_skill = current_skill
        self.suspended_from = None
        self._pending_skill = None
        self._pending_count = 0
        self._last_timestamp = None
        self._last_valid_detection_timestamp = None

    def _decision(
        self,
        prediction: SkillPrediction,
        previous: str,
        accepted: str,
        reason: str,
        source: DecisionSource,
        selected_rank: Optional[int] = None,
        ranking: Tuple[Tuple[str, float], ...] = (),
        transition: Optional[SkillTransition] = None,
    ) -> StateDecision:
        return StateDecision(
            timestamp_s=prediction.timestamp_s,
            previous_skill=previous,
            accepted_skill=accepted,
            selected_rank=selected_rank,
            top_skill=ranking[0][0] if ranking else None,
            second_skill=ranking[1][0] if len(ranking) > 1 else None,
            confidence=(ranking[selected_rank - 1][1] if selected_rank is not None else None),
            reason=reason,
            source=source,
            transition=transition,
            pending_observations=self._pending_count,
            catalog_fingerprint=self.catalog.fingerprint,
            frame_idx=prediction.frame_idx,
            suspended_from=self.suspended_from,
        )

    def _rank(self, scores: Mapping[str, float]) -> Tuple[Tuple[str, float], ...]:
        if set(scores) != set(self.catalog.labels):
            missing = sorted(set(self.catalog.labels) - set(scores))
            extra = sorted(set(scores) - set(self.catalog.labels))
            raise ValueError(f"Prediction/catalog mismatch; missing={missing}, extra={extra}")
        order = {label: index for index, label in enumerate(self.catalog.labels)}
        return tuple(sorted(scores.items(), key=lambda item: (-item[1], order[item[0]])))

    def process(self, prediction: SkillPrediction) -> StateDecision:
        prediction = SkillPrediction(**vars(prediction))
        if self._last_timestamp is not None and prediction.timestamp_s <= self._last_timestamp:
            raise ValueError("Prediction timestamps must be strictly increasing")
        self._last_timestamp = prediction.timestamp_s
        previous = self.current_skill

        if not prediction.detection_valid:
            if self._last_valid_detection_timestamp is None:
                timed_out = True
            else:
                timed_out = (
                    prediction.timestamp_s - self._last_valid_detection_timestamp
                    >= self.settings.missing_detection_timeout_s
                )
            self._pending_skill, self._pending_count = None, 0
            if not timed_out:
                return self._decision(
                    prediction,
                    previous,
                    previous,
                    "missing_detection_timeout_pending",
                    DecisionSource.NO_DETECTION_FALLBACK,
                )
            if previous != self.graph.start_skill and self.suspended_from is None:
                self.suspended_from = previous
            self.current_skill = self.graph.start_skill
            return self._decision(
                prediction,
                previous,
                self.current_skill,
                "missing_detection_idle_hold",
                DecisionSource.NO_DETECTION_FALLBACK,
            )

        self._last_valid_detection_timestamp = prediction.timestamp_s
        ranking = self._rank(prediction.state_scores)
        base = self.suspended_from or self.current_skill
        candidates = tuple(enumerate(ranking[:2], start=1))
        selected_rank: Optional[int] = None
        candidate: Optional[str] = None
        transition: Optional[SkillTransition] = None
        for rank, (label, score) in candidates:
            edge_source = self.current_skill if label == self.graph.start_skill else base
            edge = self.graph.transition(edge_source, label)
            if edge is None:
                continue
            if rank == 2 and ranking[0][1] - score > self.settings.maximum_second_choice_gap:
                continue
            selected_rank, candidate, transition = rank, label, edge
            break

        if candidate is None:
            self._pending_skill, self._pending_count = None, 0
            return self._decision(
                prediction,
                previous,
                previous,
                "no_legal_top_two_candidate",
                DecisionSource.MODEL,
                ranking=ranking,
            )

        assert selected_rank is not None and transition is not None
        confidence = ranking[selected_rank - 1][1]
        if confidence < self.settings.minimum_confidence:
            self._pending_skill, self._pending_count = None, 0
            return self._decision(
                prediction,
                previous,
                previous,
                "candidate_below_minimum_confidence",
                DecisionSource.MODEL,
                selected_rank=selected_rank,
                ranking=ranking,
                transition=transition,
            )

        if candidate in (self.current_skill, base):
            self._pending_skill, self._pending_count = None, 0
            self.current_skill = candidate
            decision = self._decision(
                prediction,
                previous,
                candidate,
                "top_candidate_retained" if selected_rank == 1 else "second_candidate_retained",
                DecisionSource.MODEL,
                selected_rank=selected_rank,
                ranking=ranking,
                transition=transition,
            )
            if candidate != self.graph.start_skill:
                self.suspended_from = None
            return decision

        base_score = prediction.state_scores[base]
        if confidence - base_score < self.settings.minimum_transition_margin:
            self._pending_skill, self._pending_count = None, 0
            return self._decision(
                prediction,
                previous,
                previous,
                "candidate_below_transition_margin",
                DecisionSource.MODEL,
                selected_rank=selected_rank,
                ranking=ranking,
                transition=transition,
            )
        if not self.transition_guard(transition):
            self._pending_skill, self._pending_count = None, 0
            return self._decision(
                prediction,
                previous,
                previous,
                "transition_guard_blocked",
                DecisionSource.MODEL,
                selected_rank=selected_rank,
                ranking=ranking,
                transition=transition,
            )

        if candidate == self._pending_skill:
            self._pending_count += 1
        else:
            self._pending_skill, self._pending_count = candidate, 1
        if self._pending_count < self.settings.required_consecutive_observations:
            return self._decision(
                prediction,
                previous,
                previous,
                "transition_persistence_pending",
                DecisionSource.MODEL,
                selected_rank=selected_rank,
                ranking=ranking,
                transition=transition,
            )

        self.current_skill = candidate
        self.suspended_from = None
        self._pending_skill, self._pending_count = None, 0
        return self._decision(
            prediction,
            previous,
            candidate,
            "top_legal_transition" if selected_rank == 1 else "top_illegal_second_legal",
            DecisionSource.MODEL,
            selected_rank=selected_rank,
            ranking=ranking,
            transition=transition,
        )
