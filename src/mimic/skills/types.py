"""Immutable prediction and decision records for skill-state processing."""

from dataclasses import dataclass
from enum import Enum
from numbers import Integral, Real
from typing import Mapping, Optional, Tuple

import numpy as np

from .graph import SkillTransition


class DecisionSource(str, Enum):
    MODEL = "model"
    EPISODE_BOUNDARY = "episode_boundary"
    NO_DETECTION_FALLBACK = "no_detection_fallback"


@dataclass(frozen=True)
class SkillPrediction:
    timestamp_s: float
    state_scores: Mapping[str, float]
    detection_valid: bool = True
    frame_idx: Optional[int] = None

    def __post_init__(self):
        if (
            isinstance(self.timestamp_s, (bool, np.bool_))
            or not isinstance(self.timestamp_s, Real)
            or not np.isfinite(self.timestamp_s)
            or self.timestamp_s < 0
        ):
            raise ValueError("timestamp_s must be finite, nonnegative seconds")
        if not isinstance(self.detection_valid, (bool, np.bool_)):
            raise ValueError("detection_valid must be boolean")
        if self.frame_idx is not None and (
            isinstance(self.frame_idx, (bool, np.bool_))
            or not isinstance(self.frame_idx, Integral)
            or self.frame_idx < 1
        ):
            raise ValueError("frame_idx must be a positive integer or None")
        scores = dict(self.state_scores)
        if self.detection_valid and not scores:
            raise ValueError("A valid detection requires state scores")
        if not self.detection_valid and scores:
            raise ValueError("An invalid detection must not carry model scores")
        for label, score in scores.items():
            if not isinstance(label, str) or not label or label != label.strip():
                raise ValueError("Score labels must be nonempty strings without edge spaces")
            if (
                isinstance(score, (bool, np.bool_))
                or not isinstance(score, Real)
                or not np.isfinite(score)
                or not 0 <= score <= 1
            ):
                raise ValueError("State scores must be finite probabilities in [0, 1]")
        if scores and not np.isclose(sum(scores.values()), 1.0, rtol=0, atol=1e-6):
            raise ValueError("State probabilities must sum to one")
        object.__setattr__(self, "timestamp_s", float(self.timestamp_s))
        object.__setattr__(self, "state_scores", scores)
        object.__setattr__(self, "detection_valid", bool(self.detection_valid))
        if self.frame_idx is not None:
            object.__setattr__(self, "frame_idx", int(self.frame_idx))


@dataclass(frozen=True)
class StateDecision:
    timestamp_s: float
    previous_skill: str
    accepted_skill: str
    selected_rank: Optional[int]
    top_skill: Optional[str]
    second_skill: Optional[str]
    confidence: Optional[float]
    reason: str
    source: DecisionSource
    transition: Optional[SkillTransition]
    pending_observations: int
    catalog_fingerprint: str
    frame_idx: Optional[int] = None
    suspended_from: Optional[str] = None

    @property
    def edge(self) -> Optional[Tuple[str, str]]:
        if self.transition is None:
            return None
        return (self.transition.source, self.transition.target)
