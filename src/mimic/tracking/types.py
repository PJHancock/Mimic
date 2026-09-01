"""Tracking-native records kept separate from calibrated robot task inputs."""

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class ImageObjectTrack:
    """One raw image-space object observation using zero-based tracker frames."""

    frame_idx: int
    center_2d: Tuple[float, float]
    bbox: Optional[Tuple[float, float, float, float]] = None
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.frame_idx, (bool, np.bool_))
            or not isinstance(self.frame_idx, Integral)
            or self.frame_idx < 0
        ):
            raise ValueError("Image tracker frame_idx must be a nonnegative integer")
        if np.shape(self.center_2d) != (2,) or not all(
            isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
            for value in self.center_2d
        ):
            raise ValueError("center_2d must contain two finite pixel coordinates")
        if not np.all(np.isfinite(self.center_2d)):
            raise ValueError("center_2d must contain two finite pixel coordinates")
        if (
            isinstance(self.confidence, (bool, np.bool_))
            or not isinstance(self.confidence, Real)
            or not np.isfinite(self.confidence)
            or not 0 <= self.confidence <= 1
        ):
            raise ValueError("confidence must be finite and in [0, 1]")
        object.__setattr__(self, "frame_idx", int(self.frame_idx))
        object.__setattr__(self, "center_2d", tuple(map(float, self.center_2d)))
        object.__setattr__(self, "confidence", float(self.confidence))
