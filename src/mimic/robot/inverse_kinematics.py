"""Backend-independent IK contract and explicit numerical settings."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from mimic.common.types import IKResult, RobotState, ToolPose


class IKSolver(Protocol):
    def solve(self, target: ToolPose, state: RobotState, dt_s: float) -> IKResult:
        """Return one bounded joint-position step without advancing the simulation."""
        ...


@dataclass(frozen=True)
class IKSettings:
    position_tolerance_m: float
    orientation_tolerance_rad: float
    position_cost: float
    orientation_cost: float
    posture_cost: float
    damping: float
    task_gain: float

    def __post_init__(self):
        values = tuple(vars(self).values())
        if not np.all(np.isfinite(values)) or min(values) < 0:
            raise ValueError("IK settings must be finite and nonnegative")
        if (
            min(
                self.position_tolerance_m,
                self.orientation_tolerance_rad,
                self.position_cost,
                self.orientation_cost,
                self.damping,
            )
            <= 0
        ):
            raise ValueError("Pose tolerances, pose costs and damping must be positive")
        if not 0 < self.task_gain <= 1:
            raise ValueError("Task gain must be in (0, 1]")
