"""Model-independent gripper requests and measured completion criteria."""

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Protocol, Tuple

import numpy as np

from mimic.common.types import GripperFeedback


class GripperDriver(Protocol):
    actuator_names: Tuple[str, ...]
    open_width_m: float
    closed_width_m: float

    def controls(self, width_m: float) -> Mapping[str, float]:
        """Convert nominal total opening to the model's actuator controls."""
        ...


class GripperAction(str, Enum):
    OPEN = "OPEN"
    CLOSE = "CLOSE"
    HOLD = "HOLD"


class GripperStatus(str, Enum):
    MOVING = "MOVING"
    OPEN = "OPEN"
    CANDIDATE_GRASP = "CANDIDATE_GRASP"
    EMPTY = "EMPTY"
    TIMEOUT = "TIMEOUT"


@dataclass(frozen=True)
class GripperSettings:
    width_tolerance_m: float
    empty_width_m: float
    contact_force_n: float
    contact_duration_s: float
    movement_timeout_s: float

    def __post_init__(self):
        values = tuple(vars(self).values())
        if not np.all(np.isfinite(values)) or min(values) <= 0:
            raise ValueError("Gripper criteria must be explicitly positive and finite")
        if self.contact_duration_s >= self.movement_timeout_s:
            raise ValueError("Contact duration must be below the movement timeout")


@dataclass(frozen=True)
class GripperResult:
    status: GripperStatus
    actuator_targets: Mapping[str, float]
    width_m: float
    action: GripperAction


class GripperLogic:
    def __init__(self, driver: GripperDriver, settings: GripperSettings):
        if not driver.closed_width_m <= settings.empty_width_m < driver.open_width_m:
            raise ValueError("Empty-grasp width must lie within the driver's opening range")
        if settings.width_tolerance_m >= driver.open_width_m - driver.closed_width_m:
            raise ValueError("Width tolerance must distinguish opening from closure")
        self.driver, self.settings = driver, settings
        self.reset()

    def reset(self):
        self._action = None
        self._started = self._last_time = self._contact_started = None

    def update(
        self, action: GripperAction, time_s: float, feedback: GripperFeedback
    ) -> GripperResult:
        action = GripperAction(action)
        readings = (time_s, feedback.width_m, feedback.speed_m_s, *feedback.finger_contact_forces_n)
        if not np.all(np.isfinite(readings)) or time_s < 0:
            raise ValueError("Gripper observation must be finite with nonnegative time")
        if self._last_time is not None and time_s < self._last_time:
            raise ValueError("Gripper time moved backwards; reset the logic with the simulation")
        self._last_time = time_s
        if action == GripperAction.HOLD:
            if self._action is None:
                raise ValueError("HOLD requires an earlier OPEN or CLOSE request")
            action = self._action
        if action != self._action:
            self._action, self._started, self._contact_started = action, time_s, None
        opening = action == GripperAction.OPEN
        target = self.driver.open_width_m if opening else self.driver.closed_width_m
        status = GripperStatus.MOVING
        forces = feedback.finger_contact_forces_n
        candidate = (
            len(forces) >= 2
            and min(forces) >= self.settings.contact_force_n
            and feedback.width_m > self.settings.empty_width_m
        )
        if opening:
            if abs(feedback.width_m - target) <= self.settings.width_tolerance_m:
                status = GripperStatus.OPEN
        elif candidate:
            if self._contact_started is None:
                self._contact_started = time_s
            if time_s - self._contact_started >= self.settings.contact_duration_s:
                status = GripperStatus.CANDIDATE_GRASP
        else:
            self._contact_started = None
            if feedback.width_m <= self.settings.empty_width_m:
                status = GripperStatus.EMPTY
        if (
            status == GripperStatus.MOVING
            and time_s - self._started >= self.settings.movement_timeout_s
        ):
            status = GripperStatus.TIMEOUT
        return GripperResult(status, self.driver.controls(target), feedback.width_m, action)
