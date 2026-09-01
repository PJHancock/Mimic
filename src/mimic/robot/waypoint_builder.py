"""Build executable fixed-orientation tool poses from processed XY geometry."""

from numbers import Real
from typing import Mapping, Tuple, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator

from mimic.common.types import PickPlaceWaypoints, ToolPose
from mimic.robot.path_processing import ProcessedPath


class WaypointConstructionSettings(BaseModel):
    """Explicit world-Z geometry and orientation; no robot-specific defaults."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", allow_inf_nan=False, revalidate_instances="always"
    )

    approach_z_m: float
    grasp_z_m: float
    lift_z_m: float
    transport_z_m: float
    lower_z_m: float
    retreat_z_m: float
    object_goal_z_m: float
    tool_quaternion_wxyz: Tuple[float, float, float, float]

    @field_validator(
        "approach_z_m",
        "grasp_z_m",
        "lift_z_m",
        "transport_z_m",
        "lower_z_m",
        "retreat_z_m",
        "object_goal_z_m",
        mode="before",
    )
    @classmethod
    def validate_z(cls, value):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError("Waypoint Z coordinates must be real numbers")
        if not np.isfinite(value):
            raise ValueError("Waypoint Z coordinates must be finite meters")
        return float(value)

    @field_validator("tool_quaternion_wxyz", mode="before")
    @classmethod
    def validate_orientation(cls, value):
        if np.shape(value) != (4,) or any(
            isinstance(entry, (bool, np.bool_)) or not isinstance(entry, Real) for entry in value
        ):
            raise ValueError("tool_quaternion_wxyz must contain four real values")
        quaternion = tuple(float(entry) for entry in value)
        if not np.all(np.isfinite(quaternion)) or not np.isclose(
            np.linalg.norm(quaternion), 1.0, rtol=0, atol=1e-8
        ):
            raise ValueError("tool_quaternion_wxyz must be finite and unit length")
        return quaternion


class WaypointBuilder:
    """Add explicit vertical geometry and orientation to one processed path."""

    def __init__(
        self,
        settings: Union[WaypointConstructionSettings, Mapping[str, object]],
    ) -> None:
        if settings is None:
            raise ValueError("Explicit waypoint-construction settings are required")
        self.settings = WaypointConstructionSettings.model_validate(settings)

    def build(self, path: ProcessedPath) -> PickPlaceWaypoints:
        settings = self.settings
        start = path.xy_m[0]
        goal = path.xy_m[-1]

        def pose(xy, z_m):
            return ToolPose(
                (float(xy[0]), float(xy[1]), z_m),
                settings.tool_quaternion_wxyz,
            )

        return PickPlaceWaypoints(
            approach=pose(start, settings.approach_z_m),
            grasp=pose(start, settings.grasp_z_m),
            lift=pose(start, settings.lift_z_m),
            path=tuple(pose(xy, settings.transport_z_m) for xy in path.xy_m),
            lower=pose(goal, settings.lower_z_m),
            retreat=pose(goal, settings.retreat_z_m),
            goal_position=(goal[0], goal[1], settings.object_goal_z_m),
        )


def build_waypoints(
    path: ProcessedPath,
    settings: Union[WaypointConstructionSettings, Mapping[str, object]],
) -> PickPlaceWaypoints:
    """Convenience wrapper for one explicit waypoint-construction operation."""
    return WaypointBuilder(settings).build(path)
