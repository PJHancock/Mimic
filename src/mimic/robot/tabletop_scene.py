"""Build the configured physical tabletop footprint in a MuJoCo world."""

from numbers import Real
from typing import Literal, Mapping, Tuple, Union

import mujoco
import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator


class TabletopCloneSettings(BaseModel):
    """Minimal left-edge table placement; dimensions and positions use meters."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", allow_inf_nan=False, revalidate_instances="always"
    )

    width_m: float
    depth_m: float
    thickness_m: float
    surface_z_m: float
    robot_edge: Literal["left"]
    robot_base_xy_m: Tuple[float, float]
    robot_setback_m: float

    @field_validator("width_m", "depth_m", "thickness_m", mode="before")
    @classmethod
    def validate_positive_length(cls, value):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError("Table dimensions must be real meters")
        result = float(value)
        if not np.isfinite(result) or result <= 0:
            raise ValueError("Table dimensions must be finite and positive")
        return result

    @field_validator("surface_z_m", mode="before")
    @classmethod
    def validate_surface_z(cls, value):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError("Table surface Z must be real meters")
        result = float(value)
        if not np.isfinite(result):
            raise ValueError("Table surface Z must be finite meters")
        return result

    @field_validator("robot_setback_m", mode="before")
    @classmethod
    def validate_robot_setback(cls, value):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError("Robot setback must be real meters")
        result = float(value)
        if not np.isfinite(result) or result < 0:
            raise ValueError("Robot setback must be finite and nonnegative")
        return result

    @field_validator("robot_base_xy_m", mode="before")
    @classmethod
    def validate_robot_base(cls, value):
        if np.shape(value) != (2,) or any(
            isinstance(entry, (bool, np.bool_)) or not isinstance(entry, Real) for entry in value
        ):
            raise ValueError("Robot base XY must contain two real meters")
        result = tuple(float(entry) for entry in value)
        if not np.all(np.isfinite(result)):
            raise ValueError("Robot base XY must contain two finite meters")
        return result


def add_tabletop_clone(
    spec: mujoco.MjSpec,
    settings: Union[TabletopCloneSettings, Mapping[str, object]],
) -> TabletopCloneSettings:
    """Add a bounded tabletop in front of a separately marked robot base."""

    config = TabletopCloneSettings.model_validate(settings)
    base_x, base_y = config.robot_base_xy_m
    near_edge_x = base_x + config.robot_setback_m
    spec.worldbody.add_geom(
        name="tabletop_clone",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[
            near_edge_x + config.width_m / 2,
            base_y,
            config.surface_z_m - config.thickness_m / 2,
        ],
        size=[config.width_m / 2, config.depth_m / 2, config.thickness_m / 2],
    )
    spec.worldbody.add_site(
        name="robot_base_frame",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        pos=[base_x, base_y, config.surface_z_m],
        size=[0.005, 0.005, 0.005],
        rgba=[1.0, 0.0, 0.0, 1.0],
    )
    return config
