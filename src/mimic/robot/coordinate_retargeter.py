"""Explicit, robot-independent, metric-preserving XY frame mapping."""

from typing import Literal, Mapping, Tuple, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, StrictFloat, field_validator, model_validator

from mimic.common.types import ExtractedTask, RetargetedTask

# Numerical representation tolerance for unit axes, not a task success tolerance.
_AXIS_ATOL = 1e-10
_XY = Tuple[StrictFloat, StrictFloat]


class MappingConfig(BaseModel):
    """Required deployment values; no robot, placement, or orientation defaults.

    Axis vectors express one meter of source-axis displacement in target XY.
    Both must be unit length and perpendicular. Reflections are valid in this
    2D mapping; arbitrary resizing/shearing is not. Input lists become immutable
    tuples, but numeric strings, unknown fields and nonfinite values are rejected.
    """

    model_config = ConfigDict(
        frozen=True, extra="forbid", allow_inf_nan=False, revalidate_instances="always"
    )

    source_frame: Literal["table"]
    target_frame: str
    table_origin_target_xy_m: _XY
    table_x_axis_target_xy: _XY
    table_y_axis_target_xy: _XY

    @field_validator("target_frame")
    @classmethod
    def validate_target_frame(cls, value: str) -> str:
        if not value.strip() or value != value.strip() or value == "table":
            raise ValueError("target_frame must be a distinct, nonempty name without edge spaces")
        return value

    @model_validator(mode="after")
    def validate_axes(self) -> "MappingConfig":
        axes = np.column_stack((self.table_x_axis_target_xy, self.table_y_axis_target_xy))
        if not np.allclose(axes.T @ axes, np.eye(2), rtol=0, atol=_AXIS_ATOL):
            raise ValueError(
                "Table axes must be unit length and perpendicular; "
                "no automatic normalization, scaling, or shear"
            )
        return self


class CoordinateRetargeter:
    """Map table meters to configured target-frame meters, preserving source."""

    def __init__(self, mapping_config: Union[MappingConfig, Mapping[str, object]]):
        if mapping_config is None:
            raise ValueError("Explicit mapping_config is required; no default mapping exists")
        self.mapping_config = MappingConfig.model_validate(mapping_config)

    def retarget(self, task: ExtractedTask) -> RetargetedTask:
        config = self.mapping_config
        if task.coordinate_frame != config.source_frame:
            raise ValueError("Task source frame does not match mapping configuration")
        axes = np.column_stack((config.table_x_axis_target_xy, config.table_y_axis_target_xy))
        source_m: np.ndarray = np.asarray(
            [s.table_xy_m for s in task.demonstrated_path], dtype=float
        )
        # Row-vector form of p_target = origin_target + axes @ p_table_m.
        with np.errstate(over="raise", invalid="raise"):
            try:
                target_m = source_m @ axes.T + config.table_origin_target_xy_m
            except FloatingPointError as exc:
                raise ValueError("Mapping produced nonfinite coordinates") from exc
        return RetargetedTask(
            source_task=task,
            target_frame=config.target_frame,
            demonstrated_path_xy_m=tuple((float(point[0]), float(point[1])) for point in target_m),
        )


def retarget_task(
    task: ExtractedTask, mapping_config: Union[MappingConfig, Mapping[str, object]]
) -> RetargetedTask:
    """Retarget every retained sample without selecting or processing a path."""
    return CoordinateRetargeter(mapping_config).retarget(task)
