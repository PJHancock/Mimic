"""Robot-independent selection and interpolation of retargeted XY paths."""

from dataclasses import dataclass
from enum import Enum
from numbers import Real
from typing import Mapping, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from scipy.interpolate import CubicSpline

from mimic.common.types import RetargetedTask

_XY = Tuple[float, float]
_NUMERICAL_ATOL_M = 1e-12


class PathInterpolation(str, Enum):
    """Path geometry policy; values intentionally match YAML configuration."""

    DIRECT = "direct"
    CORNERS_ONLY = "corners_only"
    NONE = "none"
    CUBIC = "cubic"


class PathProcessingSettings(BaseModel):
    """Mode-specific geometry settings in target-frame meters.

    Numerical values are required only when they affect the selected mode. This
    prevents stale settings from appearing to affect direct or exact following.
    """

    model_config = ConfigDict(
        frozen=True, extra="forbid", allow_inf_nan=False, revalidate_instances="always"
    )

    interpolation: PathInterpolation
    corner_max_deviation_m: Optional[float] = None
    output_spacing_m: Optional[float] = None
    maximum_spline_deviation_m: Optional[float] = None

    @field_validator(
        "corner_max_deviation_m", "output_spacing_m", "maximum_spline_deviation_m", mode="before"
    )
    @classmethod
    def validate_positive_distance(cls, value):
        if value is None:
            return None
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError("Path-processing distances must be real numbers")
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Path-processing distances must be positive and finite")
        return float(value)

    @model_validator(mode="after")
    def validate_mode_settings(self) -> "PathProcessingSettings":
        configured = {
            "corner_max_deviation_m": self.corner_max_deviation_m,
            "output_spacing_m": self.output_spacing_m,
            "maximum_spline_deviation_m": self.maximum_spline_deviation_m,
        }
        if self.interpolation in (PathInterpolation.DIRECT, PathInterpolation.NONE):
            if any(value is not None for value in configured.values()):
                raise ValueError(f"{self.interpolation.value} does not accept geometry thresholds")
        elif self.interpolation == PathInterpolation.CORNERS_ONLY:
            if self.corner_max_deviation_m is None:
                raise ValueError("corners_only requires corner_max_deviation_m")
            if self.output_spacing_m is not None or self.maximum_spline_deviation_m is not None:
                raise ValueError("corners_only accepts only corner_max_deviation_m")
        elif any(value is None for value in configured.values()):
            raise ValueError(
                "cubic requires corner_max_deviation_m, output_spacing_m, "
                "and maximum_spline_deviation_m"
            )
        return self


@dataclass(frozen=True)
class ProcessedPath:
    """Selected XY geometry plus traceability to retained source samples."""

    source_task: RetargetedTask
    interpolation: PathInterpolation
    xy_m: Tuple[_XY, ...]
    control_points_xy_m: Tuple[_XY, ...]
    control_point_source_indices: Tuple[int, ...]

    def __post_init__(self):
        points = _finite_points(self.xy_m, "Processed path")
        controls = _finite_points(self.control_points_xy_m, "Control points")
        indices = tuple(self.control_point_source_indices)
        if len(controls) != len(indices):
            raise ValueError("Each control point requires one source-path index")
        source_count = len(self.source_task.demonstrated_path_xy_m)
        if any(
            isinstance(index, (bool, np.bool_))
            or not isinstance(index, (int, np.integer))
            or index < 0
            or index >= source_count
            for index in indices
        ):
            raise ValueError("Control-point indices must refer to retained source-path samples")
        if any(a >= b for a, b in zip(indices, indices[1:])):
            raise ValueError("Control-point source indices must be strictly increasing")
        if points[0] != self.source_task.start_xy_m or points[-1] != self.source_task.goal_xy_m:
            raise ValueError("A processed path must preserve the exact task endpoints")
        object.__setattr__(self, "interpolation", PathInterpolation(self.interpolation))
        object.__setattr__(self, "xy_m", points)
        object.__setattr__(self, "control_points_xy_m", controls)
        object.__setattr__(self, "control_point_source_indices", indices)


def _finite_points(points, label: str) -> Tuple[_XY, ...]:
    result = tuple(tuple(point) for point in points)
    if not result or any(
        np.shape(point) != (2,) or not np.all(np.isfinite(point)) for point in result
    ):
        raise ValueError(f"{label} must contain finite XY coordinates")
    return tuple((float(point[0]), float(point[1])) for point in result)


def _point_to_segment_distances(points: np.ndarray, start: np.ndarray, end: np.ndarray):
    segment = end - start
    squared_length = float(segment @ segment)
    if squared_length == 0:
        return np.linalg.norm(points - start, axis=1)
    fractions = np.clip(((points - start) @ segment) / squared_length, 0.0, 1.0)
    projections = start + fractions[:, None] * segment
    return np.linalg.norm(points - projections, axis=1)


def _corner_indices(points: np.ndarray, tolerance_m: float) -> Tuple[int, ...]:
    """Return deterministic Ramer-Douglas-Peucker source indices."""
    if len(points) == 2:
        return (0, 1)
    retained = {0, len(points) - 1}
    pending = [(0, len(points) - 1)]
    while pending:
        start, end = pending.pop()
        if end - start <= 1:
            continue
        interior = points[slice(start + 1, end)]
        distances = _point_to_segment_distances(interior, points[start], points[end])
        relative = int(np.argmax(distances))
        furthest = start + 1 + relative
        if distances[relative] > tolerance_m:
            retained.add(furthest)
            pending.extend(((start, furthest), (furthest, end)))
    return tuple(sorted(retained))


def _without_consecutive_duplicates(
    points: np.ndarray, indices: Tuple[int, ...]
) -> Tuple[np.ndarray, Tuple[int, ...]]:
    retained = [indices[0]]
    for index in indices[1:]:
        if not np.array_equal(points[index], points[retained[-1]]):
            retained.append(index)
        elif index == len(points) - 1:
            retained[-1] = index
    return points[retained], tuple(retained)


def _minimum_distances_to_polyline(query: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    distances = np.full(len(query), np.inf)
    for start, end in zip(polyline, polyline[1:]):
        distances = np.minimum(distances, _point_to_segment_distances(query, start, end))
    return distances


class PathProcessor:
    """Apply one explicit path policy without altering its retargeted task."""

    def __init__(self, settings: Union[PathProcessingSettings, Mapping[str, object]]) -> None:
        if settings is None:
            raise ValueError("Explicit path-processing settings are required")
        self.settings = PathProcessingSettings.model_validate(settings)

    def process(self, task: RetargetedTask) -> ProcessedPath:
        source = np.asarray(task.demonstrated_path_xy_m, dtype=float)
        last = len(source) - 1
        mode = self.settings.interpolation

        if mode == PathInterpolation.DIRECT:
            return self._result(task, source[[0, last]], (0, last), source[[0, last]])
        if mode == PathInterpolation.NONE:
            indices = tuple(range(len(source)))
            return self._result(task, source, indices, source)

        corner_tolerance = self.settings.corner_max_deviation_m
        if corner_tolerance is None:  # Defensive check after settings validation.
            raise RuntimeError("Validated corner processing is missing its tolerance")
        corner_indices = _corner_indices(source, corner_tolerance)
        corners = source[list(corner_indices)]
        if mode == PathInterpolation.CORNERS_ONLY:
            return self._result(task, corners, corner_indices, corners)
        return self._cubic(task, source, corner_indices)

    def _cubic(
        self,
        task: RetargetedTask,
        source: np.ndarray,
        corner_indices: Tuple[int, ...],
    ) -> ProcessedPath:
        controls, control_indices = _without_consecutive_duplicates(source, corner_indices)
        if len(controls) < 2:
            raise ValueError("cubic requires at least two distinct control points")
        output_spacing = self.settings.output_spacing_m
        maximum_allowed_deviation = self.settings.maximum_spline_deviation_m
        if output_spacing is None or maximum_allowed_deviation is None:
            raise RuntimeError("Validated cubic processing is missing required distances")

        chord_lengths = np.linalg.norm(np.diff(controls, axis=0), axis=1)
        parameters = np.concatenate(([0.0], np.cumsum(chord_lengths)))
        total_chord_length = parameters[-1]
        parameters /= total_chord_length
        spline = CubicSpline(parameters, controls, axis=0, bc_type="natural", extrapolate=False)

        # Twenty samples per requested output interval plus a fixed minimum make
        # arc-length resampling and deviation checks independent of video rate.
        output_intervals = max(1, int(np.ceil(total_chord_length / output_spacing)))
        dense_count = max(1001, output_intervals * 20 + 1, (len(controls) - 1) * 50 + 1)
        dense_parameters = np.linspace(0.0, 1.0, dense_count)
        dense_points = spline(dense_parameters)
        dense_lengths = np.concatenate(
            ([0.0], np.cumsum(np.linalg.norm(np.diff(dense_points, axis=0), axis=1)))
        )
        curve_length = dense_lengths[-1]
        if not np.isfinite(curve_length) or curve_length <= 0:
            raise ValueError("cubic produced a zero-length or nonfinite path")

        maximum_deviation = float(np.max(_minimum_distances_to_polyline(dense_points, source)))
        if maximum_deviation > maximum_allowed_deviation + _NUMERICAL_ATOL_M:
            raise ValueError(
                "cubic exceeds maximum_spline_deviation_m " f"({maximum_deviation:.9g} m observed)"
            )

        sample_count = max(2, int(np.ceil(curve_length / output_spacing)) + 1)
        target_lengths = np.linspace(0.0, curve_length, sample_count)
        sample_parameters = np.interp(target_lengths, dense_lengths, dense_parameters)
        sampled = spline(sample_parameters)
        sampled[0] = source[0]
        sampled[-1] = source[-1]
        return self._result(task, sampled, control_indices, controls)

    def _result(
        self,
        task: RetargetedTask,
        points: np.ndarray,
        control_indices: Tuple[int, ...],
        controls: np.ndarray,
    ) -> ProcessedPath:
        def to_tuples(values):
            return tuple((float(x), float(y)) for x, y in values)

        return ProcessedPath(
            source_task=task,
            interpolation=self.settings.interpolation,
            xy_m=to_tuples(points),
            control_points_xy_m=to_tuples(controls),
            control_point_source_indices=control_indices,
        )


def process_path(
    task: RetargetedTask,
    settings: Union[PathProcessingSettings, Mapping[str, object]],
) -> ProcessedPath:
    """Convenience wrapper for one explicitly configured processing operation."""
    return PathProcessor(settings).process(task)
