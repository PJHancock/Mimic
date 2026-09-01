"""Canonical post-model artifact for deterministic robot task construction."""

from __future__ import annotations

import json
from datetime import datetime
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mimic.common.types import ActionPrediction

from .action_results import (
    CatalogProvenance,
    PostprocessingProvenance,
    RobotActionFrame,
    RobotActionResults,
    load_robot_actions,
)

DEMO_TASK_INPUT_SCHEMA = "mimic.demo_task_input.v1"
_JSON_SOURCE = Union[str, Path, Mapping[str, Any], "DemoTaskInput"]


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _zero_based_frame(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError("tracker frame must be a nonnegative zero-based integer")
    return int(value)


def _finite_nonnegative(value: object, name: str) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


class DemoVideoMetadata(BaseModel):
    """Source-video and tracker coordinate metadata stored once per demo."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    created_at: datetime
    fps: float
    frame_count: int
    duration_s: float
    tracking_coordinate_frame: Literal["image_pixels"] = "image_pixels"
    image_width_px: int
    image_height_px: int

    @field_validator("fps", mode="before")
    @classmethod
    def validate_fps(cls, value: object) -> float:
        result = _finite_nonnegative(value, "fps")
        if result == 0:
            raise ValueError("fps must be positive")
        return result

    @field_validator("duration_s", mode="before")
    @classmethod
    def validate_duration(cls, value: object) -> float:
        return _finite_nonnegative(value, "duration_s")

    @field_validator("frame_count", "image_width_px", "image_height_px", mode="before")
    @classmethod
    def validate_positive_integer(cls, value: object, info: object) -> int:
        return _positive_integer(value, info.field_name)


class ImagePixelPosition(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    x: float
    y: float
    confidence: float

    @field_validator("x", "y", mode="before")
    @classmethod
    def validate_coordinate(cls, value: object, info: object) -> float:
        return _finite_nonnegative(value, info.field_name)

    @field_validator("confidence", mode="before")
    @classmethod
    def validate_confidence(cls, value: object) -> float:
        result = _finite_nonnegative(value, "confidence")
        if result > 1:
            raise ValueError("confidence must be a probability")
        return result


class ImageObjectTrackFrame(BaseModel):
    """One tracker-native sample; a null position means no observation."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    frame_idx: int
    timestamp_s: float
    position: Optional[ImagePixelPosition]

    @field_validator("frame_idx", mode="before")
    @classmethod
    def validate_frame_idx(cls, value: object) -> int:
        return _positive_integer(value, "frame_idx")

    @field_validator("timestamp_s", mode="before")
    @classmethod
    def validate_timestamp(cls, value: object) -> float:
        return _finite_nonnegative(value, "timestamp_s")


class DemoTaskInput(BaseModel):
    """One self-contained post-model input for deterministic robot processing.

    ``resolved_actions`` deliberately contains exactly one accepted state per
    classifier timestep. ``object_tracks`` remains independently sampled so the
    robot boundary does not assume one classifier prediction per video frame.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
        populate_by_name=True,
        serialize_by_alias=True,
    )

    schema_name: Literal["mimic.demo_task_input.v1"] = Field(
        default=DEMO_TASK_INPUT_SCHEMA, alias="schema"
    )
    video: DemoVideoMetadata
    catalog: CatalogProvenance
    checkpoint_sha256: str
    postprocessing: PostprocessingProvenance
    resolved_actions: Tuple[RobotActionFrame, ...]
    object_tracks: Tuple[ImageObjectTrackFrame, ...]

    @field_validator("checkpoint_sha256")
    @classmethod
    def validate_checkpoint_sha256(cls, value: str) -> str:
        return RobotActionResults.validate_checkpoint_sha256(value)

    @model_validator(mode="after")
    def validate_streams(self) -> "DemoTaskInput":
        # Reuse the action-only contract so its one-state, catalog, frame, and
        # timestamp invariants cannot diverge from this consolidated artifact.
        self.as_robot_actions()
        _validate_ordered_frames(self.object_tracks, "object track")
        max_frame = max(
            self.resolved_actions[-1].frame_idx,
            self.object_tracks[-1].frame_idx,
        )
        if max_frame > self.video.frame_count:
            raise ValueError("stream frame_idx exceeds video.frame_count")
        max_timestamp = max(
            self.resolved_actions[-1].timestamp_s,
            self.object_tracks[-1].timestamp_s,
        )
        if max_timestamp > self.video.duration_s + 1e-9:
            raise ValueError("stream timestamp exceeds video.duration_s")
        for track in self.object_tracks:
            if track.position is None:
                continue
            if track.position.x >= self.video.image_width_px:
                raise ValueError("tracking x is outside the declared image frame")
            if track.position.y >= self.video.image_height_px:
                raise ValueError("tracking y is outside the declared image frame")
        return self

    def as_robot_actions(self) -> RobotActionResults:
        """Return the deliberately narrow, one-state-per-timestep adapter view."""

        return RobotActionResults(
            catalog=self.catalog,
            checkpoint_sha256=self.checkpoint_sha256,
            postprocessing=self.postprocessing,
            frames=self.resolved_actions,
        )


def _validate_ordered_frames(
    frames: Sequence[ImageObjectTrackFrame], description: str
) -> None:
    if not frames:
        raise ValueError(f"{description} frames must not be empty")
    previous_frame = 0
    previous_timestamp = -1.0
    for frame in frames:
        if frame.frame_idx <= previous_frame:
            raise ValueError(f"{description} frame_idx values must be unique and increasing")
        if frame.timestamp_s <= previous_timestamp:
            raise ValueError(f"{description} timestamps must be strictly increasing")
        previous_frame = frame.frame_idx
        previous_timestamp = frame.timestamp_s


def build_demo_task_input(
    tracks_data: Mapping[str, Any],
    actions: RobotActionResults,
    *,
    created_at: Optional[datetime] = None,
) -> DemoTaskInput:
    """Combine post-processed actions and tracker-native observations without joining rows."""

    positions = tracks_data.get("positions")
    if not isinstance(positions, Sequence) or isinstance(positions, (str, bytes)):
        raise ValueError("tracks_data.positions must be a sequence")
    object_tracks = []
    for sample in positions:
        if not isinstance(sample, Mapping):
            raise ValueError("Each tracker sample must be a mapping")
        x, y = sample.get("x"), sample.get("y")
        if x is None and y is None:
            position = None
        elif x is None or y is None:
            raise ValueError("Tracker x and y must both be numeric or both be null")
        else:
            position = ImagePixelPosition(
                x=x,
                y=y,
                confidence=sample.get("confidence"),
            )
        object_tracks.append(
            ImageObjectTrackFrame(
                frame_idx=_zero_based_frame(sample.get("frame")) + 1,
                timestamp_s=sample.get("time"),
                position=position,
            )
        )

    video = DemoVideoMetadata(
        created_at=created_at or datetime.now().astimezone(),
        fps=tracks_data.get("fps"),
        frame_count=tracks_data.get("frame_count"),
        duration_s=tracks_data.get("duration"),
        image_width_px=tracks_data.get("frame_width_px"),
        image_height_px=tracks_data.get("frame_height_px"),
    )
    return DemoTaskInput(
        video=video,
        catalog=actions.catalog,
        checkpoint_sha256=actions.checkpoint_sha256,
        postprocessing=actions.postprocessing,
        resolved_actions=actions.frames,
        object_tracks=tuple(object_tracks),
    )


def load_demo_task_input(source: _JSON_SOURCE) -> DemoTaskInput:
    if isinstance(source, DemoTaskInput):
        return source
    if isinstance(source, Mapping):
        payload = dict(source)
    else:
        payload = json.loads(Path(source).read_text())
    if payload.get("schema") != DEMO_TASK_INPUT_SCHEMA:
        raise ValueError(f"Robot task input requires schema {DEMO_TASK_INPUT_SCHEMA}")
    return DemoTaskInput.model_validate(payload)


def load_task_actions(
    source: _JSON_SOURCE,
    *,
    expected_catalog_fingerprint: Optional[str] = None,
) -> Tuple[ActionPrediction, ...]:
    """Expose only resolved actions from the consolidated artifact to robot code."""

    task_input = load_demo_task_input(source)
    return load_robot_actions(
        task_input.as_robot_actions(),
        expected_catalog_fingerprint=expected_catalog_fingerprint,
    )
