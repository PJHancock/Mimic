"""Validated classifier-score and single-state robot action artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mimic.common.types import ActionPhase, ActionPrediction
from mimic.skills.catalog import SkillCatalog
from mimic.skills.graph import SkillGraph, SkillTransition
from mimic.skills.post_state import GraphStatePostProcessor, PostStateSettings
from mimic.skills.types import DecisionSource, SkillPrediction, StateDecision

SCORE_RESULTS_SCHEMA = "mimic.skill_scores.v2"
ROBOT_ACTIONS_SCHEMA = "mimic.robot_actions.v1"
FAIL_CLOSED_GUARD_POLICY = "fail_closed_without_runtime_observations"
DEFER_RUNTIME_GUARD_POLICY = "defer_runtime_guards_to_execution"
GuardPolicy = Literal[
    "fail_closed_without_runtime_observations",
    "defer_runtime_guards_to_execution",
]


def _defer_runtime_guard(transition: SkillTransition) -> bool:
    return transition.guard is None or transition.guard_scope == "runtime"


def _frame_idx(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError("frame_idx must be a positive one-based integer")
    return int(value)


def _timestamp_s(value: float) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or value < 0
    ):
        raise ValueError("timestamp_s must be finite nonnegative seconds")
    return float(value)


@dataclass(frozen=True)
class SkillSystemDefinition:
    """Loaded catalog/graph plus deliberately unresolved-or-explicit settings."""

    catalog: SkillCatalog
    graph: SkillGraph
    post_state: Mapping[str, object]

    def build_postprocessor(
        self, guard_policy: GuardPolicy = FAIL_CLOSED_GUARD_POLICY
    ) -> GraphStatePostProcessor:
        unresolved = sorted(key for key, value in self.post_state.items() if value is None)
        if unresolved:
            raise ValueError(
                "Post-state settings are unresolved: "
                f"{unresolved}. Supply an experiment skill config with explicit values."
            )
        if guard_policy == FAIL_CLOSED_GUARD_POLICY:
            transition_guard = None
        elif guard_policy == DEFER_RUNTIME_GUARD_POLICY:
            transition_guard = _defer_runtime_guard
        else:
            raise ValueError(f"Unknown transition guard policy: {guard_policy}")
        return GraphStatePostProcessor(
            self.catalog,
            self.graph,
            self.post_state,
            transition_guard=transition_guard,
        )


class CatalogProvenance(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    schema_version: int
    fingerprint: str
    labels: Tuple[str, ...]

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version(cls, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError("catalog schema_version must be a positive integer")
        return int(value)

    @field_validator("fingerprint")
    @classmethod
    def validate_fingerprint(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("catalog fingerprint must be a lowercase SHA-256 digest")
        return value

    @field_validator("labels")
    @classmethod
    def validate_labels(cls, value: Tuple[str, ...]) -> Tuple[str, ...]:
        if not value or len(set(value)) != len(value):
            raise ValueError("catalog labels must be nonempty and unique")
        return value


class SkillScoreFrame(BaseModel):
    """Classifier evidence retained before graph-aware post-processing."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    frame_idx: int
    timestamp_s: float
    detection_valid: bool
    state_scores: Dict[str, float]

    @field_validator("frame_idx", mode="before")
    @classmethod
    def validate_frame_idx(cls, value: int) -> int:
        return _frame_idx(value)

    @field_validator("timestamp_s", mode="before")
    @classmethod
    def validate_timestamp(cls, value: float) -> float:
        return _timestamp_s(value)

    @field_validator("detection_valid", mode="before")
    @classmethod
    def validate_detection_valid(cls, value: bool) -> bool:
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError("detection_valid must be boolean")
        return bool(value)

    @field_validator("state_scores", mode="before")
    @classmethod
    def validate_raw_scores(cls, value: Mapping[str, float]) -> Dict[str, float]:
        if not isinstance(value, Mapping):
            raise ValueError("state_scores must be a label-to-probability mapping")
        for score in value.values():
            if isinstance(score, (bool, np.bool_)) or not isinstance(score, Real):
                raise ValueError("state_scores values must be numeric probabilities")
        return dict(value)

    @model_validator(mode="after")
    def validate_prediction(self) -> "SkillScoreFrame":
        SkillPrediction(
            frame_idx=self.frame_idx,
            timestamp_s=self.timestamp_s,
            detection_valid=self.detection_valid,
            state_scores=self.state_scores,
        )
        return self


class RobotActionFrame(BaseModel):
    """Exactly one accepted state at one classifier timestep."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    frame_idx: int
    timestamp_s: float
    phase: ActionPhase
    confidence: Optional[float]
    decision_source: DecisionSource

    @field_validator("frame_idx", mode="before")
    @classmethod
    def validate_frame_idx(cls, value: int) -> int:
        return _frame_idx(value)

    @field_validator("timestamp_s", mode="before")
    @classmethod
    def validate_timestamp(cls, value: float) -> float:
        return _timestamp_s(value)

    @field_validator("confidence", mode="before")
    @classmethod
    def validate_confidence(cls, value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Real)
            or not np.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError("confidence must be a finite probability or null")
        return float(value)


class PostprocessingProvenance(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    fingerprint: str
    settings: PostStateSettings
    guard_policy: GuardPolicy

    @field_validator("fingerprint")
    @classmethod
    def validate_fingerprint(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("postprocessing fingerprint must be a lowercase SHA-256 digest")
        return value


def _validate_frame_sequence(frames: Sequence[Union[SkillScoreFrame, RobotActionFrame]]) -> None:
    if not frames:
        raise ValueError("result frames must not be empty")
    previous_frame = 0
    previous_time = -1.0
    for frame in frames:
        if frame.frame_idx <= previous_frame:
            raise ValueError("result frame_idx values must be unique and strictly increasing")
        if frame.timestamp_s <= previous_time:
            raise ValueError("result timestamps must be strictly increasing")
        previous_frame = frame.frame_idx
        previous_time = frame.timestamp_s


class SkillScoreResults(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
        populate_by_name=True,
        serialize_by_alias=True,
    )

    schema_name: Literal["mimic.skill_scores.v2"] = Field(
        default=SCORE_RESULTS_SCHEMA, alias="schema"
    )
    catalog: CatalogProvenance
    checkpoint_sha256: str
    frames: Tuple[SkillScoreFrame, ...]

    @field_validator("checkpoint_sha256")
    @classmethod
    def validate_checkpoint_sha256(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("checkpoint_sha256 must be a lowercase SHA-256 digest")
        return value

    @model_validator(mode="after")
    def validate_frames(self) -> "SkillScoreResults":
        _validate_frame_sequence(self.frames)
        expected = set(self.catalog.labels)
        for frame in self.frames:
            if frame.detection_valid and set(frame.state_scores) != expected:
                missing = sorted(expected - set(frame.state_scores))
                extra = sorted(set(frame.state_scores) - expected)
                raise ValueError(f"Prediction/catalog mismatch; missing={missing}, extra={extra}")
        return self


class RobotActionResults(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
        populate_by_name=True,
        serialize_by_alias=True,
    )

    schema_name: Literal["mimic.robot_actions.v1"] = Field(
        default=ROBOT_ACTIONS_SCHEMA, alias="schema"
    )
    catalog: CatalogProvenance
    checkpoint_sha256: str
    postprocessing: PostprocessingProvenance
    frames: Tuple[RobotActionFrame, ...]

    @field_validator("checkpoint_sha256")
    @classmethod
    def validate_checkpoint_sha256(cls, value: str) -> str:
        return SkillScoreResults.validate_checkpoint_sha256(value)

    @model_validator(mode="after")
    def validate_frames(self) -> "RobotActionResults":
        _validate_frame_sequence(self.frames)
        if any(frame.phase.value not in self.catalog.labels for frame in self.frames):
            raise ValueError("Robot action phase is absent from the active catalog")
        return self


@dataclass(frozen=True)
class ActionInferenceArtifacts:
    scores: SkillScoreResults
    robot_actions: RobotActionResults
    decisions: Tuple[StateDecision, ...]


def load_skill_system(path: Union[str, Path]) -> SkillSystemDefinition:
    """Load the versioned catalog, graph, and post-state settings from YAML."""
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text())
    if not isinstance(payload, Mapping) or set(payload) != {"skill_system"}:
        raise ValueError("Skill config must contain exactly the skill_system root")
    system = payload["skill_system"]
    if not isinstance(system, Mapping) or set(system) != {"catalog", "graph", "post_state"}:
        raise ValueError("skill_system must contain exactly catalog, graph, and post_state")
    catalog = SkillCatalog.from_mapping(system["catalog"])
    graph = SkillGraph.from_mapping(catalog, system["graph"])
    post_state = system["post_state"]
    if not isinstance(post_state, Mapping):
        raise ValueError("post_state must be a mapping")
    return SkillSystemDefinition(catalog, graph, dict(post_state))


def catalog_provenance(catalog: SkillCatalog) -> CatalogProvenance:
    return CatalogProvenance(
        schema_version=catalog.schema_version,
        fingerprint=catalog.fingerprint,
        labels=catalog.labels,
    )


def checkpoint_sha256(path: Union[str, Path]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def predictions_from_probabilities(
    probabilities: np.ndarray,
    timestamps_s: Sequence[float],
    catalog: SkillCatalog,
    *,
    frame_indices: Optional[Sequence[int]] = None,
    detection_valid: Optional[Sequence[bool]] = None,
) -> Tuple[SkillPrediction, ...]:
    """Map classifier columns to catalog labels without taking argmax."""
    values = np.asarray(probabilities)
    if values.ndim != 2 or values.shape[1] != catalog.class_count:
        raise ValueError(
            "Classifier probabilities must have shape "
            f"(frames, {catalog.class_count}) for the active catalog"
        )
    frame_count = values.shape[0]
    if len(timestamps_s) != frame_count:
        raise ValueError("timestamps and classifier probabilities must have equal length")
    indices = (
        tuple(frame_indices) if frame_indices is not None else tuple(range(1, frame_count + 1))
    )
    valid_flags = (
        tuple(detection_valid)
        if detection_valid is not None
        else tuple(True for _ in range(frame_count))
    )
    if len(indices) != frame_count or len(valid_flags) != frame_count:
        raise ValueError("frame indices and detection flags must match the probability rows")

    predictions = []
    for row_index, row in enumerate(values):
        valid = valid_flags[row_index]
        scores = (
            {label: float(row[catalog.index_for(label)]) for label in catalog.labels}
            if valid
            else {}
        )
        predictions.append(
            SkillPrediction(
                frame_idx=indices[row_index],
                timestamp_s=timestamps_s[row_index],
                detection_valid=valid,
                state_scores=scores,
            )
        )
    return tuple(predictions)


def _postprocessing_provenance(
    system: SkillSystemDefinition,
    settings: PostStateSettings,
    guard_policy: GuardPolicy,
) -> PostprocessingProvenance:
    graph_payload = {
        "start_skill": system.graph.start_skill,
        "transitions": [
            transition.model_dump(mode="json") for transition in system.graph.transitions
        ],
    }
    fingerprint_payload = {
        "catalog_fingerprint": system.catalog.fingerprint,
        "graph": graph_payload,
        "settings": settings.model_dump(mode="json"),
        "guard_policy": guard_policy,
    }
    encoded = json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
    return PostprocessingProvenance(
        fingerprint=hashlib.sha256(encoded).hexdigest(),
        settings=settings,
        guard_policy=guard_policy,
    )


def build_action_inference_artifacts(
    predictions: Iterable[SkillPrediction],
    system: SkillSystemDefinition,
    checkpoint_digest: str,
) -> ActionInferenceArtifacts:
    """Resolve scores and produce a deliberately narrower robot contract."""
    prediction_records = tuple(predictions)
    if not prediction_records:
        raise ValueError("At least one classifier prediction is required")
    # Classifier export has no MuJoCo observations. Runtime-scoped guards are
    # deliberately deferred to execution rather than treated as failed evidence.
    guard_policy: GuardPolicy = DEFER_RUNTIME_GUARD_POLICY
    processor = system.build_postprocessor(guard_policy=guard_policy)
    decisions = tuple(processor.process(prediction) for prediction in prediction_records)
    catalog = catalog_provenance(system.catalog)
    scores = SkillScoreResults(
        catalog=catalog,
        checkpoint_sha256=checkpoint_digest,
        frames=tuple(
            SkillScoreFrame(
                frame_idx=prediction.frame_idx,
                timestamp_s=prediction.timestamp_s,
                detection_valid=prediction.detection_valid,
                state_scores=dict(prediction.state_scores),
            )
            for prediction in prediction_records
        ),
    )
    robot_actions = RobotActionResults(
        catalog=catalog,
        checkpoint_sha256=checkpoint_digest,
        postprocessing=_postprocessing_provenance(system, processor.settings, guard_policy),
        frames=tuple(
            RobotActionFrame(
                frame_idx=prediction.frame_idx,
                timestamp_s=prediction.timestamp_s,
                phase=ActionPhase(decision.accepted_skill),
                confidence=(
                    prediction.state_scores[decision.accepted_skill]
                    if prediction.detection_valid
                    else None
                ),
                decision_source=decision.source,
            )
            for prediction, decision in zip(prediction_records, decisions)
        ),
    )
    return ActionInferenceArtifacts(scores, robot_actions, decisions)


def write_results(path: Union[str, Path], result: BaseModel) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, allow_nan=False) + "\n"
    )


def load_robot_actions(
    source: Union[str, Path, Mapping[str, Any], RobotActionResults],
    *,
    expected_catalog_fingerprint: Optional[str] = None,
) -> Tuple[ActionPrediction, ...]:
    """Load only the post-processed, one-state-per-timestep robot schema."""
    if isinstance(source, RobotActionResults):
        result = source
        payload = result.model_dump(mode="json")
    elif isinstance(source, Mapping):
        payload = dict(source)
    else:
        payload = json.loads(Path(source).read_text())
    if payload.get("schema") != ROBOT_ACTIONS_SCHEMA:
        raise ValueError(
            f"Robot input requires schema {ROBOT_ACTIONS_SCHEMA}; raw scores and legacy "
            "top-one results are not robot inputs"
        )
    if not isinstance(source, RobotActionResults):
        result = RobotActionResults.model_validate(payload)
    if (
        expected_catalog_fingerprint is not None
        and result.catalog.fingerprint != expected_catalog_fingerprint
    ):
        raise ValueError("Robot action catalog fingerprint does not match the active catalog")
    return tuple(
        ActionPrediction(
            frame_idx=frame.frame_idx,
            phase=frame.phase,
            confidence=frame.confidence,
            timestamp=frame.timestamp_s,
        )
        for frame in result.frames
    )
