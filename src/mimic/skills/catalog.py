"""Versioned bijection between training labels and robot skill handlers."""

import hashlib
import json
from typing import Mapping, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class SkillSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    id: str
    training_index: int
    handler_id: str

    @field_validator("id", "handler_id")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("Skill identifiers must be nonempty without edge spaces")
        return value

    @field_validator("training_index")
    @classmethod
    def validate_index(cls, value: int) -> int:
        if isinstance(value, bool) or value < 0:
            raise ValueError("training_index must be a nonnegative integer")
        return value


class SkillCatalog(BaseModel):
    """One deployment's ordered classifier vocabulary and handler bindings."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: int
    skills: Tuple[SkillSpec, ...]

    @field_validator("schema_version")
    @classmethod
    def validate_version(cls, value: int) -> int:
        if isinstance(value, bool) or value <= 0:
            raise ValueError("schema_version must be a positive integer")
        return value

    @model_validator(mode="after")
    def validate_bijection(self) -> "SkillCatalog":
        if not self.skills:
            raise ValueError("A skill catalog must not be empty")
        ids = tuple(skill.id for skill in self.skills)
        handlers = tuple(skill.handler_id for skill in self.skills)
        indices = tuple(skill.training_index for skill in self.skills)
        if len(set(ids)) != len(ids):
            raise ValueError("Skill IDs must be unique")
        if len(set(handlers)) != len(handlers):
            raise ValueError("Each active skill requires one unique handler binding")
        if set(indices) != set(range(len(indices))):
            raise ValueError("Training indices must be unique and contiguous from zero")
        return self

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "SkillCatalog":
        return cls.model_validate(value)

    @property
    def ordered_skills(self) -> Tuple[SkillSpec, ...]:
        return tuple(sorted(self.skills, key=lambda skill: skill.training_index))

    @property
    def labels(self) -> Tuple[str, ...]:
        return tuple(skill.id for skill in self.ordered_skills)

    @property
    def class_count(self) -> int:
        return len(self.skills)

    @property
    def fingerprint(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "skills": [skill.model_dump(mode="json") for skill in self.ordered_skills],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def index_for(self, skill_id: str) -> int:
        try:
            return next(skill.training_index for skill in self.skills if skill.id == skill_id)
        except StopIteration as exc:
            raise KeyError(f"Unknown skill ID: {skill_id}") from exc

    def label_for(self, training_index: int) -> str:
        try:
            return next(skill.id for skill in self.skills if skill.training_index == training_index)
        except StopIteration as exc:
            raise KeyError(f"Unknown training index: {training_index}") from exc

    def require_handlers(self, handler_ids: Sequence[str]) -> None:
        configured = set(handler_ids)
        required = {skill.handler_id for skill in self.skills}
        if configured != required:
            missing = sorted(required - configured)
            extra = sorted(configured - required)
            raise ValueError(f"Handler/catalog mismatch; missing={missing}, extra={extra}")
