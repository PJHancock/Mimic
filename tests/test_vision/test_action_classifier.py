"""Tests for complete classifier probabilities and checkpoint provenance."""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from mimic.integration import load_skill_system
from mimic.skills.catalog import SkillCatalog
from mimic.vision.action_classifier import ActionClassifier


class FixedLogits(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.tensor(
            [[1.0, 2.0, 0.0], [3.0, 1.0, 2.0]],
            dtype=inputs.dtype,
            device=inputs.device,
        )[: inputs.shape[0]]


def test_predict_probabilities_retains_every_class_score() -> None:
    classifier = ActionClassifier(
        embedding_dim=2,
        num_actions=3,
        device="cpu",
        model_type="mlp",
    )
    classifier.model = FixedLogits()
    embeddings = np.zeros((2, 2), dtype=np.float32)

    probabilities = classifier.predict_probabilities(embeddings)
    actions, confidences = classifier.predict(embeddings)

    assert probabilities.shape == (2, 3)
    assert probabilities.sum(axis=1) == pytest.approx(np.ones(2))
    assert actions.tolist() == [1, 0]
    assert confidences == pytest.approx(probabilities.max(axis=1))


def test_checkpoint_catalog_metadata_is_required_for_robot_inference(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    catalog = load_skill_system(root / "configs" / "skills" / "pick_place.yaml").catalog
    classifier = ActionClassifier(
        embedding_dim=2,
        num_actions=catalog.class_count,
        device="cpu",
        model_type="mlp",
    )
    checkpoint = tmp_path / "classifier.pt"
    classifier.save(str(checkpoint), catalog=catalog)

    restored = ActionClassifier(
        embedding_dim=2,
        num_actions=catalog.class_count,
        device="cpu",
        model_type="mlp",
    )
    restored.load(str(checkpoint), catalog=catalog)

    wrong_catalog = SkillCatalog(schema_version=catalog.schema_version + 1, skills=catalog.skills)
    with pytest.raises(ValueError, match="does not match the active catalog"):
        restored.load(str(checkpoint), catalog=wrong_catalog)


def test_legacy_checkpoint_is_rejected_when_catalog_validation_is_requested(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[2]
    catalog = load_skill_system(root / "configs" / "skills" / "pick_place.yaml").catalog
    classifier = ActionClassifier(
        embedding_dim=2,
        num_actions=catalog.class_count,
        device="cpu",
        model_type="mlp",
    )
    checkpoint = tmp_path / "legacy.pt"
    classifier.save(str(checkpoint))

    with pytest.raises(ValueError, match="Legacy checkpoint"):
        classifier.load(str(checkpoint), catalog=catalog)
