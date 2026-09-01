"""Tests for YAML-backed project configuration."""

from pathlib import Path

import pytest

import mimic.config as config_module
from mimic.config import Config


def test_default_yaml_is_the_only_baseline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    baseline = tmp_path / "default.yaml"
    baseline.write_text("source: yaml\nnested:\n  value: 7\n")
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", baseline)
    monkeypatch.chdir(tmp_path)

    assert Config().to_dict() == {"source": "yaml", "nested": {"value": 7}}


def test_missing_explicit_overlay_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = tmp_path / "default.yaml"
    baseline.write_text("source: yaml\n")
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", baseline)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError, match="Configuration overlay not found"):
        Config(tmp_path / "missing.yaml")
