from pathlib import Path

import pytest
import yaml

from mimic.skills import GraphStatePostProcessor, SkillCatalog, SkillGraph


@pytest.fixture
def skill_config():
    root = Path(__file__).resolve().parents[2]
    return yaml.safe_load((root / "configs" / "skills" / "pick_place.yaml").read_text())[
        "skill_system"
    ]


@pytest.fixture
def skill_catalog(skill_config):
    return SkillCatalog.model_validate(skill_config["catalog"])


@pytest.fixture
def skill_graph(skill_catalog, skill_config):
    return SkillGraph.from_mapping(skill_catalog, skill_config["graph"])


@pytest.fixture
def post_state_settings():
    return {
        "minimum_confidence": 0.3,
        "minimum_transition_margin": 0.05,
        "maximum_second_choice_gap": 0.05,
        "required_consecutive_observations": 1,
        "missing_detection_timeout_s": 0.2,
    }


@pytest.fixture
def post_processor(skill_catalog, skill_graph, post_state_settings):
    return GraphStatePostProcessor(skill_catalog, skill_graph, post_state_settings)
