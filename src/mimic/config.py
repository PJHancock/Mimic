"""Central configuration for the Mimic project."""

import os
from pathlib import Path
from typing import Any, Dict

import yaml


class Config:
    """Configuration management for Mimic."""

    _defaults = {
        "data_dir": "data",
        "output_dir": "outputs",
        "fps": 30,
        "video_extension": ".mp4",
        "cache_embeddings": True,
        "cache_tracks": True,
        "device": "cuda",  # or "cpu"
        "random_seed": 42,
        "temporal_model": {
            "type": "gru",  # "gru", "transformer", "mlp"
            "hidden_size": 256,
            "num_layers": 2,
            "dropout": 0.1,
        },
        "tracking": {
            "hand_confidence_threshold": 0.5,
            "object_confidence_threshold": 0.5,
            "use_mediapipe": True,
            "use_sam2": True,
        },
        "robot": {
            "arm_control_hz": 100,
            "gripper_control_hz": 10,
        },
    }

    def __init__(self, config_path: str = None):
        """Initialize config from file or defaults.

        Args:
            config_path: Path to YAML config file. If None, uses defaults.
        """
        self.config = self._defaults.copy()

        if config_path and os.path.exists(config_path):
            with open(config_path) as f:
                user_config = yaml.safe_load(f) or {}
                self.config.update(user_config)

        # Check for local override
        if os.path.exists("config.local.yaml"):
            with open("config.local.yaml") as f:
                local_config = yaml.safe_load(f) or {}
                self.config.update(local_config)

    def __getitem__(self, key: str) -> Any:
        """Get config value by key."""
        return self.config.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set config value by key."""
        self.config[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value with default."""
        return self.config.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Get config as dictionary."""
        return self.config.copy()

    def save(self, path: str) -> None:
        """Save config to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.config, f)


# Global config instance
_global_config = None


def get_config(config_path: str = None) -> Config:
    """Get or create global config instance."""
    global _global_config
    if _global_config is None:
        _global_config = Config(config_path)
    return _global_config


def reset_config() -> None:
    """Reset global config (for testing)."""
    global _global_config
    _global_config = None


# Convenience paths
def get_data_dir() -> Path:
    """Get data directory path."""
    return Path(get_config()["data_dir"])


def get_output_dir() -> Path:
    """Get output directory path."""
    return Path(get_config()["output_dir"])


def get_embeddings_dir() -> Path:
    """Get embeddings cache directory."""
    return get_data_dir() / "embeddings"


def get_tracks_dir() -> Path:
    """Get tracks cache directory."""
    return get_data_dir() / "tracks"
