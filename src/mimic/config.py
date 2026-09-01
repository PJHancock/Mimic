"""Central configuration for the Mimic project."""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
LOCAL_CONFIG_PATH = Path("config.local.yaml")


def _load_mapping(path: Path, description: str) -> Dict[str, Any]:
    """Load one YAML mapping, failing clearly when the configuration is invalid."""

    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    with path.open() as stream:
        payload = yaml.safe_load(stream) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must contain a YAML mapping: {path}")
    return payload


class Config:
    """Configuration management for Mimic."""

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize from the canonical YAML, then apply optional YAML overlays.

        Args:
            config_path: Optional experiment-specific YAML overlay.
        """
        self.config = _load_mapping(DEFAULT_CONFIG_PATH, "Default configuration")

        if config_path is not None:
            requested_path = Path(config_path)
            if requested_path.resolve() != DEFAULT_CONFIG_PATH.resolve():
                self.config.update(_load_mapping(requested_path, "Configuration overlay"))

        # Check for local override
        if LOCAL_CONFIG_PATH.is_file():
            self.config.update(_load_mapping(LOCAL_CONFIG_PATH, "Local configuration override"))

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


def get_config(config_path: Optional[Union[str, Path]] = None) -> Config:
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
