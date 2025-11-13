"""Parse YAML configuration files."""

import yaml
from pathlib import Path
from .schema import ExperimentConfig


def load_config(config_path: Path) -> ExperimentConfig:
    """
    Load and parse experiment configuration from YAML.

    Args:
        config_path: Path to YAML config file

    Returns:
        Parsed ExperimentConfig
    """
    with open(config_path) as f:
        data = yaml.safe_load(f)

    return ExperimentConfig(**data)
