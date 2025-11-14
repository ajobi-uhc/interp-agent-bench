"""Parse YAML configuration files."""

import yaml
from pathlib import Path
from .schema import ExperimentConfig


def load_config(config_path: Path) -> ExperimentConfig:
    """
    Load and parse experiment configuration from YAML.

    If the 'task' field in the YAML ends with '.md', it will be treated as a path
    to a markdown file (relative to the config file) and loaded.

    Args:
        config_path: Path to YAML config file

    Returns:
        Parsed ExperimentConfig
    """
    config_path = Path(config_path)

    with open(config_path) as f:
        data = yaml.safe_load(f)

    # Check if task is a path to a .md file
    if 'task' in data and isinstance(data['task'], str) and data['task'].endswith('.md'):
        task_path = Path(data['task'])

        # If it's not an absolute path, resolve relative to config directory
        if not task_path.is_absolute():
            task_path = config_path.parent / task_path

        if task_path.exists():
            with open(task_path) as f:
                data['task'] = f.read()
        else:
            raise FileNotFoundError(f"Task file not found: {task_path}")

    return ExperimentConfig(**data)
