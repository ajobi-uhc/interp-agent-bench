"""Environment stage - Infrastructure setup (GPU, models download, repos)."""

from .modal import setup_environment, EnvironmentHandle

__all__ = ['setup_environment', 'EnvironmentHandle']
