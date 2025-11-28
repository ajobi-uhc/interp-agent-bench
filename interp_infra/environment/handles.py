"""Handles for prepared resources."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelHandle:
    """Handle to a prepared model."""
    name: str
    volume_path: str
    hidden: bool = False
    is_peft: bool = False
    base_model: Optional[str] = None
    base_model_path: Optional[str] = None


@dataclass  
class RepoHandle:
    """Handle to a prepared repo."""
    url: str
    local_path: str
    dockerfile: Optional[str] = None
    container_name: Optional[str] = None
    container_running: bool = False