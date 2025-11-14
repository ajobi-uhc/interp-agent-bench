"""Save and load sandbox deployment state for pause/resume functionality."""

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class SandboxState:
    """Saved state of a Modal sandbox deployment."""
    sandbox_id: str
    jupyter_url: str
    jupyter_port: int
    jupyter_token: str
    experiment_name: str
    config_path: str
    workspace_path: str
    timestamp: str  # When deployment was created

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SandboxState":
        """Create from dictionary."""
        return cls(**data)


def get_state_file_path(experiment_name: str) -> Path:
    """Get path to state file for an experiment."""
    state_dir = Path.cwd() / ".sandbox_states"
    state_dir.mkdir(exist_ok=True)
    return state_dir / f"{experiment_name}.json"


def save_sandbox_state(state: SandboxState) -> Path:
    """
    Save sandbox state to disk.

    Args:
        state: Sandbox state to save

    Returns:
        Path to saved state file
    """
    state_file = get_state_file_path(state.experiment_name)
    state_file.write_text(json.dumps(state.to_dict(), indent=2))
    print(f"💾 Sandbox state saved to: {state_file}")
    return state_file


def load_sandbox_state(experiment_name: str) -> Optional[SandboxState]:
    """
    Load sandbox state from disk.

    Args:
        experiment_name: Name of experiment

    Returns:
        SandboxState if found, None otherwise
    """
    state_file = get_state_file_path(experiment_name)
    if not state_file.exists():
        return None

    data = json.loads(state_file.read_text())
    return SandboxState.from_dict(data)


def list_saved_sandboxes() -> list[SandboxState]:
    """
    List all saved sandbox states.

    Returns:
        List of SandboxState objects
    """
    state_dir = Path.cwd() / ".sandbox_states"
    if not state_dir.exists():
        return []

    states = []
    for state_file in state_dir.glob("*.json"):
        data = json.loads(state_file.read_text())
        states.append(SandboxState.from_dict(data))

    return states


def delete_sandbox_state(experiment_name: str) -> bool:
    """
    Delete saved sandbox state.

    Args:
        experiment_name: Name of experiment

    Returns:
        True if deleted, False if not found
    """
    state_file = get_state_file_path(experiment_name)
    if state_file.exists():
        state_file.unlink()
        print(f"🗑️  Deleted sandbox state: {experiment_name}")
        return True
    return False
