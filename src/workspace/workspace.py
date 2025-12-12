"""Workspace - configuration for agent's filesystem."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .library import Library
from .skill import Skill

if TYPE_CHECKING:
    from ..execution.session_base import SessionBase


@dataclass
class Workspace:
    """
    Declarative configuration for agent's workspace.

    Pure data class - no behavior. Sessions interpret this config.

    Specifies:
    - Local files/directories to mount
    - Code libraries to make importable
    - Skills for agent discovery
    - Custom initialization code

    Examples:
        # Basic workspace with libraries
        workspace = Workspace(
            libraries=[
                Library.from_file("utils.py"),
                Library.from_skill_dir("skills/steering"),
            ]
        )

        # Mount data and skills
        workspace = Workspace(
            local_dirs=[("./data", "/workspace/data")],
            skill_dirs=["./skills/research"],
        )

        # Custom model loading
        workspace = Workspace(
            custom_init_code=\"\"\"
            from transformers import AutoModel
            model = AutoModel.from_pretrained(...)
            \"\"\"
        )
    """

    # Files and directories to mount (runtime)
    local_dirs: list[tuple[str, str]] = field(default_factory=list)
    local_files: list[tuple[str, str]] = field(default_factory=list)

    # Code libraries for agent
    libraries: list[Library] = field(default_factory=list)

    # Skills for Claude discovery
    skills: list[Skill] = field(default_factory=list)
    skill_dirs: list[str] = field(default_factory=list)

    # Custom initialization code
    custom_init_code: Optional[str] = None

    # Model loading configuration
    preload_models: bool = True
    hidden_model_loading: bool = True

    def get_library_docs(self) -> str:
        """Get combined documentation for all libraries, for including in agent prompt."""
        if not self.libraries:
            return ""
        docs = [lib.get_prompt_docs() for lib in self.libraries]
        return "# Available Libraries\n\n" + "\n\n".join(docs)


def default_workspace() -> Workspace:
    """
    Create default workspace (empty).

    Model and repo loading is handled by session factories
    based on sandbox configuration.

    Returns:
        Empty Workspace instance
    """
    return Workspace()
