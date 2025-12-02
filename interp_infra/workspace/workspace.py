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
    Configuration for agent's workspace filesystem.

    Specifies:
    - Local files/directories to mount
    - Code libraries to make importable
    - Skill directories for discovery
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

    # Files and directories to mount
    local_dirs: list[tuple[str, str]] = field(default_factory=list)
    local_files: list[tuple[str, str]] = field(default_factory=list)

    # Code libraries for agent
    libraries: list[Library] = field(default_factory=list)

    # Skills for Claude discovery (generates .claude/skills/)
    skills: list[Skill] = field(default_factory=list)

    # Skill directories (copied to workspace/.claude/skills/)
    skill_dirs: list[str] = field(default_factory=list)

    # Custom initialization code (overrides defaults)
    custom_init_code: Optional[str] = None

    def setup_in(self, session: "SessionBase"):
        """
        Setup this workspace in a session.

        Installs:
        1. Mounts local files/dirs
        2. Installs libraries
        3. Installs skills (generates SKILL.md)
        4. Copies skill directories
        5. Runs custom init code

        Args:
            session: Session to setup workspace in
        """
        # Mount local directories
        for src, dest in self.local_dirs:
            session._mount_dir(src, dest)

        # Mount local files
        for src, dest in self.local_files:
            session._mount_file(src, dest)

        # Install libraries (makes them importable)
        for library in self.libraries:
            library.install_in(session)

        # Install skills (generates .claude/skills/{name}/SKILL.md)
        for skill in self.skills:
            skill.install_in(session)

        # Copy skill directories to workspace/.claude/skills/
        for skill_dir in self.skill_dirs:
            session._copy_skill_dir(skill_dir)

        # Run custom initialization code
        if self.custom_init_code:
            session._execute_code(self.custom_init_code)


def default_workspace() -> Workspace:
    """
    Create default workspace (empty).

    Model and repo loading is handled by session factories
    based on sandbox configuration.

    Returns:
        Empty Workspace instance
    """
    return Workspace()
