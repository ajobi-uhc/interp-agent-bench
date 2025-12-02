"""Base session functionality."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..workspace import Workspace


@dataclass
class SessionBase:
    """
    Base class for all session types.

    Session = execution context (sandbox + workspace).

    Does NOT manage:
    - MCP servers (passed to harness)
    - System prompts (passed to harness)
    """

    workspace_path: Path

    def _mount_dir(self, src: str, dest: str):
        """Mount local directory into workspace. Subclasses implement."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement _mount_dir()")

    def _mount_file(self, src: str, dest: str):
        """Mount local file into workspace. Subclasses implement."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement _mount_file()")

    def _copy_skill_dir(self, skill_dir: str):
        """Copy skill directory to workspace/.claude/skills/. Subclasses implement."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement _copy_skill_dir()")

    def _execute_code(self, code: str):
        """Execute code in session context. Subclasses implement."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement _execute_code()")

    def exec_file(self, file_path: str, **kwargs):
        """Execute a Python file. Subclasses implement."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement exec_file()")
