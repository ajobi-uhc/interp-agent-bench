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

    Session = execution context that interprets workspace config.

    All sessions provide:
    - exec(code): Execute Python code
    - setup(workspace): Apply workspace configuration

    Does NOT manage:
    - MCP servers (passed to harness)
    - System prompts (passed to harness)
    """

    workspace_path: Path

    def exec(self, code: str, **kwargs):
        """
        Execute Python code in this session's context.

        Args:
            code: Python code to execute
            **kwargs: Session-specific options (e.g., hidden=True for notebooks)

        Returns:
            Execution result (format varies by session type)
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement exec()")

    def setup(self, workspace: "Workspace"):
        """
        Apply workspace configuration to this session.

        Each session type implements this to handle:
        - Mounting local files/dirs (if applicable)
        - Installing libraries
        - Installing skills
        - Running custom init code

        Args:
            workspace: Workspace configuration to apply
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement setup()")
