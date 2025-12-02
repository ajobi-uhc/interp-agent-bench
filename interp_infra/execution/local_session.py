"""Local session management."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import shutil

from ..workspace import Workspace
from .session_base import SessionBase


@dataclass
class LocalSession(SessionBase):
    """A local session for running agents with MCP tools only."""

    session_id: str
    _namespace: dict = field(default_factory=dict, init=False, repr=False)

    def _mount_dir(self, src: str, dest: str):
        """Copy local directory to workspace."""
        src_path = Path(src)
        dest_path = self.workspace_path / dest.lstrip("/")

        if not src_path.exists():
            raise FileNotFoundError(f"Directory not found: {src}")

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src_path, dest_path, dirs_exist_ok=True)

    def _mount_file(self, src: str, dest: str):
        """Copy local file to workspace."""
        src_path = Path(src)
        dest_path = self.workspace_path / dest.lstrip("/")

        if not src_path.exists():
            raise FileNotFoundError(f"File not found: {src}")

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dest_path)

    def _copy_skill_dir(self, skill_dir: str):
        """Copy skill directory to workspace/.claude/skills/."""
        src_path = Path(skill_dir)
        if not src_path.exists():
            raise FileNotFoundError(f"Skill directory not found: {skill_dir}")

        skills_base = self.workspace_path / ".claude" / "skills"
        dest_path = skills_base / src_path.name

        skills_base.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src_path, dest_path, dirs_exist_ok=True)

    def _execute_code(self, code: str):
        """Execute code in local namespace."""
        exec(code, self._namespace)

    def exec_file(self, file_path: str, **kwargs):
        """Not supported - use regular Python imports."""
        raise NotImplementedError(
            "exec_file() not supported for LocalSession. Use regular Python imports."
        )


def create_local_session(
    workspace: Optional[Workspace] = None,
    name: str = "local-session",
    workspace_dir: str = "./workspace",
) -> LocalSession:
    """
    Create a local session for agent work.

    Args:
        workspace: Workspace configuration
        name: Session name
        workspace_dir: Local workspace directory path

    Returns:
        LocalSession with workspace setup
    """
    print(f"Creating local session: {name}")

    workspace_path = Path(workspace_dir)
    workspace_path.mkdir(parents=True, exist_ok=True)

    session = LocalSession(
        session_id=name,
        workspace_path=workspace_path,
    )

    # Setup workspace if provided
    if workspace:
        workspace.setup_in(session)

    print(f"  Local session ready: {name}")
    print(f"  Workspace: {workspace_path.absolute()}")
    return session
