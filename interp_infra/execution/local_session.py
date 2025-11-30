"""Local session management for lightweight agent work."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .session_base import SessionBase

if TYPE_CHECKING:
    from ..extension import Extension


@dataclass
class LocalSession(SessionBase):
    """
    A local session for running agents with MCP tools.

    Unlike NotebookSession/CLISession, this runs entirely on the local machine.
    Use this when you want to use MCP tools (like from ScopedSandbox) with an agent
    but don't need a full notebook/CLI environment.

    Manages:
    1. Local workspace directory
    2. Configuration accumulation (MCP endpoints, prompts, extensions)
    3. Local Python namespace for extension code
    """
    session_id: str
    workspace: Path
    _namespace: dict = field(default_factory=dict, init=False, repr=False)

    def _execute_extension(self, extension: "Extension"):
        """Execute extension code in local namespace."""
        if extension.code:
            exec(extension.code, self._namespace)

    def exec_file(self, file_path: str, **kwargs):
        """
        Not supported for LocalSession.

        LocalSession runs entirely on the local machine - use regular Python imports instead.
        """
        raise NotImplementedError(
            "exec_file() is not supported for LocalSession. "
            "LocalSession runs locally - use regular Python imports instead."
        )

    @property
    def mcp_config(self) -> dict:
        """
        MCP configuration for connecting agents.

        Includes all added proxies/extensions exposed as MCP.
        """
        config = {}

        # Add all MCP endpoints
        for endpoint in self._mcp_endpoints:
            config.update(endpoint)

        return config

    @property
    def system_prompt(self) -> str:
        """Accumulated system prompt from extensions with workspace info."""
        prompts = [
            f"You have a local workspace directory at: {self.workspace.absolute()}\n"
            "You can read/write files in this directory using your built-in file tools."
        ]
        prompts.extend(self._prompts)
        return "\n\n".join(prompts)


def create_local_session(
    name: str = "local-session",
    workspace: str = "./workspace",
) -> LocalSession:
    """
    Create a local session for agent work.

    Args:
        name: Session name/identifier
        workspace: Local workspace directory path

    Returns:
        LocalSession ready for agent

    Example:
        # Create local session
        session = create_local_session(name="redteam", workspace="./outputs")

        # Add ScopedSandbox proxy as MCP tool
        session.add(autorater_proxy)

        # Add extensions
        session.add(Extension.from_file("skills/redteam.md"))

        # Run agent
        async for msg in run_agent(
            mcp_config=session.mcp_config,
            system_prompt=session.system_prompt,
            task="Red team the autorater",
            provider="claude"
        ):
            print(msg)
    """
    print(f"Creating local session: {name}")

    # Create workspace directory
    workspace_path = Path(workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)

    session = LocalSession(
        session_id=name,
        workspace=workspace_path,
    )

    print(f"  Local session ready: {name}")
    print(f"  Workspace: {workspace_path.absolute()}")

    return session
