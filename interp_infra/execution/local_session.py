"""Local session management for lightweight agent work."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from ..environment.scoped_sandbox import Proxy
    from ..extension import Extension


@dataclass
class LocalSession:
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

    # Configuration accumulators
    _mcp_endpoints: list[dict] = field(default_factory=list, init=False)
    _prompts: list[str] = field(default_factory=list, init=False)
    _namespace: dict = field(default_factory=dict, init=False)

    def add(self, item: Union["Proxy", "Extension"]):
        """
        Add extension or proxy to the session.

        - Extension: Code executed locally in namespace, docs added to prompt
        - Proxy: Added as MCP tool (from ScopedSandbox)

        Args:
            item: Extension (local code) or Proxy (from ScopedSandbox)

        Examples:
            session.add(proxy)        # Adds autorater as MCP tool
            session.add(extension)    # Executes code locally
        """
        from ..environment.scoped_sandbox import Proxy
        from ..extension import Extension

        if isinstance(item, Proxy):
            # Proxy → MCP tool (ScopedSandbox interface)
            self._mcp_endpoints.append(item.as_mcp_config())

        elif isinstance(item, Extension):
            # Extension → Execute code locally + docs to prompt
            if item.code:
                exec(item.code, self._namespace)
            if item.docs:
                self._prompts.append(item.docs)

        else:
            raise TypeError(f"Expected Proxy or Extension, got {type(item)}")

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
        """Accumulated system prompt from extensions."""
        prompts = []

        # Add workspace info
        prompts.append(f"You have a local workspace directory at: {self.workspace.absolute()}\nYou can read/write files in this directory using your built-in file tools.")

        # Add extension prompts
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
