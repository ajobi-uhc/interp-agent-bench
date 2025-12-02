"""Simple agent runner using MCP and system prompts."""

from typing import Optional, Literal, AsyncIterator, TYPE_CHECKING

from .providers import run_claude, run_openai, run_gemini

if TYPE_CHECKING:
    from ..execution.session_base import SessionBase


Provider = Literal["claude", "gemini", "openai"]

# Default models for each provider (can be overridden via model parameter)
DEFAULT_MODELS = {
    "claude": "claude-sonnet-4-5-20250929",
    "gemini": "gemini-2.0-flash",
    "openai": "gpt-4o-mini",
}


async def run_agent(
    session: "SessionBase",
    task: str,
    mcp_servers: Optional[list[dict]] = None,
    prompts: Optional[list[str]] = None,
    provider: Provider = "claude",
    model: Optional[str] = None,
) -> AsyncIterator[dict]:
    """
    Run an agent with session, MCP servers, and prompts.

    Three separate inputs:
    1. session - execution context (workspace)
    2. mcp_servers - list of MCP server configs
    3. prompts - list of system prompt strings

    Args:
        session: Session with workspace
        task: What the agent should do
        mcp_servers: List of MCP server configs (optional)
        prompts: List of system prompt strings (optional)
        provider: "claude", "gemini", or "openai"
        model: Override default model for provider

    Yields:
        Agent messages (dict format)

    Example:
        # Setup
        workspace = Workspace(libraries=[...])
        session = create_notebook_session(sandbox, workspace)

        # MCP servers from various sources
        mcp_servers = [
            {"web": {"command": "..."}},
            scoped_sandbox.serve("tools.py", expose_as="mcp"),
        ]

        # Prompts from various sources
        prompts = [
            "# Base instructions...",
            "# Model info...",
        ]

        # Run agent
        async for msg in run_agent(
            session=session,
            task="Find steering vectors",
            mcp_servers=mcp_servers,
            prompts=prompts,
            provider="claude",
        ):
            print(msg)
    """
    # Build MCP config
    mcp_config = {}
    if mcp_servers:
        for server in mcp_servers:
            mcp_config.update(server)

    # Build system prompt
    system_prompt_parts = [
        f"Workspace: {session.workspace_path}",
        "You can write code in your execution context and use MCP tools.",
    ]

    if prompts:
        system_prompt_parts.extend(prompts)

    system_prompt = "\n\n".join(system_prompt_parts)

    # Use model parameter or default for provider
    model = model or DEFAULT_MODELS[provider]

    # Run with appropriate provider
    if provider == "claude":
        async for message in run_claude(
            mcp_config=mcp_config,
            system_prompt=system_prompt,
            task=task,
            model=model,
        ):
            yield message

    elif provider == "gemini":
        async for message in run_gemini(
            mcp_config=mcp_config,
            system_prompt=system_prompt,
            task=task,
            model=model,
        ):
            yield message

    elif provider == "openai":
        async for message in run_openai(
            mcp_config=mcp_config,
            system_prompt=system_prompt,
            task=task,
            model=model,
        ):
            yield message

    else:
        raise ValueError(f"Unknown provider: {provider}")
