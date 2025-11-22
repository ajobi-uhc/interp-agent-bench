"""Toolkit helpers for building agent orchestration harnesses."""

from pathlib import Path
from typing import Optional, Callable, Any, AsyncIterator
from .providers import AgentProvider, AgentOptions, create_agent_provider


def create_agent(
    deployment,
    system_prompt: str,
    user_prompt: str,
    provider: str = "claude",
    session_id: Optional[str] = None,
    agent_model: Optional[str] = None,
    allowed_tools: Optional[list[str]] = None,
    stderr_callback: Optional[Callable] = None,
    verbose: bool = False,
    **options
) -> AgentProvider:
    """
    Create an agent with minimal boilerplate.

    This abstracts away MCP configuration, workspace setup, and provider details.
    Harness authors just provide prompts and get back a ready-to-use agent.

    Args:
        deployment: Deployment object from Stage 1+2 (has jupyter_url, session_id, workspace)
        system_prompt: System prompt for the agent
        user_prompt: Initial user prompt/task
        provider: Agent provider ("claude" or "openai")
        session_id: Override session_id (default: use deployment.session_id)
        agent_model: Override model (default: provider default)
        allowed_tools: Override allowed tools list (default: all notebook tools)
        stderr_callback: Callback for MCP server stderr logs
        verbose: Enable verbose logging
        **options: Additional provider-specific options

    Returns:
        AgentProvider ready to use

    Example:
        >>> agent = toolkit.create_agent(
        ...     deployment,
        ...     system_prompt="You are a research assistant.",
        ...     user_prompt="Analyze this model's behavior."
        ... )
        >>> async for message in agent.receive_response():
        ...     print(message)
    """
    # Get Python from virtual environment for MCP server
    venv_python = Path.cwd() / ".venv" / "bin" / "python"

    # Default session_id from deployment
    if session_id is None:
        session_id = deployment.session_id

    # Build MCP server configuration
    mcp_config = {
        "notebooks": {
            "type": "stdio",
            "command": str(venv_python),
            "args": ["-m", "scribe.notebook.notebook_mcp_server"],
            "env": {
                "SCRIBE_URL": deployment.jupyter_url,
                "NOTEBOOK_OUTPUT_DIR": str(getattr(deployment, 'workspace', Path.cwd() / "notebooks")),
            }
        }
    }

    # Default allowed tools (all notebook operations)
    if allowed_tools is None:
        allowed_tools = [
            "mcp__notebooks__attach_to_session",
            "mcp__notebooks__start_new_session",
            "mcp__notebooks__execute_code",
            "mcp__notebooks__add_markdown",
            "mcp__notebooks__edit_cell",
            "mcp__notebooks__shutdown_session",
        ]

    # Default stderr callback (print MCP logs)
    if stderr_callback is None:
        def stderr_callback(line: str):
            if line.strip():
                print(f"[MCP] {line.rstrip()}", flush=True)

    # Get workspace path
    workspace_path = getattr(deployment, 'workspace', Path.cwd() / "notebooks")
    if not isinstance(workspace_path, Path):
        workspace_path = Path(workspace_path)

    # Build agent options
    agent_options = AgentOptions(
        system_prompt=system_prompt,
        workspace_path=workspace_path,
        mcp_config=mcp_config,
        allowed_tools=allowed_tools,
        stderr_callback=stderr_callback,
        verbose=verbose,
        agent_model=agent_model,
        **options
    )

    # Create and return provider
    agent = create_agent_provider(provider, agent_options)

    # Store user prompt for later use
    agent._user_prompt = user_prompt

    return agent


async def run_agent(
    agent: AgentProvider,
    stream_callback: Optional[Callable[[Any], None]] = None
) -> dict:
    """
    Run an agent and collect results.

    Args:
        agent: Agent to run (from create_agent())
        stream_callback: Optional callback for each message (for live streaming)

    Returns:
        Dictionary with outputs, token usage, etc.

    Example:
        >>> agent = toolkit.create_agent(deployment, sys_prompt, user_prompt)
        >>> result = await toolkit.run_agent(agent)
        >>> print(result['outputs'])
    """
    # Query agent with stored user prompt
    user_prompt = getattr(agent, '_user_prompt', '')
    await agent.query(user_prompt)

    # Collect results
    outputs = []
    total_tokens = 0
    message_count = 0

    async for message in agent.receive_response():
        # Call stream callback if provided
        if stream_callback:
            stream_callback(message)

        # Track token usage
        if hasattr(message, 'usage') and message.usage:
            input_tokens = getattr(message.usage, 'input_tokens', 0)
            output_tokens = getattr(message.usage, 'output_tokens', 0)
            total_tokens += input_tokens + output_tokens
            message_count += 1

        # Collect message content
        if hasattr(message, 'content'):
            outputs.append(message.content)

    return {
        'outputs': outputs,
        'total_tokens': total_tokens,
        'message_count': message_count,
    }


def get_workspace_outputs(workspace_path: Path) -> dict:
    """
    Read outputs from workspace (notebooks, prompts, etc.).

    Args:
        workspace_path: Path to workspace directory

    Returns:
        Dictionary with workspace contents

    Example:
        >>> outputs = toolkit.get_workspace_outputs(deployment.workspace)
        >>> print(outputs['notebook'])
    """
    workspace_path = Path(workspace_path)

    outputs = {}

    # Read system/user prompts if saved
    if (workspace_path / "system_prompt.md").exists():
        outputs['system_prompt'] = (workspace_path / "system_prompt.md").read_text()

    if (workspace_path / "user_prompt.md").exists():
        outputs['user_prompt'] = (workspace_path / "user_prompt.md").read_text()

    # Find notebook files
    notebooks = list(workspace_path.glob("*.ipynb"))
    if notebooks:
        outputs['notebooks'] = [str(nb) for nb in notebooks]

    return outputs
