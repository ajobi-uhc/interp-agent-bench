"""Thin wrapper over agent SDKs - just prompt + MCP."""

from typing import Optional, Literal, AsyncIterator
import asyncio

from .providers import run_claude, run_openai, run_gemini
from .logging import run_agent_with_logging


Provider = Literal["claude", "gemini", "openai"]

# Default models for each provider (can be overridden via model parameter)
DEFAULT_MODELS = {
    "claude": "claude-sonnet-4-5-20250929",
    "gemini": "gemini-2.0-flash",
    "openai": "gpt-4o-mini",
}


async def run_agent(
    prompt: str,
    mcp_config: dict,
    user_message: str = "",
    provider: Provider = "claude",
    model: Optional[str] = None,
    kwargs: Optional[dict] = None,
    enable_logging: bool = True,
) -> AsyncIterator[dict]:
    """
    Run an agent with explicit prompt and MCP config.

    Thin wrapper over agent SDKs - no magic, no hidden behavior.
    You build the prompt, you provide the MCP config.

    Args:
        prompt: System prompt for the agent
        mcp_config: MCP server configuration dict
        user_message: Initial user message (optional, defaults to empty)
        provider: "claude", "gemini", or "openai"
        model: Override default model for provider
        kwargs: Additional provider-specific options
        enable_logging: Enable logging adapter (default: True)

    Yields:
        Agent messages (dict format)

    Example:
        # Setup environment
        session = create_notebook_session(sandbox, workspace)

        # Build prompt explicitly
        prompt = f'''
        {session.model_info_text}

        Find steering vectors for the loaded model.
        '''

        # Run agent (explicit prompt + mcp)
        async for msg in run_agent(
            prompt=prompt,
            mcp_config=session.mcp_config,
            provider="claude",
        ):
            print(msg)
    """
    # Use model parameter or default for provider
    model = model or DEFAULT_MODELS[provider]

    # Get provider stream
    if provider == "claude":
        stream = run_claude(
            mcp_config=mcp_config,
            system_prompt=prompt,
            task=user_message,
            model=model,
            kwargs=kwargs or {}
        )

    elif provider == "gemini":
        stream = run_gemini(
            mcp_config=mcp_config,
            system_prompt=prompt,
            task=user_message,
            model=model,
        )

    elif provider == "openai":
        stream = run_openai(
            mcp_config=mcp_config,
            system_prompt=prompt,
            task=user_message,
            model=model,
        )

    else:
        raise ValueError(f"Unknown provider: {provider}")

    # Wrap with logging if enabled
    if enable_logging:
        async for message in run_agent_with_logging(stream):
            yield message
    else:
        async for message in stream:
            yield message
