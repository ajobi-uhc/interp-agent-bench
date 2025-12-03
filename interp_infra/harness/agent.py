"""Thin wrapper over agent SDKs - just prompt + MCP."""

from typing import Optional, Literal, AsyncIterator

from .providers import run_claude, run_openai, run_gemini


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

    # Run with appropriate provider
    if provider == "claude":
        async for message in run_claude(
            mcp_config=mcp_config,
            system_prompt=prompt,
            task=user_message,
            model=model,
        ):
            yield message

    elif provider == "gemini":
        async for message in run_gemini(
            mcp_config=mcp_config,
            system_prompt=prompt,
            task=user_message,
            model=model,
        ):
            yield message

    elif provider == "openai":
        async for message in run_openai(
            mcp_config=mcp_config,
            system_prompt=prompt,
            task=user_message,
            model=model,
        ):
            yield message

    else:
        raise ValueError(f"Unknown provider: {provider}")
