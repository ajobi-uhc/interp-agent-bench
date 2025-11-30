"""Simple agent runner using MCP and system prompts."""

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
    mcp_config: dict,
    system_prompt: str,
    task: str,
    provider: Provider = "claude",
    model: Optional[str] = None,
    verbose: bool = False,
) -> AsyncIterator[dict]:
    """
    Run an agent with MCP tools and system prompt.

    This is a simple harness. For full control, use the provider
    functions directly or write your own harness with Claude SDK.

    Args:
        mcp_config: MCP server configuration (from session.mcp_config)
        system_prompt: System prompt (from session.system_prompt or custom)
        task: What the agent should do
        provider: "claude", "gemini", or "openai"
        model: Override default model for provider
        verbose: Print debug info

    Yields:
        Agent messages (dict format)

    Example:
        # Using session primitives
        session = create_notebook_session(sandbox)
        session.add(steering_extension)
        session.add(target_proxy, as_="mcp")

        async for msg in run_agent(
            mcp_config=session.mcp_config,
            system_prompt=session.system_prompt,
            task="Find steering vectors",
            provider="claude"
        ):
            print(msg)
    """
    # Use model parameter or default for provider
    model = model or DEFAULT_MODELS[provider]

    # Run with appropriate provider
    if provider == "claude":
        async for message in run_claude(
            mcp_config=mcp_config,
            system_prompt=system_prompt,
            task=task,
            model=model,
            verbose=verbose,
        ):
            yield message

    elif provider == "gemini":
        async for message in run_gemini(
            mcp_config=mcp_config,
            system_prompt=system_prompt,
            task=task,
            model=model,
            verbose=verbose,
        ):
            yield message

    elif provider == "openai":
        async for message in run_openai(
            mcp_config=mcp_config,
            system_prompt=system_prompt,
            task=task,
            model=model,
            verbose=verbose,
        ):
            yield message

    else:
        raise ValueError(f"Unknown provider: {provider}")
