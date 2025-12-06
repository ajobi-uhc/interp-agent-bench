"""OpenAI provider (placeholder for future implementation)."""

from typing import AsyncIterator


async def run_openai(
    mcp_config: dict,
    system_prompt: str,
    task: str,
    model: str = "gpt-4o-mini",
    verbose: bool = False,
) -> AsyncIterator[dict]:
    """
    Run OpenAI agent with MCP.

    Args:
        mcp_config: MCP server configuration
        system_prompt: System prompt for agent
        task: User task/query
        model: OpenAI model to use
        verbose: Print debug info

    Yields:
        Agent messages
    """
    raise NotImplementedError("OpenAI provider not yet implemented")
