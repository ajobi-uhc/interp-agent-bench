"""Gemini provider (placeholder for future implementation)."""

from typing import AsyncIterator


async def run_gemini(
    mcp_config: dict,
    system_prompt: str,
    task: str,
    model: str = "gemini-2.0-flash",
    verbose: bool = False,
) -> AsyncIterator[dict]:
    """
    Run Gemini agent with MCP.

    Args:
        mcp_config: MCP server configuration
        system_prompt: System prompt for agent
        task: User task/query
        model: Gemini model to use
        verbose: Print debug info

    Yields:
        Agent messages
    """
    raise NotImplementedError("Gemini provider not yet implemented")
