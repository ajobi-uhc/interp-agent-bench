"""Claude provider using claude-agent-sdk."""

from typing import AsyncIterator


async def run_claude(
    mcp_config: dict,
    system_prompt: str,
    task: str,
    model: str = "claude-sonnet-4-5-20250929",
    verbose: bool = False,
) -> AsyncIterator[dict]:
    """
    Run Claude agent with MCP.

    Args:
        mcp_config: MCP server configuration
        system_prompt: System prompt for agent
        task: User task/query
        model: Claude model to use
        verbose: Print debug info

    Yields:
        Agent messages (text, tool use, tool results, etc.)
    """
    from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions

    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        model=model,
        mcp_servers=mcp_config,
        permission_mode="bypassPermissions",
        include_partial_messages=True,
    )

    if verbose:
        print(f"Using Claude model: {model}")

    async with ClaudeSDKClient(options=options) as client:
        await client.query(task)

        async for message in client.receive_response():
            # Yield message as-is
            yield message
