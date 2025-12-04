"""Claude provider using claude-agent-sdk."""

from typing import AsyncIterator


async def run_claude(
    mcp_config: dict,
    system_prompt: str,
    task: str,
    model: str = "claude-sonnet-4-5-20250929",
    kwargs={},
) -> AsyncIterator[dict]:
    """
    Run Claude agent with MCP.

    Args:
        mcp_config: MCP server configuration
        system_prompt: System prompt for agent
        task: User task/query
        model: Claude model to use (can be overridden in kwargs)
        kwargs: Additional options passed to ClaudeAgentOptions

    Yields:
        Agent messages (text, tool use, tool results, etc.)
    """
    from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions

    # Build options dict - kwargs take precedence over defaults
    options_dict = {
        "system_prompt": system_prompt,
        "model": model,
        "mcp_servers": mcp_config,
        "permission_mode": "bypassPermissions",
    }

    # Merge with user kwargs (user values override defaults)
    options_dict.update(kwargs)

    options = ClaudeAgentOptions(**options_dict)

    async with ClaudeSDKClient(options=options) as client:
        # Yield init message with tools info
        if hasattr(client, 'tools') and client.tools:
            yield {
                "type": "init",
                "provider": "claude",
                "model": options_dict.get('model', model),
                "tools": [{"name": getattr(tool, 'name', str(tool))} for tool in client.tools],
            }

        await client.query(task)

        async for message in client.receive_response():
            # Yield message as-is for custom processing
            yield message
