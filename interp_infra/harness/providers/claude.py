"""Claude provider using claude-agent-sdk."""

from typing import AsyncIterator
from datetime import datetime

from ..logging import log_tool_call, log_tool_done, log_tool_error, log_output, print_start, print_done


async def run_claude(
    mcp_config: dict,
    system_prompt: str,
    task: str,
    model: str = "claude-sonnet-4-5-20250929",
) -> AsyncIterator[dict]:
    """
    Run Claude agent with MCP.

    Args:
        mcp_config: MCP server configuration
        system_prompt: System prompt for agent
        task: User task/query
        model: Claude model to use

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

    print_start("Claude", model)

    tool_times = {}  # Track tool start times

    async with ClaudeSDKClient(options=options) as client:
        await client.query(task)

        async for message in client.receive_response():
            # Log based on message type
            if hasattr(message, 'content'):
                for block in message.content:
                    # Tool call starting
                    if hasattr(block, 'name') and hasattr(block, 'id'):
                        tool_times[block.id] = datetime.now()
                        log_tool_call(block.name)

                    # Agent text output
                    elif hasattr(block, 'text'):
                        log_output(block.text)

                    # Tool result
                    elif hasattr(block, 'tool_use_id'):
                        tool_id = block.tool_use_id
                        if tool_id in tool_times:
                            elapsed = (datetime.now() - tool_times[tool_id]).total_seconds()
                            if hasattr(block, 'is_error') and block.is_error:
                                log_tool_error(str(block.content))
                            else:
                                log_tool_done(elapsed)
                            del tool_times[tool_id]

            # Yield message as-is for custom processing
            yield message

    print_done()
