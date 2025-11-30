"""OpenAI provider using agents SDK."""

from typing import AsyncIterator
from pathlib import Path


async def run_openai(
    mcp_config: dict,
    system_prompt: str,
    task: str,
    model: str = "gpt-5-high",
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
    from agents import Agent, Runner
    from agents.mcp import MCPServerStdio
    from agents.model_settings import ModelSettings
    from pydantic import BaseModel, Field

    # Get Python from virtual environment
    venv_python = Path.cwd() / ".venv" / "bin" / "python"
    if not venv_python.exists():
        raise RuntimeError(f"Python not found at {venv_python}")

    # Extract MCP config
    notebooks_config = mcp_config.get("notebooks", {})

    # Create MCP server
    mcp_server = MCPServerStdio(
        name="notebooks",
        params={
            "command": str(venv_python),
            "args": ["-m", "scribe.notebook.notebook_mcp_server"],
            "env": notebooks_config.get("env", {}),
        },
        cache_tools_list=True,
        client_session_timeout_seconds=300,
    )

    if verbose:
        print(f"Using OpenAI model: {model}")

    # Define completion signal
    class TaskCompletion(BaseModel):
        """Signal that the task is complete."""
        status: str = Field(description="Must be exactly 'TASK_DONE'")
        summary: str = Field(description="Brief summary of what was accomplished")

    async with mcp_server:
        # Create agent
        agent = Agent(
            name="Assistant",
            instructions=system_prompt,
            mcp_servers=[mcp_server],
            model_settings=ModelSettings(model=model, timeout=300.0),
            output_type=TaskCompletion,
        )

        # Run agent
        result = Runner.run_streamed(agent, task, max_turns=1024)

        # Get tools for init message
        tools = await mcp_server.list_tools()
        yield {
            "subtype": "init",
            "data": {
                "mcp_servers": [{
                    "name": "notebooks",
                    "status": "connected",
                    "tools": [
                        {"name": tool.name, "description": tool.description}
                        for tool in tools
                    ],
                }]
            },
        }

        # Stream events
        async for event in result.stream_events():
            if verbose and event.type != "raw_response_event":
                print(f"[DEBUG] Event: {event.type}")

            if event.type == "run_item_stream_event":
                item = event.item
                if hasattr(item, 'type'):
                    if item.type == 'tool_call':
                        # Tool use
                        yield {
                            "type": "tool_use",
                            "id": getattr(item, 'id', ''),
                            "name": getattr(item, 'name', ''),
                            "input": getattr(item, 'arguments', {}),
                        }
                    elif item.type == 'tool_result':
                        # Tool result
                        yield {
                            "type": "tool_result",
                            "tool_use_id": getattr(item, 'tool_call_id', ''),
                            "content": getattr(item, 'output', ''),
                            "is_error": getattr(item, 'is_error', False),
                        }
                    elif item.type == 'message':
                        # Text message
                        text = getattr(item, 'content', '')
                        if text:
                            yield {"type": "text", "text": text}

        # Final status
        if result._stored_exception:
            yield {"subtype": "error", "error": str(result._stored_exception)}
        elif not result.is_complete:
            yield {
                "subtype": "error",
                "error": f"Agent stopped at turn {result.current_turn}/{result.max_turns}",
            }
        elif hasattr(result.final_output, 'status') and result.final_output.status == "TASK_DONE":
            yield {"subtype": "success"}
        else:
            yield {"subtype": "error", "error": "Invalid completion status"}
