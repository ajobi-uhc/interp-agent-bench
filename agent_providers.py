"""
Agent provider abstraction for Claude and OpenAI agents.

This module provides a unified interface for different agent SDKs (Claude, OpenAI)
to be used in run_agent.py with MCP server integration.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Optional


@dataclass
class AgentMessage:
    """Represents a message from the agent."""
    content: list[Any]
    usage: Optional[Any] = None
    subtype: Optional[str] = None
    data: Optional[dict] = None

    # Message metadata
    id: Optional[str] = None
    model: Optional[str] = None
    stop_reason: Optional[str] = None


@dataclass
class AgentOptions:
    """Configuration options for an agent."""
    system_prompt: str
    workspace_path: Path
    allowed_tools: list[str]
    mcp_config: dict
    stderr_callback: Any
    hooks: Optional[dict] = None
    verbose: bool = False
    agent_model: Optional[str] = None  # Override default model for the agent provider


class AgentProvider(ABC):
    """Abstract base class for agent providers."""

    @abstractmethod
    async def __aenter__(self):
        """Enter async context."""
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context."""
        pass

    @abstractmethod
    async def query(self, prompt: str) -> None:
        """Send a query to the agent."""
        pass

    @abstractmethod
    async def receive_response(self) -> AsyncIterator[AgentMessage]:
        """Receive streaming response from the agent."""
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the provider name (e.g., 'claude', 'openai')."""
        pass


class ClaudeAgentProvider(AgentProvider):
    """Claude agent provider using claude-agent-sdk."""

    def __init__(self, options: AgentOptions):
        from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
        import time

        self.options = options
        self._tool_start_times = {}  # Track tool call start times

        # Determine which model to use (allow override from config)
        model = options.agent_model or "claude-sonnet-4-5-20250929"

        # Build Claude-specific options
        claude_options = ClaudeAgentOptions(
            system_prompt=options.system_prompt,
            model=model,
            mcp_servers=options.mcp_config,
            permission_mode="bypassPermissions",
            add_dirs=[str(options.workspace_path)],
            allowed_tools=options.allowed_tools,
            include_partial_messages=True,
            stderr=options.stderr_callback,
            hooks=options.hooks or {},
        )

        print(f"Using model: {claude_options.model}")
        self.client = ClaudeSDKClient(options=claude_options)

    async def __aenter__(self):
        await self.client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.__aexit__(exc_type, exc_val, exc_tb)

    async def query(self, prompt: str) -> None:
        await self.client.query(prompt)

    async def receive_response(self) -> AsyncIterator[AgentMessage]:
        """Receive response from Claude SDK."""
        import time

        async for message in self.client.receive_response():
            # Track tool timing
            content = getattr(message, 'content', [])
            for block in content:
                # Check if it's a tool use block
                if hasattr(block, 'id') and hasattr(block, 'name'):
                    # Tool use - record start time
                    self._tool_start_times[block.id] = time.time()
                # Check if it's a tool result block
                elif hasattr(block, 'tool_use_id'):
                    # Tool result - log elapsed time
                    tool_id = block.tool_use_id
                    if tool_id in self._tool_start_times:
                        elapsed = time.time() - self._tool_start_times[tool_id]
                        print(f"  Execution time: {elapsed:.2f}s")
                        del self._tool_start_times[tool_id]

            # Convert Claude SDK message to AgentMessage
            agent_msg = AgentMessage(
                content=content,
                usage=getattr(message, 'usage', None),
                subtype=getattr(message, 'subtype', None),
                data=getattr(message, 'data', None),
                id=getattr(message, 'id', None),
                model=getattr(message, 'model', None),
                stop_reason=getattr(message, 'stop_reason', None),
            )
            yield agent_msg

    def get_provider_name(self) -> str:
        return "claude"


class OpenAIAgentProvider(AgentProvider):
    """OpenAI agent provider using openai agents SDK."""

    def __init__(self, options: AgentOptions):
        from agents import Agent, Runner
        from agents.mcp import MCPServerStdio

        self.options = options
        self.runner = None
        self.current_prompt = None

        # Get Python from virtual environment
        venv_python = Path.cwd() / ".venv" / "bin" / "python"
        if not venv_python.exists():
            raise RuntimeError(f"Python not found at {venv_python}")

        # Extract MCP server config (assume single server for now)
        # The mcp_config comes in Claude format, convert to OpenAI format
        notebooks_config = options.mcp_config.get("notebooks", {})

        # Create MCPServerStdio instance
        self.mcp_server = MCPServerStdio(
            name="notebooks",
            params={
                "command": str(venv_python),
                "args": ["-m", "scribe.notebook.notebook_mcp_server"],
                "env": notebooks_config.get("env", {}),
            },
            cache_tools_list=True,
            client_session_timeout_seconds=3000,  # 2 minutes for code execution
        )

        # Create OpenAI agent (will be initialized in __aenter__)
        self.agent = None

    async def __aenter__(self):
        from agents import Agent
        from agents.model_settings import ModelSettings
        from pydantic import BaseModel, Field

        # Start MCP server
        await self.mcp_server.__aenter__()

        # Define output type that requires explicit completion signal
        class TaskCompletion(BaseModel):
            """Signal that the task is complete. Only return this when truly done."""
            status: str = Field(
                description="Must be exactly 'TASK_DONE'"
            )
            summary: str = Field(
                description="Brief summary of what was accomplished"
            )

        # Determine which model to use (allow override from config)
        model = self.options.agent_model or "gpt-5-high"  # Default: equivalent to Claude 4.5 Sonnet

        # Create agent with MCP server
        self.agent = Agent(
            name="Assistant",
            instructions=self.options.system_prompt,
            mcp_servers=[self.mcp_server],
            model_settings=ModelSettings(
                model=model,
                timeout=300.0,  # 5 minute timeout for long-running operations
            ),
            output_type=TaskCompletion,  # Agent must return this structure to complete
        )

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Clean up MCP server
        await self.mcp_server.__aexit__(exc_type, exc_val, exc_tb)

    async def query(self, prompt: str) -> None:
        """Store the prompt for processing in receive_response."""
        self.current_prompt = prompt

    async def receive_response(self) -> AsyncIterator[AgentMessage]:
        """Receive response from OpenAI agent."""
        from agents import Runner

        if not self.current_prompt:
            return

        # Run the agent with streaming (set high max_turns to avoid premature termination)
        result = Runner.run_streamed(self.agent, self.current_prompt, max_turns=1024)

        # Track if we've sent init message
        sent_init = False

        # Stream events
        async for event in result.stream_events():
            # Send init message on first event
            if not sent_init:
                # Get tool list from MCP server
                tools = await self.mcp_server.list_tools()

                yield AgentMessage(
                    content=[],
                    subtype="init",
                    data={
                        "mcp_servers": [{
                            "name": "notebooks",
                            "status": "connected",
                            "tools": [
                                {"name": tool.name, "description": tool.description}
                                for tool in tools
                            ],
                        }]
                    },
                )
                sent_init = True

            # Debug: Print event details (skip noisy response events) - only in verbose mode
            if self.options.verbose and event.type != "raw_response_event":
                print(f"\n[DEBUG OpenAI] ========== EVENT ==========", flush=True)
                print(f"[DEBUG OpenAI] Event type: {event.type}", flush=True)
                
                if hasattr(event, 'item'):
                    print(f"[DEBUG OpenAI] Item type: {getattr(event.item, 'type', 'NO TYPE')}", flush=True)
                    # Only print first 200 chars to avoid spam
                    item_str = str(event.item)
                    if len(item_str) > 200:
                        print(f"[DEBUG OpenAI] Item preview: {item_str[:200]}...", flush=True)
                    else:
                        print(f"[DEBUG OpenAI] Item: {item_str}", flush=True)
                
                if hasattr(event, 'delta') and event.delta:
                    print(f"[DEBUG OpenAI] Delta: {event.delta}", flush=True)
                
                print(f"[DEBUG OpenAI] ================================\n", flush=True)

            # Process stream events
            if event.type == "run_item_stream_event":
                # Map OpenAI event to AgentMessage format
                item = event.item

                # Check if it's a tool call or text
                if hasattr(item, 'type'):
                    if item.type == 'tool_call':
                        # Tool use
                        from claude_agent_sdk import ToolUseBlock

                        tool_block = type('ToolUseBlock', (), {
                            'id': getattr(item, 'id', ''),
                            'name': getattr(item, 'name', ''),
                            'input': getattr(item, 'arguments', {}),
                        })()

                        yield AgentMessage(content=[tool_block])

                    elif item.type == 'tool_result':
                        # Tool result
                        from claude_agent_sdk import ToolResultBlock

                        result_block = type('ToolResultBlock', (), {
                            'tool_use_id': getattr(item, 'tool_call_id', ''),
                            'content': getattr(item, 'output', ''),
                            'is_error': getattr(item, 'is_error', False),
                        })()

                        yield AgentMessage(content=[result_block])

                    elif item.type == 'message':
                        # Text message
                        text = getattr(item, 'content', '')
                        if text:
                            from claude_agent_sdk import TextBlock

                            text_block = type('TextBlock', (), {'text': text})()
                            yield AgentMessage(content=[text_block])

        # After streaming is complete, check agent completion status
        if self.options.verbose:
            print(f"\n[DEBUG OpenAI] ========== STREAM ENDED ==========", flush=True)
            print(f"[DEBUG OpenAI] result.is_complete: {result.is_complete}", flush=True)
            print(f"[DEBUG OpenAI] result.current_turn: {result.current_turn}", flush=True)
            print(f"[DEBUG OpenAI] result.max_turns: {result.max_turns}", flush=True)
            print(f"[DEBUG OpenAI] result._stored_exception: {result._stored_exception}", flush=True)
            print(f"[DEBUG OpenAI] result.final_output: {str(result.final_output)[:200]}", flush=True)
            print(f"[DEBUG OpenAI] ====================================\n", flush=True)
        
        # Extract token usage from the result object
        if hasattr(result, 'raw_responses') and result.raw_responses:
            # Get the last response for usage info
            last_response = result.raw_responses[-1]
            if hasattr(last_response, 'usage'):
                usage_obj = type('Usage', (), {
                    'input_tokens': getattr(last_response.usage, 'prompt_tokens', 0),
                    'output_tokens': getattr(last_response.usage, 'completion_tokens', 0),
                })()

                yield AgentMessage(
                    content=[],
                    usage=usage_obj,
                    model=getattr(last_response, 'model', None),
                )

        # Check completion status and send appropriate message
        if result._stored_exception:
            # Agent hit an exception
            error_msg = str(result._stored_exception)
            if self.options.verbose:
                print(f"[DEBUG OpenAI] Sending ERROR message: {error_msg}", flush=True)
            yield AgentMessage(
                content=[],
                subtype="error",
                data={"error": error_msg},
            )
        elif not result.is_complete:
            # Agent stopped but didn't complete
            warning_msg = f"Agent stopped prematurely at turn {result.current_turn}/{result.max_turns} without reaching completion"
            if self.options.verbose:
                print(f"[DEBUG OpenAI] Sending ERROR message: {warning_msg}", flush=True)
            yield AgentMessage(
                content=[],
                subtype="error",
                data={"error": warning_msg},
            )
        else:
            # Check if agent returned proper completion structure
            if hasattr(result.final_output, 'status'):
                if result.final_output.status == "TASK_DONE":
                    if self.options.verbose:
                        print(f"[DEBUG OpenAI] Sending SUCCESS message (valid TASK_DONE)", flush=True)
                    yield AgentMessage(
                        content=[],
                        subtype="success",
                    )
                else:
                    error_msg = f"Agent completed with invalid status: '{result.final_output.status}' (expected 'TASK_DONE')"
                    if self.options.verbose:
                        print(f"[DEBUG OpenAI] Sending ERROR message: {error_msg}", flush=True)
                    yield AgentMessage(
                        content=[],
                        subtype="error",
                        data={"error": error_msg},
                    )
            else:
                # Shouldn't happen with output_type set, but check anyway
                error_msg = f"Agent completed without proper output structure"
                if self.options.verbose:
                    print(f"[DEBUG OpenAI] Sending ERROR message: {error_msg}", flush=True)
                yield AgentMessage(
                    content=[],
                    subtype="error",
                    data={"error": error_msg},
                )

    def get_provider_name(self) -> str:
        return "openai"


def create_agent_provider(
    provider_name: str,
    options: AgentOptions,
) -> AgentProvider:
    """
    Factory function to create an agent provider.

    Args:
        provider_name: Name of the provider ('claude' or 'openai')
        options: Agent configuration options

    Returns:
        AgentProvider instance

    Raises:
        ValueError: If provider_name is not supported
    """
    providers = {
        "claude": ClaudeAgentProvider,
        "openai": OpenAIAgentProvider,
    }
    print(f"🤖 Agent Provider: {provider_name}")
    if provider_name not in providers:
        raise ValueError(
            f"Unknown agent provider: {provider_name}. "
            f"Supported: {', '.join(providers.keys())}"
        )

    return providers[provider_name](options)
