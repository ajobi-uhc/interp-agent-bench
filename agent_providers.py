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

        self.options = options

        # Build Claude-specific options
        claude_options = ClaudeAgentOptions(
            system_prompt=options.system_prompt,
            mcp_servers=options.mcp_config,
            permission_mode="bypassPermissions",
            add_dirs=[str(options.workspace_path)],
            allowed_tools=options.allowed_tools,
            include_partial_messages=True,
            stderr=options.stderr_callback,
            hooks=options.hooks or {},
        )

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
        async for message in self.client.receive_response():
            # Convert Claude SDK message to AgentMessage
            agent_msg = AgentMessage(
                content=getattr(message, 'content', []),
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
            client_session_timeout_seconds=800,  # 2 minutes for code execution
        )

        # Create OpenAI agent (will be initialized in __aenter__)
        self.agent = None

    async def __aenter__(self):
        from agents import Agent
        from agents.model_settings import ModelSettings

        # Start MCP server
        await self.mcp_server.__aenter__()

        # Create agent with MCP server
        self.agent = Agent(
            name="Assistant",
            instructions=self.options.system_prompt,
            mcp_servers=[self.mcp_server],
            model_settings=ModelSettings(
                # Use default model (can be overridden via environment)
                model="gpt-4o",
            ),
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

        # Run the agent with streaming
        result = Runner.run_streamed(self.agent, self.current_prompt)

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

        # Get final result with token usage
        final_result = await result

        # Extract token usage from OpenAI format
        if hasattr(final_result, 'usage'):
            usage_obj = type('Usage', (), {
                'input_tokens': getattr(final_result.usage, 'prompt_tokens', 0),
                'output_tokens': getattr(final_result.usage, 'completion_tokens', 0),
            })()

            yield AgentMessage(
                content=[],
                usage=usage_obj,
                model=getattr(final_result, 'model', None),
            )

        # Send success message
        yield AgentMessage(
            content=[],
            subtype="success",
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

    if provider_name not in providers:
        raise ValueError(
            f"Unknown agent provider: {provider_name}. "
            f"Supported: {', '.join(providers.keys())}"
        )

    return providers[provider_name](options)
