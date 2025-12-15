"""
Target Agent - Manages the model being audited.

Handles:
  - OpenRouter API calls to target model
  - Conversation history and state
  - Tool definitions and execution
  - Message formatting for OpenAI API
"""

import os
import asyncio
import json
import inspect
from typing import Optional


class TargetAgent:
    """
    Manages conversation with a target model via OpenRouter.

    The auditor controls this agent through MCP tools, creating a realistic
    conversation scenario to test the target's behavior.
    """

    def __init__(self, model: str = "openai/gpt-5-mini"):
        self.model = model
        self.system_message: Optional[str] = None
        self.messages: list[dict] = []
        self.tools: list[dict] = []
        self.pending_tool_calls: list[dict] = []

        # OpenRouter API setup
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        self.api_base = "https://openrouter.ai/api/v1"

    # ========================================================================
    # Tool Management
    # ========================================================================

    def add_tool(self, name: str, code: str, environment: str):
        """
        Register a tool that the target can call.

        Parses Python function code to extract:
          - Function signature
          - Parameter types and defaults
          - Docstring

        Then builds an OpenAI function calling schema.

        Args:
            name: Tool name
            code: Python function definition
            environment: Hidden context (not shown to target)
        """
        # Execute code to get function object
        local_scope = {}
        try:
            exec(code, {}, local_scope)
        except Exception as e:
            raise ValueError(f"Failed to parse tool '{name}': {e}")

        # Find the function in executed scope
        func = None
        for item in local_scope.values():
            if callable(item) and not isinstance(item, type):
                func = item
                break

        if func is None:
            raise ValueError(f"No function found in code for tool '{name}'")

        # Build OpenAI tool schema
        tool_def = self._build_tool_schema(func, name)
        self.tools.append(tool_def)

    def _build_tool_schema(self, func: callable, name: str) -> dict:
        """Build OpenAI tool schema from Python function."""
        sig = inspect.signature(func)
        docstring = inspect.getdoc(func) or f"Function {name}"

        # Build parameter schema
        parameters = {"type": "object", "properties": {}, "required": []}

        for param_name, param in sig.parameters.items():
            # Map Python type to JSON schema type
            param_type = "string"  # default
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == int:
                    param_type = "integer"
                elif param.annotation == float:
                    param_type = "number"
                elif param.annotation == bool:
                    param_type = "boolean"
                elif param.annotation == list:
                    param_type = "array"
                elif param.annotation == dict:
                    param_type = "object"

            parameters["properties"][param_name] = {
                "type": param_type,
                "description": f"Parameter {param_name}"
            }

            # Mark as required if no default
            if param.default == inspect.Parameter.empty:
                parameters["required"].append(param_name)

        return {
            "type": "function",
            "function": {
                "name": name,
                "description": docstring,
                "parameters": parameters
            }
        }

    # ========================================================================
    # Conversation Flow
    # ========================================================================

    async def initialize(self, system_message: str, tools: list[dict]):
        """Initialize target with system prompt and tools."""
        self.system_message = system_message
        self.messages = []
        self.tools = tools
        self.pending_tool_calls = []

    async def send_message(self, message: str, prefill: str = "") -> dict:
        """
        Send user message to target.

        Args:
            message: User message content
            prefill: Optional text to steer target's response

        Returns:
            Dict with type "text" or "tool_calls"
        """
        # Add to history
        self.messages.append({"role": "user", "content": message})

        # Add prefill if provided (for response steering)
        if prefill and prefill.strip():
            self.messages.append({
                "role": "assistant",
                "content": prefill.strip(),
                "prefill": True  # Mark as prefill (not sent to API)
            })

        return await self._call_target()

    async def resume_with_tool_result(
        self,
        tool_call_id: str,
        result: str,
        status: str = "success",
        prefill: str = ""
    ) -> dict:
        """
        Resume target after simulating a tool result.

        Args:
            tool_call_id: ID of tool call to respond to
            result: Simulated result string
            status: "success" or "error" (for tracking)
            prefill: Optional response steering

        Returns:
            Dict with response or waiting status
        """
        # Add tool result to history
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result
        })

        # Remove from pending
        self.pending_tool_calls = [
            tc for tc in self.pending_tool_calls
            if tc["id"] != tool_call_id
        ]

        # If still waiting for more results, don't call API yet
        if self.pending_tool_calls:
            return {"type": "waiting", "pending": len(self.pending_tool_calls)}

        # Add prefill if provided
        if prefill and prefill.strip():
            self.messages.append({
                "role": "assistant",
                "content": prefill.strip(),
                "prefill": True
            })

        return await self._call_target()

    # ========================================================================
    # API Communication
    # ========================================================================

    async def _call_target(self) -> dict:
        """Call target model via OpenRouter API."""
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.api_base
            )

            # Build messages for API (filter prefills, format tool messages)
            api_messages = self._format_messages_for_api()

            # Prepare API call
            params = {
                "model": self.model,
                "messages": api_messages,
            }
            if self.tools:
                params["tools"] = self.tools

            # Call API
            response = await client.chat.completions.create(**params)
            message = response.choices[0].message

            # Handle tool calls
            if message.tool_calls:
                return self._process_tool_calls(message)

            # Handle text response
            text = message.content or ""
            self.messages.append({"role": "assistant", "content": text})
            return {"type": "text", "content": text}

        except Exception as e:
            return {"type": "text", "content": f"[Error: {e}]"}

    def _format_messages_for_api(self) -> list[dict]:
        """
        Format conversation history for OpenAI API.

        Handles:
          - System message
          - Filtering out prefills
          - Proper tool message formatting
        """
        api_messages = []

        # System message
        if self.system_message:
            api_messages.append({"role": "system", "content": self.system_message})

        # Conversation messages
        for msg in self.messages:
            # Skip prefills (not sent to API)
            if msg.get("prefill"):
                continue

            # Tool result messages
            if msg["role"] == "tool":
                api_messages.append({
                    "role": "tool",
                    "tool_call_id": msg["tool_call_id"],
                    "content": msg["content"]
                })

            # Assistant messages with tool calls
            elif msg["role"] == "assistant" and "tool_calls" in msg:
                api_messages.append({
                    "role": "assistant",
                    "content": msg.get("content"),
                    "tool_calls": msg["tool_calls"]
                })

            # Regular user/assistant messages
            else:
                api_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        return api_messages

    def _process_tool_calls(self, message) -> dict:
        """Extract and store tool calls from API response."""
        tool_calls = []

        for tc in message.tool_calls:
            # Parse arguments (may be string or dict)
            arguments = tc.function.arguments
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}

            tool_calls.append({
                "id": tc.id,
                "name": tc.function.name,
                "arguments": arguments
            })

        # Add to history (with proper formatting for future API calls)
        self.messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": (
                            tc.function.arguments if isinstance(tc.function.arguments, str)
                            else json.dumps(tc.function.arguments)
                        )
                    }
                }
                for tc in message.tool_calls
            ]
        })

        self.pending_tool_calls = tool_calls

        return {"type": "tool_calls", "tool_calls": tool_calls}

    # ========================================================================
    # State
    # ========================================================================

    def get_state(self) -> dict:
        """Get current conversation state."""
        return {
            "num_messages": len(self.messages),
            "num_tools": len(self.tools),
            "pending_tool_calls": len(self.pending_tool_calls),
        }
