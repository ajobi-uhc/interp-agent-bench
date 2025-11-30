"""Generic MCP server that wraps RPC endpoints.

This MCP server dynamically creates tools for each function exposed by
a ScopedSandbox's RPC endpoint. It runs locally and translates MCP tool
calls into HTTP requests to the remote RPC server.

Environment variables:
    RPC_URL: URL of the RPC endpoint (e.g., https://xyz.modal.run)
    FUNCTIONS: JSON list of function names to expose (e.g., ["chat", "reset"])

Usage:
    Called automatically by MCP client via:
    python -m interp_infra.mcp.interface_mcp_server
"""

import os
import json
import sys
import asyncio
from typing import Any

import requests


def main():
    """Run the MCP server."""
    # Import here to avoid issues if mcp not installed
    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
    except ImportError:
        print("Error: mcp package not installed. Install with: pip install mcp", file=sys.stderr)
        sys.exit(1)

    # Read configuration from environment
    rpc_url = os.environ.get("RPC_URL")
    functions_json = os.environ.get("FUNCTIONS")

    if not rpc_url:
        print("Error: RPC_URL environment variable not set", file=sys.stderr)
        sys.exit(1)

    if not functions_json:
        print("Error: FUNCTIONS environment variable not set", file=sys.stderr)
        sys.exit(1)

    try:
        functions = json.loads(functions_json)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid FUNCTIONS JSON: {e}", file=sys.stderr)
        sys.exit(1)

    # Create MCP server
    server = Server("interface")

    # Handler to list available tools
    @server.list_tools()
    async def list_tools():
        from mcp.types import Tool
        return [
            Tool(
                name=fn_name,
                description=f"Call {fn_name}() on the remote interface",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": True
                }
            )
            for fn_name in functions
        ]

    # Handler to call tools
    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list:
        from mcp.types import TextContent

        if name not in functions:
            raise ValueError(f"Unknown tool: {name}")

        try:
            resp = requests.post(
                rpc_url,
                json={"fn": name, "args": [], "kwargs": arguments},
                timeout=600
            )
            resp.raise_for_status()
            data = resp.json()

            if not data.get("ok"):
                error = data.get("error", "Unknown error")
                if "traceback" in data:
                    error += f"\n\nRemote traceback:\n{data['traceback']}"
                raise RuntimeError(error)

            result = data.get("result")

            # Convert result to string for MCP
            if isinstance(result, (dict, list)):
                result_str = json.dumps(result, indent=2)
            else:
                result_str = str(result)

            return [TextContent(type="text", text=result_str)]

        except requests.RequestException as e:
            raise RuntimeError(f"RPC request failed: {e}")

    # Run stdio server
    async def run():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options()
            )

    asyncio.run(run())


if __name__ == "__main__":
    main()
