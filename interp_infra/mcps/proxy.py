"""Generic MCP server that wraps RPC endpoints."""

import os
import json
import sys
import asyncio

import requests


def main():
    """Run the MCP server."""
    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        from mcp.types import Tool, TextContent
    except ImportError:
        print("Error: mcp package not installed. Install with: pip install mcp", file=sys.stderr)
        sys.exit(1)

    rpc_url = os.environ.get("RPC_URL")
    functions_json = os.environ.get("FUNCTIONS")

    if not rpc_url or not functions_json:
        print("Error: RPC_URL and FUNCTIONS environment variables required", file=sys.stderr)
        sys.exit(1)

    try:
        functions = json.loads(functions_json)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid FUNCTIONS JSON: {e}", file=sys.stderr)
        sys.exit(1)

    # Fetch schemas from RPC server
    try:
        resp = requests.get(rpc_url, timeout=10)
        resp.raise_for_status()
        schemas = resp.json().get("schemas", {})
    except Exception as e:
        print(f"Warning: Could not fetch schemas from RPC server: {e}", file=sys.stderr)
        schemas = {}

    server = Server("interface")

    @server.list_tools()
    async def list_tools():
        return [
            Tool(
                name=fn_name,
                description=schemas.get(fn_name, {}).get("description", f"Call {fn_name}()"),
                inputSchema=schemas.get(fn_name, {}).get("inputSchema", {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": True
                })
            )
            for fn_name in functions
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list:
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
            result_str = json.dumps(result, indent=2) if isinstance(result, (dict, list)) else str(result)

            return [TextContent(type="text", text=result_str)]

        except requests.RequestException as e:
            raise RuntimeError(f"RPC request failed: {e}")

    async def run():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    asyncio.run(run())


if __name__ == "__main__":
    main()
