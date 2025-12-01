"""Simple logging for agent runs."""

import sys
import json
from datetime import datetime


def log_tool_call(name: str, tool_input: dict = None):
    """Print when a tool is called."""
    clean_name = name.replace("mcp__", "").replace("plugin_modal-infra-plugin_scribe__", "")
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{timestamp}] 🔧 {clean_name}", file=sys.stderr, flush=True)

    if tool_input:
        print("Input:", file=sys.stderr, flush=True)
        for key, value in tool_input.items():
            val_str = json.dumps(value, indent=2) if isinstance(value, (dict, list)) else str(value)
            if len(val_str) > 200:
                val_str = val_str[:200] + f"... ({len(val_str)} chars)"
            print(f"  {key}: {val_str}", file=sys.stderr, flush=True)


def log_tool_result(result: str, is_error: bool = False):
    """Print tool result."""
    prefix = "✗ Error" if is_error else "✓ Result"
    print(f"{prefix}:", file=sys.stderr, flush=True)

    lines = result.split("\n")[:10 if is_error else 5]
    display = "\n".join(lines)

    if len(result) > 300:
        print(f"  {display[:300]}... ({len(result)} chars)", file=sys.stderr, flush=True)
    else:
        for line in lines:
            print(f"  {line}", file=sys.stderr, flush=True)


def log_tool_done(elapsed_sec: float):
    """Print when a tool finishes."""
    print(f"⏱ {elapsed_sec:.1f}s\n", file=sys.stderr, flush=True)


def log_tool_error(error: str):
    """Print when a tool errors."""
    print("✗ Error:", file=sys.stderr, flush=True)
    for line in error.split("\n")[:10]:
        print(f"  {line}", file=sys.stderr, flush=True)
    print(file=sys.stderr, flush=True)


def log_output(text: str):
    """Print agent text output."""
    print(text, end="", flush=True)


def print_start(provider: str, model: str):
    """Print session start."""
    print(f"\n▶ Running {provider} ({model})\n", file=sys.stderr)


def log_available_tools(tools: list):
    """Log available MCP tools at startup."""
    if not tools:
        return

    print(f"Available tools: {len(tools)}", file=sys.stderr)
    for tool in tools:
        name = tool.name if hasattr(tool, 'name') else str(tool)
        clean_name = name.replace("mcp__", "").replace("plugin_modal-infra-plugin_scribe__", "")
        print(f"  • {clean_name}", file=sys.stderr)
    print(file=sys.stderr)


def print_done():
    """Print session end."""
    print("\n✓ Done\n", file=sys.stderr)
