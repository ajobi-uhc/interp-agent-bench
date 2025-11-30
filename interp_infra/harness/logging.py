"""Simple, opinionated logging for agent runs."""

import sys
import json
from datetime import datetime


def log_tool_call(name: str, tool_input: dict = None):
    """Print when a tool is called with full input."""
    clean_name = name.replace("mcp__", "").replace("plugin_modal-infra-plugin_scribe__", "")
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{timestamp}] 🔧 {clean_name}", file=sys.stderr, flush=True)

    if tool_input:
        print(f"         Input:", file=sys.stderr, flush=True)
        # Pretty print the input
        for key, value in tool_input.items():
            if isinstance(value, str) and len(value) > 200:
                # Truncate long strings
                print(f"           {key}: {value[:200]}... ({len(value)} chars)", file=sys.stderr, flush=True)
            elif isinstance(value, dict) or isinstance(value, list):
                print(f"           {key}: {json.dumps(value, indent=2)[:200]}...", file=sys.stderr, flush=True)
            else:
                print(f"           {key}: {value}", file=sys.stderr, flush=True)


def log_tool_result(result: str, is_error: bool = False):
    """Print tool result."""
    if is_error:
        print(f"         ✗ Error:", file=sys.stderr, flush=True)
        # Show first 500 chars of error
        lines = result.split("\n")[:10]  # First 10 lines
        for line in lines:
            print(f"           {line}", file=sys.stderr, flush=True)
    else:
        print(f"         ✓ Result:", file=sys.stderr, flush=True)
        # Show first 300 chars of result
        if len(result) > 300:
            print(f"           {result[:300]}... ({len(result)} chars)", file=sys.stderr, flush=True)
        else:
            for line in result.split("\n")[:5]:  # First 5 lines
                print(f"           {line}", file=sys.stderr, flush=True)


def log_tool_done(elapsed_sec: float):
    """Print when a tool finishes."""
    print(f"         ⏱ {elapsed_sec:.1f}s\n", file=sys.stderr, flush=True)


def log_tool_error(error: str):
    """Print when a tool errors."""
    print(f"         ✗ Error:", file=sys.stderr, flush=True)
    lines = error.split("\n")[:10]
    for line in lines:
        print(f"           {line}", file=sys.stderr, flush=True)
    print(file=sys.stderr, flush=True)


def log_output(text: str):
    """Print agent text output."""
    print(text, end="", flush=True)


def print_start(provider: str, model: str):
    """Print session start."""
    print(f"\n▶ Running {provider} ({model})\n", file=sys.stderr)


def print_done():
    """Print session end."""
    print(f"\n✓ Done\n", file=sys.stderr)
