"""Simple, opinionated logging for agent runs."""

import sys
from datetime import datetime


def log_tool_call(name: str):
    """Print when a tool is called."""
    clean_name = name.replace("mcp__", "").replace("plugin_modal-infra-plugin_scribe__", "")
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] 🔧 {clean_name}", file=sys.stderr, flush=True)


def log_tool_done(elapsed_sec: float):
    """Print when a tool finishes."""
    print(f"         ✓ {elapsed_sec:.1f}s\n", file=sys.stderr, flush=True)


def log_tool_error(error: str):
    """Print when a tool errors."""
    first_line = error.split("\n")[0][:100]
    print(f"         ✗ {first_line}\n", file=sys.stderr, flush=True)


def log_output(text: str):
    """Print agent text output."""
    print(text, end="", flush=True)


def print_start(provider: str, model: str):
    """Print session start."""
    print(f"\n▶ Running {provider} ({model})\n", file=sys.stderr)


def print_done():
    """Print session end."""
    print(f"\n✓ Done\n", file=sys.stderr)
