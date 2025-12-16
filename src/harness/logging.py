"""Centralized logging for agent runs using Python's logging library."""

import sys
import json
import logging
import os
from datetime import datetime
from typing import Optional, AsyncIterator
from contextlib import contextmanager

from rich.console import Console
from rich.logging import RichHandler
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text


# Global console for rich output
_console = Console(stderr=True)
_live_context: Optional[Live] = None

# Configure logging
LOG_LEVEL = os.environ.get("INTERP_LOG_LEVEL", "INFO").upper()
VALID_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
if LOG_LEVEL not in VALID_LEVELS:
    LOG_LEVEL = "INFO"

# Create logger
logger = logging.getLogger("interp_agent")
logger.setLevel(getattr(logging, LOG_LEVEL))
logger.propagate = False

# Remove existing handlers
logger.handlers.clear()

# Add rich handler with compact formatting
handler = RichHandler(
    console=_console,
    show_time=False,  # Hide timestamps for cleaner output
    show_level=False,  # Hide log level (INFO, DEBUG, etc.)
    show_path=False,
    markup=True,
    rich_tracebacks=True,
    tracebacks_show_locals=False,
)
handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(handler)


def get_logger(name: str = "interp_agent") -> logging.Logger:
    """Get a logger instance.

    All loggers are children of 'interp_agent' so they inherit the RichHandler.
    """
    if name == "interp_agent":
        return logging.getLogger(name)
    return logging.getLogger(f"interp_agent.{name}")


@contextmanager
def interactive_mode():
    """
    Context manager for interactive mode that uses rich.Live for better log/input separation.

    Usage:
        with interactive_mode():
            # Your interactive code here
            # Logs will be displayed separately from user input
    """
    global _live_context

    # Create layout for logs
    layout = Layout()
    layout.split_column(
        Layout(name="logs", ratio=1),
    )

    # Start live display
    _live_context = Live(
        layout,
        console=_console,
        refresh_per_second=10,
        screen=False,
    )

    with _live_context:
        yield _live_context

    _live_context = None


def _clean_tool_name(name: str) -> str:
    """Remove plugin prefixes from tool names."""
    return name.replace("mcp__", "").replace("plugin_modal-infra-plugin_scribe__", "")


def _truncate(text: str, max_length: int = 200) -> str:
    """Truncate text if too long."""
    if len(text) > max_length:
        return f"{text[:max_length]}... [truncated, {len(text)} total chars]"
    return text


def log_tool_call(name: str, tool_input: dict = None):
    """Log when a tool is called."""
    clean_name = _clean_tool_name(name)

    msg = f"\n🔧 [bold cyan]{clean_name}[/bold cyan]"
    logger.info(msg, extra={"markup": True})

    if tool_input:
        logger.info("  Input:")
        for key, value in tool_input.items():
            val_str = json.dumps(value, indent=2) if isinstance(value, (dict, list)) else str(value)
            val_str = _truncate(val_str, 200)
            logger.info(f"    {key}: {val_str}")

    logger.info("  ⏳ Running...")


def log_tool_result(result: str, is_error: bool = False):
    """Log tool result."""
    # Get max chars from environment variable, default to 2000
    max_chars_env = os.environ.get("INTERP_LOG_MAX_CHARS", "2000")
    try:
        default_max_chars = int(max_chars_env)
    except ValueError:
        default_max_chars = 2000

    if is_error:
        prefix = "  ✗ [red]Error[/red]"
        log_level = logging.ERROR
        max_lines = 20
        max_chars = default_max_chars
    else:
        prefix = "  ✓ [green]Result[/green]"
        log_level = logging.INFO
        max_lines = 30  # Show many more lines
        max_chars = default_max_chars

    logger.log(log_level, prefix, extra={"markup": True})

    lines = result.split("\n")[:max_lines]
    display = "\n".join(lines)
    truncated = _truncate(display, max_chars)

    for line in truncated.split("\n"):
        logger.log(log_level, f"    {line}")


def log_tool_done(elapsed_sec: float):
    """Log when a tool finishes."""
    logger.info(f"  ⏱  [dim]{elapsed_sec:.1f}s[/dim]", extra={"markup": True})


def log_tool_error(error: str):
    """Log when a tool errors."""
    logger.error("  ✗ [red bold]Error[/red bold]:", extra={"markup": True})
    for line in error.split("\n")[:10]:
        logger.error(f"    {line}")


def log_output(text: str):
    """
    Log agent text output to stdout (not stderr like other logs).
    This is the actual agent response that should be visible to the user.
    """
    # Ensure output is on a new line if needed
    print(text, end="", flush=True)


def print_start(provider: str, model: str):
    """Log session start."""
    logger.info(f"▶ [bold]Running {provider}[/bold] [dim]({model})[/dim]", extra={"markup": True})


def log_available_tools(tools: list):
    """Log available MCP tools at startup."""
    if not tools:
        return

    logger.info(f"Available tools: [cyan]{len(tools)}[/cyan]", extra={"markup": True})
    for tool in tools:
        name = tool.name if hasattr(tool, 'name') else str(tool)
        clean_name = _clean_tool_name(name)
        logger.info(f"  • {clean_name}")


def print_done():
    """Log session end."""
    logger.info("✓ [green]Done[/green]", extra={"markup": True})


def log_debug(message: str):
    """Log debug message."""
    logger.debug(message)


def log_info(message: str):
    """Log info message."""
    logger.info(message)


def log_warning(message: str):
    """Log warning message."""
    logger.warning(message)


def log_error(message: str):
    """Log error message."""
    logger.error(message)


def log_user_message(message: str, is_interrupt: bool = False):
    """Log user messages in interactive mode."""
    truncated = _truncate(message, 200)
    logger.info(f"💬 [bold]{truncated}[/bold]", extra={"markup": True})


async def run_agent_with_logging(
    agent_stream: AsyncIterator[dict],
) -> AsyncIterator[dict]:
    """
    Wrap agent execution with logging.

    Args:
        agent_stream: Async iterator of agent messages

    Yields:
        Agent messages (passed through from agent_stream)
    """
    tool_times = {}  # Track tool start times
    started = False

    async for message in agent_stream:
        # Handle init message
        if isinstance(message, dict) and message.get("type") == "init":
            provider = message.get("provider", "agent")
            model = message.get("model", "")
            tools = message.get("tools", [])

            print_start(provider, model)
            if tools:
                log_available_tools(tools)
            started = True
            yield message
            continue

        # Log based on message type
        if hasattr(message, 'content'):
            for block in message.content:
                # Tool call starting
                if hasattr(block, 'name') and hasattr(block, 'id'):
                    tool_times[block.id] = datetime.now()
                    tool_input = block.input if hasattr(block, 'input') else None
                    log_tool_call(block.name, tool_input)

                # Agent text output
                elif hasattr(block, 'text'):
                    log_output(block.text)

                # Tool result
                elif hasattr(block, 'tool_use_id'):
                    tool_id = block.tool_use_id
                    if tool_id in tool_times:
                        elapsed = (datetime.now() - tool_times[tool_id]).total_seconds()

                        # Get result content
                        result_content = ""
                        if hasattr(block, 'content'):
                            if isinstance(block.content, str):
                                result_content = block.content
                            elif isinstance(block.content, list):
                                # Extract text from content blocks
                                result_content = "\n".join(
                                    item.get("text", str(item)) if isinstance(item, dict) else str(item)
                                    for item in block.content
                                )

                        is_error = hasattr(block, 'is_error') and block.is_error

                        # Log result
                        log_tool_result(result_content, is_error)

                        # Log timing
                        log_tool_done(elapsed)

                        del tool_times[tool_id]

        # Yield message for custom processing
        yield message

    # Print done if we started
    if started:
        print_done()
