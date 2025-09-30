"""Main CLI for scribe - clean click interface."""

import functools
import os
import subprocess
import sys
import time
from pathlib import Path

import click
from dotenv import load_dotenv

from scribe.cli.commands import copilot_impl
from scribe.providers._provider_utils import detect_available_providers
from scribe.cli.constants import ALL_PROVIDERS, DEFAULT_PROVIDER


@click.group(
    invoke_without_command=True,
    context_settings=dict(ignore_unknown_options=True, allow_extra_args=True)
)
@click.pass_context
def cli(ctx):
    """Scribe - AI-powered research assistant with notebook support.

    \b
    Usage:
      scribe                  # Launch with default provider (claude)
      scribe codex            # Launch with codex provider
      scribe claude           # Launch with claude provider
      scribe gemini           # Launch with gemini provider
      scribe <command>        # Run a specific command
    """
    # Load environment variables from .env file
    load_dotenv()

    # If no command is provided, default to copilot behavior
    if ctx.invoked_subcommand is None:
        # Get extra args that Click captured
        all_args = ctx.args[:]

        # Check if first arg is a provider
        if all_args and all_args[0] in ALL_PROVIDERS:
            provider = all_args[0]
            remaining_args = all_args[1:]
        else:
            provider = DEFAULT_PROVIDER
            remaining_args = all_args

        # Run copilot with the specified provider
        copilot_impl(remaining_args, provider_name=provider, verbose=False)


def main():
    """Entry point for the scribe command."""
    # Capture the Python interpreter path that the user would get from "which python"
    # This ensures all subprocess calls use the user's default Python environment
    if "SCRIBE_PYTHON" not in os.environ:
        try:
            result = subprocess.run(
                ["which", "python"], capture_output=True, text=True, check=True
            )
            python_path = result.stdout.strip()
            os.environ["SCRIBE_PYTHON"] = python_path
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback to sys.executable if "which python" fails
            os.environ["SCRIBE_PYTHON"] = sys.executable

    # Check if we need to handle special cases before Click processes args
    import sys
    args = sys.argv[1:]

    # If no args, first arg is a known command, or help requested, let Click handle normally
    if (not args or
        args[0] in cli.commands or
        args[0] in ['--help', '-h', 'help']):
        cli()
        return

    # If first arg is a provider, handle it specially
    if args[0] in ALL_PROVIDERS:
        provider = args[0]
        remaining_args = args[1:]
        load_dotenv()
        copilot_impl(remaining_args, provider_name=provider, verbose=False)
        return

    # Otherwise, treat all args as arguments to default provider
    load_dotenv()
    copilot_impl(args, provider_name=DEFAULT_PROVIDER, verbose=False)
