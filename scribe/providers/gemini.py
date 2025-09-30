"""
Gemini CLI provider implementation.

Docs on configuring Gemini: https://github.com/google-gemini/gemini-cli/blob/main/docs/cli/configuration.md
"""

import subprocess
import os
from typing import List

from .base import AICLIProvider


def get_copilot_settings(python_path: str) -> dict:
    # Gemini-specific settings
    # Note: Unlike Claude, Gemini doesn't use permissions/tools allow lists
    # MCP servers are configured directly in mcpServers section
    config = {
        # Empty by default - mcpServers will be added dynamically
        "hideBanner": True,
        "theme": "Ayu",
        "usageStatisticsEnabled": False,
        "autoAccept": True,
        "mcpServers": {
            "scribe": {
                "command": python_path,
                "args": ["-m", "scribe.notebook.notebook_mcp_server"],
                "env": {
                    "SCRIBE_PROVIDER": "gemini",
                },
                "trust": True,
            }
        },
    }

    return config


class GeminiProvider(AICLIProvider):
    """Gemini CLI provider."""

    def get_provider_name(self) -> str:
        return "gemini"

    def get_provider_display_name(self) -> str:
        return "Gemini CLI"

    def get_command_base(self) -> List[str]:
        # Check if global gemini is available, fallback to npx
        try:
            subprocess.run(
                ["which", "gemini"], capture_output=True, check=True, timeout=3
            )
            return ["gemini"]
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ):
            return ["npx", "@google/gemini-cli"]


    def is_available(self) -> bool:
        # Try global gemini first
        try:
            result = subprocess.run(
                ["which", "gemini"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode == 0:
                return True
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ):
            pass

        # Try npx version
        try:
            result = subprocess.run(
                ["npx", "@google/gemini-cli", "--version"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            return result.returncode == 0
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ):
            return False
