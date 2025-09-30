"""Claude CLI provider implementation."""

import subprocess
import os

from scribe.providers.base import AICLIProvider


# Claude-specific settings
CLAUDE_COPILOT_SETTINGS = {
    "permissions": {
        "allow": [
            "mcp__scribe__start_new_session",
            "mcp__scribe__execute_code",
            "mcp__scribe__add_markdown",
            "mcp__scribe__edit_cell",
        ]
    },
    "enableAllProjectMcpServers": True,
    "enabledMcpjsonServers": ["scribe"],
}


class ClaudeProvider(AICLIProvider):
    """Claude CLI provider"""

    def get_provider_name(self) -> str:
        return "claude"

    def get_copilot_mcp_config(self, python_path: str) -> dict:
        config = {
            "mcpServers": {
                "scribe": {
                    "command": python_path,
                    "args": ["-m", "scribe.notebook.notebook_mcp_server"],
                    "env": {
                        # Location of notebook outputs - only include if set to avoid "null" string
                        **(
                            {}
                            if os.environ.get("NOTEBOOK_OUTPUT_DIR") is None
                            else {
                                "NOTEBOOK_OUTPUT_DIR": os.environ.get(
                                    "NOTEBOOK_OUTPUT_DIR"
                                )
                            }
                        )
                    },
                }
            }
        }

        return config

    def get_provider_display_name(self) -> str:
        return "Claude Code"

    def is_available(self) -> bool:
        try:
            result = subprocess.run(
                ["which", "claude"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            return result.returncode == 0
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ):
            return False
