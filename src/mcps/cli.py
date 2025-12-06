"""CLI MCP Server - Exposes sandbox shell/Python execution as MCP tools."""

import os
from fastmcp import FastMCP

mcp = FastMCP("cli")

# Get sandbox connection from environment
SANDBOX_ID = os.environ.get("SANDBOX_ID")

if not SANDBOX_ID:
    raise RuntimeError("SANDBOX_ID environment variable required")

# Import Modal and get sandbox reference
import modal

app = modal.App.lookup(SANDBOX_ID)
sandbox = modal.Sandbox.from_id(SANDBOX_ID)


@mcp.tool()
def run_command(command: str) -> str:
    """
    Execute a shell command in the sandbox.

    Args:
        command: Shell command to run

    Returns:
        stdout from the command
    """
    result = sandbox.exec("bash", "-c", command)
    return result.stdout.strip()


@mcp.tool()
def run_python(code: str) -> str:
    """
    Execute Python code in the sandbox.

    Args:
        code: Python code to execute

    Returns:
        stdout from execution
    """
    result = sandbox.exec("python", "-c", code)
    return result.stdout.strip()


@mcp.tool()
def read_file(path: str) -> str:
    """
    Read a file from the sandbox.

    Args:
        path: Absolute path to file in sandbox

    Returns:
        File contents
    """
    result = sandbox.exec("cat", path)
    return result.stdout


@mcp.tool()
def write_file(path: str, content: str) -> str:
    """
    Write content to a file in the sandbox.

    Args:
        path: Absolute path to file in sandbox
        content: Content to write

    Returns:
        Success message
    """
    # Use heredoc for reliable multi-line content
    result = sandbox.exec("bash", "-c", f"cat > {path} << 'EOFMARKER'\n{content}\nEOFMARKER")
    return f"Wrote {len(content)} bytes to {path}"


@mcp.tool()
def list_files(path: str = "/workspace") -> str:
    """
    List files in a directory.

    Args:
        path: Directory path (default: /workspace)

    Returns:
        ls -la output
    """
    result = sandbox.exec("ls", "-la", path)
    return result.stdout
