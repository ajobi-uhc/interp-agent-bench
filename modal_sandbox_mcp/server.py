"""
Modal Sandbox MCP Server - Model Context Protocol interface for managing Modal sandboxes.

This MCP server provides tools to:
- List running sandboxes
- Execute commands in sandboxes
- Get sandbox status and logs
- Terminate sandboxes
- Create filesystem snapshots

The server uses modal.Sandbox.from_id() to reconnect to existing sandboxes,
so it doesn't need to maintain state between calls.
"""

import logging
import os
from typing import Optional, List, Dict, Any

import modal
from fastmcp import FastMCP

# Setup logging
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger("modal_sandbox_mcp")

# Initialize MCP server
mcp = FastMCP("modal-sandbox")

# Default app name for sandbox operations
DEFAULT_APP_NAME = os.environ.get("MODAL_APP_NAME", "sandbox")


def get_sandbox(sandbox_id: str) -> modal.Sandbox:
    """Reconnect to a sandbox by ID."""
    return modal.Sandbox.from_id(sandbox_id)


@mcp.tool()
def list_sandboxes(app_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List all running Modal sandboxes.

    Args:
        app_name: Optional app name to filter by. Defaults to "sandbox".

    Returns:
        List of sandbox info dicts with keys: sandbox_id
    """
    app_name = app_name or DEFAULT_APP_NAME

    try:
        app = modal.App.lookup(app_name)
        sandboxes = []

        for sb in modal.Sandbox.list(app_id=app.app_id):
            sandboxes.append({
                "sandbox_id": sb.object_id,
            })

        return sandboxes
    except Exception as e:
        logger.error(f"Failed to list sandboxes: {e}")
        return []


@mcp.tool()
def get_sandbox_info(sandbox_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a sandbox.

    Args:
        sandbox_id: The Modal sandbox ID (e.g., "sb-xxx...")

    Returns:
        Dict with sandbox info including tunnels, status, etc.
    """
    try:
        sb = get_sandbox(sandbox_id)

        # Get tunnel URLs
        tunnels = {}
        try:
            for port, tunnel in sb.tunnels().items():
                tunnels[port] = tunnel.url
        except Exception:
            pass

        return {
            "sandbox_id": sandbox_id,
            "tunnels": tunnels,
            "status": "running",
        }
    except Exception as e:
        logger.error(f"Failed to get sandbox info: {e}")
        return {
            "sandbox_id": sandbox_id,
            "error": str(e),
            "status": "unknown",
        }


@mcp.tool()
def exec_in_sandbox(
    sandbox_id: str,
    command: str,
    timeout: int = 60
) -> Dict[str, Any]:
    """
    Execute a shell command in a running sandbox.

    Args:
        sandbox_id: The Modal sandbox ID
        command: Shell command to execute
        timeout: Command timeout in seconds (default: 60)

    Returns:
        Dict with stdout, stderr, and return_code
    """
    try:
        sb = get_sandbox(sandbox_id)

        p = sb.exec("bash", "-c", command, timeout=timeout)
        stdout = p.stdout.read()
        stderr = p.stderr.read()
        p.wait()

        return {
            "stdout": stdout,
            "stderr": stderr,
            "return_code": p.returncode,
            "success": p.returncode == 0,
        }
    except Exception as e:
        logger.error(f"Failed to exec in sandbox: {e}")
        return {
            "stdout": "",
            "stderr": str(e),
            "return_code": -1,
            "success": False,
            "error": str(e),
        }


@mcp.tool()
def exec_python_in_sandbox(
    sandbox_id: str,
    code: str,
    timeout: int = 60
) -> Dict[str, Any]:
    """
    Execute Python code in a running sandbox.

    Args:
        sandbox_id: The Modal sandbox ID
        code: Python code to execute
        timeout: Execution timeout in seconds (default: 60)

    Returns:
        Dict with stdout, stderr, and return_code
    """
    try:
        sb = get_sandbox(sandbox_id)

        p = sb.exec("python", "-c", code, timeout=timeout)
        stdout = p.stdout.read()
        stderr = p.stderr.read()
        p.wait()

        return {
            "stdout": stdout,
            "stderr": stderr,
            "return_code": p.returncode,
            "success": p.returncode == 0,
        }
    except Exception as e:
        logger.error(f"Failed to exec python in sandbox: {e}")
        return {
            "stdout": "",
            "stderr": str(e),
            "return_code": -1,
            "success": False,
            "error": str(e),
        }


@mcp.tool()
def terminate_sandbox(sandbox_id: str) -> Dict[str, Any]:
    """
    Terminate a running sandbox.

    Args:
        sandbox_id: The Modal sandbox ID to terminate

    Returns:
        Dict with success status and message
    """
    try:
        sb = get_sandbox(sandbox_id)
        sb.terminate()

        return {
            "success": True,
            "message": f"Sandbox {sandbox_id} terminated successfully",
        }
    except Exception as e:
        logger.error(f"Failed to terminate sandbox: {e}")
        return {
            "success": False,
            "message": str(e),
            "error": str(e),
        }


@mcp.tool()
def snapshot_sandbox(
    sandbox_id: str,
    description: str = ""
) -> Dict[str, Any]:
    """
    Create a filesystem snapshot of the sandbox's current state.

    The snapshot captures all filesystem changes and can be used to
    restore the sandbox state later.

    Args:
        sandbox_id: The Modal sandbox ID
        description: Optional description for the snapshot

    Returns:
        Dict with snapshot info (image_id can be used with Sandbox.from_snapshot)
    """
    try:
        sb = get_sandbox(sandbox_id)
        image = sb.snapshot_filesystem()

        return {
            "success": True,
            "message": f"Snapshot created{f': {description}' if description else ''}",
            "image_id": image.object_id if hasattr(image, 'object_id') else str(image),
        }
    except Exception as e:
        logger.error(f"Failed to snapshot sandbox: {e}")
        return {
            "success": False,
            "message": str(e),
            "error": str(e),
        }


@mcp.tool()
def read_sandbox_file(
    sandbox_id: str,
    path: str
) -> Dict[str, Any]:
    """
    Read a file from the sandbox filesystem.

    Args:
        sandbox_id: The Modal sandbox ID
        path: Absolute path to the file in the sandbox

    Returns:
        Dict with file content or error
    """
    try:
        sb = get_sandbox(sandbox_id)

        with sb.open(path, "r") as f:
            content = f.read()

        return {
            "success": True,
            "path": path,
            "content": content,
        }
    except Exception as e:
        logger.error(f"Failed to read file: {e}")
        return {
            "success": False,
            "path": path,
            "error": str(e),
        }


@mcp.tool()
def write_sandbox_file(
    sandbox_id: str,
    path: str,
    content: str
) -> Dict[str, Any]:
    """
    Write a file to the sandbox filesystem.

    Args:
        sandbox_id: The Modal sandbox ID
        path: Absolute path to the file in the sandbox
        content: Content to write to the file

    Returns:
        Dict with success status
    """
    try:
        sb = get_sandbox(sandbox_id)

        with sb.open(path, "w") as f:
            f.write(content)

        return {
            "success": True,
            "path": path,
            "message": f"File written: {path}",
        }
    except Exception as e:
        logger.error(f"Failed to write file: {e}")
        return {
            "success": False,
            "path": path,
            "error": str(e),
        }


@mcp.tool()
def list_sandbox_files(
    sandbox_id: str,
    path: str = "/root"
) -> Dict[str, Any]:
    """
    List files in a directory in the sandbox.

    Args:
        sandbox_id: The Modal sandbox ID
        path: Directory path to list (default: /root)

    Returns:
        Dict with list of files/directories
    """
    try:
        sb = get_sandbox(sandbox_id)

        entries = list(sb.ls(path))

        return {
            "success": True,
            "path": path,
            "entries": entries,
        }
    except Exception as e:
        logger.error(f"Failed to list files: {e}")
        return {
            "success": False,
            "path": path,
            "error": str(e),
        }


@mcp.tool()
def get_gpu_status(sandbox_id: str) -> Dict[str, Any]:
    """
    Get GPU status (nvidia-smi) from the sandbox.

    Args:
        sandbox_id: The Modal sandbox ID

    Returns:
        Dict with nvidia-smi output
    """
    return exec_in_sandbox(sandbox_id, "nvidia-smi", timeout=30)


@mcp.tool()
def get_running_processes(sandbox_id: str) -> Dict[str, Any]:
    """
    Get list of running processes in the sandbox.

    Args:
        sandbox_id: The Modal sandbox ID

    Returns:
        Dict with ps output
    """
    return exec_in_sandbox(sandbox_id, "ps aux", timeout=30)


def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
