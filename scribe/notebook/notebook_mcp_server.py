"""
Scribe Notebook MCP Server - Model Context Protocol interface for agents to work with
the Scribe notebook server. MCP endpoints are easier for agents to interact with than
raw HTTP requests or Jupyter Server API calls.
"""

import atexit
import os
import secrets
import signal
import subprocess
import sys
from typing import Dict, Any, Optional, List, Union

import requests
from fastmcp import FastMCP
from fastmcp.utilities.types import Image

from scribe.notebook._notebook_server_utils import (
    find_safe_port,
    check_server_health,
    start_scribe_server,
    cleanup_scribe_server,
    process_jupyter_outputs,
)  # noqa: E402
from scribe.notebook.technique_manager import TechniqueManager


# Initialize MCP server
mcp = FastMCP("scribe")

# Parse selected techniques from env
_selected_techniques_str = os.environ.get("SELECTED_TECHNIQUES", "")
_selected_techniques = (
    [t.strip() for t in _selected_techniques_str.split(",") if t.strip()]
    if _selected_techniques_str
    else None
)

print("[MCP] Initializing TechniqueManager...", file=sys.stderr)
_technique_manager = TechniqueManager(
    experiment_name=os.environ.get("EXPERIMENT_NAME", "scribe"),
    model_name=os.environ.get("MODEL_NAME"),
    model_is_peft=os.environ.get("MODEL_IS_PEFT", "").lower() == "true",
    model_base=os.environ.get("MODEL_BASE"),
    tokenizer_name=os.environ.get("TOKENIZER_NAME"),
    selected_techniques=_selected_techniques,
    execution_mode=os.environ.get("EXECUTION_MODE", "modal"),
    device=os.environ.get("DEVICE", "auto"),
    hidden_system_prompt=os.environ.get("HIDDEN_SYSTEM_PROMPT", ""),
    api_provider=os.environ.get("API_PROVIDER"),
)
print(f"[MCP] TechniqueManager initialized", file=sys.stderr)

# Global server management
_server_process: Optional[subprocess.Popen] = None
_server_port: Optional[int] = None
_server_url: Optional[str] = None
_server_token: Optional[str] = None
# Down the line, we may wish to keep the Jupyter server around even after MCP server exits
_is_external_server: bool = False

SCRIBE_PROVIDER: str = os.environ.get("SCRIBE_PROVIDER")

# Session tracking for cleanup
_active_sessions: set = set()


def start_jupyter_server() -> tuple[subprocess.Popen, int, str]:
    """Start a Jupyter server subprocess and return process, port, and URL."""
    port = find_safe_port()
    if port is None:
        raise Exception("Could not find an available port for Jupyter server")

    # Generate token for this server instance
    token = get_token()

    # Get notebook output directory from environment variable
    notebook_output_dir = os.environ.get("NOTEBOOK_OUTPUT_DIR")

    # Use utils function to start server
    process = start_scribe_server(port, token, notebook_output_dir)
    url = f"http://127.0.0.1:{port}"

    return process, port, url


def cleanup_server():
    """Clean up the managed Jupyter server."""
    global _server_process, _server_token, _active_sessions

    if _server_process and not _is_external_server:
        cleanup_scribe_server(_server_process)
        _server_process = None
        _server_token = None  # Clear token on cleanup


def ensure_server_running() -> str:
    """Ensure a Jupyter server is running and return its URL."""
    global _server_process, _server_port, _server_url, _is_external_server

    # Check if SCRIBE_PORT is set (external server)
    if "SCRIBE_PORT" in os.environ:
        port = os.environ["SCRIBE_PORT"]
        _server_port = int(port)
        _server_url = f"http://127.0.0.1:{port}"
        _is_external_server = True
        return _server_url

    # Check if our managed server is still running
    if _server_process and _server_process.poll() is None:
        return _server_url

    # Start a new managed server
    _is_external_server = False
    _server_process, _server_port, _server_url = start_jupyter_server()

    # Register cleanup handlers
    atexit.register(cleanup_server)
    signal.signal(signal.SIGTERM, lambda sig, frame: cleanup_server())
    signal.signal(signal.SIGINT, lambda sig, frame: cleanup_server())

    print(f"Started managed Jupyter server at {_server_url}", file=sys.stderr)
    return _server_url


def get_token() -> str:
    """Generate or return cached auth token."""
    global _server_token
    if not _server_token and not _is_external_server:
        _server_token = secrets.token_urlsafe(32)
    return _server_token or ""


def get_server_status() -> Dict[str, Any]:
    """Get current server status information."""
    global _server_port, _server_url, _is_external_server, _server_process

    if not _server_url:
        return {
            "status": "not_started",
            "url": None,
            "port": None,
            "vscode_url": None,
            "will_shutdown_on_exit": True,
            "is_external": False,
            "health": "unknown",
        }

    # Check health using utils function
    health_data = check_server_health(_server_port) if _server_port else None
    health = "healthy" if health_data else "unreachable"

    # Check if process is still running (for managed servers)
    process_running = True
    if not _is_external_server and _server_process:
        process_running = _server_process.poll() is None

    return {
        "status": "running" if process_running else "stopped",
        "url": _server_url,
        "port": _server_port,
        "vscode_url": f"{_server_url}/?token={get_token()}" if _server_url else None,
        "will_shutdown_on_exit": not _is_external_server,
        "is_external": _is_external_server,
        "health": health,
    }


async def _start_session_internal(
    experiment_name: Optional[str] = None,
    notebook_path: Optional[str] = None,
    fork_prev_notebook: bool = True,
    tool_name: str = "start_session",
) -> Dict[str, Any]:
    """
    Internal helper function for starting sessions from scratch versus resuming versus forking existing notebook.

    Args:
        experiment_name: Custom name for the notebook
        notebook_path: Path to existing notebook (if any)
        fork_prev_notebook: If True, create new notebook; if False, use existing in-place
        tool_name: Name of the calling tool for logging/debugging
    """
    try:
        # Ensure server is running
        server_url = ensure_server_running()

        # Build request body
        request_body = {}
        if experiment_name:
            request_body["experiment_name"] = experiment_name
        if notebook_path:
            request_body["notebook_path"] = notebook_path
            request_body["fork_prev_notebook"] = fork_prev_notebook

        # Start session
        token = get_token()
        headers = {"Authorization": f"token {token}"} if token else {}
        print(f"[DEBUG MCP] {tool_name}: Connecting to {server_url}", file=sys.stderr)

        response = requests.post(
            f"{server_url}/api/scribe/start", json=request_body, headers=headers
        )
        response.raise_for_status()
        data = response.json()

        result = {
            "session_id": data["session_id"],
            "kernel_id": data.get("kernel_id"),
            "status": "started",
            "notebook_path": data["notebook_path"],
            "vscode_url": f"{data.get('server_url', server_url)}/?token={data.get('token', token)}",
            "kernel_name": data.get(
                "kernel_name", data.get("kernel_display_name", "Scribe Kernel")
            ),
        }

        # Track session for cleanup
        global _active_sessions
        _active_sessions.add(data["session_id"])

        # Handle restoration results if present (only for notebook-based sessions)
        if notebook_path:
            # Pass through restoration summary if present
            if "restoration_summary" in data:
                result["restoration_summary"] = data["restoration_summary"]

            # Only pass error details for debugging, not full restoration results
            if "restoration_results" in data:
                errors = [
                    r for r in data["restoration_results"] if r.get("status") == "error"
                ]
                if errors:
                    # Summarize errors with cell numbers and error messages only
                    error_summary = []
                    for error in errors:
                        error_info = {
                            "cell": error.get("cell"),
                            "error": error.get("error", "").split(":")[0]
                            if ":" in error.get("error", "")
                            else error.get("error", ""),
                        }
                        error_summary.append(error_info)
                    result["restoration_errors"] = error_summary

            # Add guidance for agent when working with existing notebooks
            if "restoration_summary" in data:
                has_errors = (
                    "restoration_errors" in result and result["restoration_errors"]
                )
                if fork_prev_notebook:
                    # Continue/fork scenario
                    if has_errors:
                        result["note"] = (
                            f"A new notebook has been created at {data['notebook_path']} "
                            f"with the restored state from {notebook_path}. "
                            f"Some cells had errors during restoration - see restoration_errors for details. "
                            "Please use the NotebookRead tool to review the notebook contents."
                        )
                    else:
                        result["note"] = (
                            f"A new notebook has been created at {data['notebook_path']} "
                            f"with the restored state from {notebook_path}. "
                            "All cells executed successfully during restoration. "
                            "Please use the NotebookRead tool to review the notebook contents."
                        )
                else:
                    # Resume scenario
                    if has_errors:
                        result["note"] = (
                            f"Resumed notebook at {data['notebook_path']} in-place. "
                            f"Some cells had errors during restoration - see restoration_errors for details. "
                            "Please use the NotebookRead tool to review the notebook contents."
                        )
                    else:
                        result["note"] = (
                            f"Successfully resumed notebook at {data['notebook_path']} in-place. "
                            "All cells executed successfully during restoration. "
                            "Please use the NotebookRead tool to review the notebook contents."
                        )

        return result

    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to start session ({tool_name}): {str(e)}")


@mcp.tool
async def start_new_session(experiment_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Start a completely new Jupyter kernel session with an empty notebook.

    Args:
        experiment_name: Custom name for the notebook (e.g., "ImageGeneration")

    Returns:
        Dictionary with:
        - session_id: Unique session identifier
        - notebook_path: Path to the new notebook
        - status: "started"
        - kernel_id: The kernel ID
        - vscode_url: URL to connect with VSCode
        - kernel_name: Display name of the kernel
    """
    return await _start_session_internal(
        experiment_name=experiment_name,
        notebook_path=None,
        fork_prev_notebook=True,
        tool_name="start_new_session",
    )


@mcp.tool(
    name="start_session_resume_notebook",
    # Description pulled from docstring
    tags=None,
    annotations={
        "title": "Start Session - Resume Notebook"  # A human-readable title for the tool.
    },
)
async def start_session_resume_notebook(notebook_path: str) -> Dict[str, Any]:
    """
    Start a new session by resuming an existing notebook in-place, modifying the original notebook file.

    This executes all cells in the existing notebook to restore the kernel state and updates
    the notebook file with new outputs. Use this to continue working in an existing notebook file.

    Args:
        notebook_path: Path to the existing notebook to resume from

    Returns:
        Dictionary with:
        - session_id: Unique session identifier
        - notebook_path: Path to the resumed notebook (same as input)
        - status: "started"
        - kernel_id: The kernel ID
        - vscode_url: URL to connect with VSCode
        - kernel_name: Display name of the kernel
        - restoration_summary: Summary of the resume operation
        - restoration_errors: List of any errors that occurred during cell execution
        - note: Guidance message about the resumed notebook
    """
    return await _start_session_internal(
        experiment_name=None,
        notebook_path=notebook_path,
        fork_prev_notebook=False,
        tool_name="start_session_resume_notebook",
    )


@mcp.tool
async def start_session_continue_notebook(
    notebook_path: str, experiment_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Start a session by continuing from an existing notebook (creates a new notebook file).

    This creates a new notebook with "_continued" suffix, copies all cells from the existing
    notebook, and executes them to restore the kernel state. The original notebook is unchanged.

    Args:
        notebook_path: Path to the existing notebook to continue from
        experiment_name: Optional custom name for the new notebook

    Returns:
        Dictionary with:
        - session_id: Unique session identifier
        - notebook_path: Path to the new notebook (with "_continued" suffix)
        - status: "started"
        - kernel_id: The kernel ID
        - vscode_url: URL to connect with VSCode
        - kernel_name: Display name of the kernel
        - restoration_summary: Summary of the continuation operation
        - restoration_errors: List of any errors that occurred during cell execution
        - note: Guidance to read the new notebook
    """
    return await _start_session_internal(
        experiment_name=experiment_name,
        notebook_path=notebook_path,
        fork_prev_notebook=True,
        tool_name="start_session_continue_notebook",
    )


async def _execute_code_internal(
    session_id: str, code: str, hidden: bool = False
) -> List[Union[Dict[str, Any], Image]]:
    """Internal helper to execute code - can be called from other functions.

    Args:
        session_id: The session ID
        code: Python code to execute
        hidden: If True, code executes in kernel but doesn't create a visible notebook cell
    """
    try:
        server_url = ensure_server_running()
        token = get_token()
        headers = {"Authorization": f"token {token}"} if token else {}

        response = requests.post(
            f"{server_url}/api/scribe/exec",
            json={"session_id": session_id, "code": code, "hidden": hidden},
            headers=headers,
        )
        response.raise_for_status()
        data = response.json()

        # Process outputs using utils function
        outputs, images = process_jupyter_outputs(
            data["outputs"],
            session_id=session_id,
            save_images_locally=False,
        )

        # Create result list with execution metadata first, then images
        result = [
            {
                "session_id": session_id,
                "execution_count": data["execution_count"],
                "outputs": outputs,
            }
        ]
        result.extend(images)
        return result

    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to execute code: {str(e)}")


@mcp.tool
async def execute_code(
    session_id: str, code: str
) -> List[Union[Dict[str, Any], Image]]:
    """
    Execute Python code in the specified kernel session.

    Images generated during execution (e.g., via .show()) are returned as
    fastmcp.Image objects that can be directly viewed.

    Args:
        session_id: The session ID returned by start_session
        code: Python code to execute

    Returns:
        Dictionary containing:
        - session_id: The session ID
        - execution_count: The cell execution number
        - outputs: List of output objects with type and content/data
    """
    return await _execute_code_internal(session_id, code)


@mcp.tool
async def add_markdown(session_id: str, content: str) -> Dict[str, int]:
    """
    Add a markdown cell to the notebook for documentation.

    Args:
        session_id: The session ID
        content: Markdown content to add

    Returns:
        Dictionary with the cell number
    """
    try:
        server_url = ensure_server_running()
        token = get_token()
        headers = {"Authorization": f"token {token}"} if token else {}

        response = requests.post(
            f"{server_url}/api/scribe/markdown",
            json={"session_id": session_id, "content": content},
            headers=headers,
        )
        response.raise_for_status()
        data = response.json()


        return {"cell_number": data["cell_number"]}

    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to add markdown: {str(e)}")


@mcp.tool
async def edit_cell(
    session_id: str, code: str, cell_index: int = -1
) -> List[Union[Dict[str, Any], Image]]:
    """
    Edit an existing code cell in the notebook and execute the new code.

    This is especially useful for fixing errors or modifying the most recent cell.

    Args:
        session_id: The session ID
        code: New Python code to replace the cell content
        cell_index: Index of the code cell to edit (default -1 for last cell)
                   Use -1 for the most recent cell, -2 for second to last, etc.
                   Or use 0, 1, 2... for specific cells from the beginning

    Returns:
        Dictionary containing:
        - session_id: The session ID
        - cell_index: The code cell index that was edited
        - actual_notebook_index: The actual index in the notebook (including markdown cells)
        - execution_count: The cell execution number
        - outputs: List of output objects with type and content/data
    """
    # start_time = time.time()
    try:
        server_url = ensure_server_running()
        token = get_token()
        headers = {"Authorization": f"token {token}"} if token else {}

        response = requests.post(
            f"{server_url}/api/scribe/edit",
            json={"session_id": session_id, "code": code, "cell_index": cell_index},
            headers=headers,
        )
        response.raise_for_status()
        data = response.json()

        # Process outputs using utils function
        outputs, images = process_jupyter_outputs(
            data["outputs"],
            session_id=session_id,
            save_images_locally=False,
        )


        # Create result list with execution metadata first, then images
        result = [
            {
                "session_id": session_id,
                "cell_index": data["cell_index"],
                "actual_notebook_index": data["actual_notebook_index"],
                "execution_count": data["execution_count"],
                "outputs": outputs,
            }
        ]
        result.extend(images)
        return result

    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to edit cell: {str(e)}")


@mcp.tool
async def init_session(session_id: Optional[str] = None) -> Dict[str, Any]:
    """Return protocol instructions and technique catalogue.

    If session_id is provided, automatically executes hidden setup code
    to initialize the environment without exposing implementation details.

    Returns:
        Dict with:
        - instructions: Setup workflow instructions
        - setup_snippet: Code to execute (if applicable)
        - techniques: Available techniques
        - hidden_setup_executed: Whether hidden setup ran (True/False)
        - hidden_setup_status: Status of hidden setup ('success'/'error'/'skipped')
        - hidden_setup_message: Human-readable status message
    """
    print("[MCP] init_session called", file=sys.stderr)
    print(f"[MCP] session_id provided: {session_id is not None}", file=sys.stderr)

    result = _technique_manager.init_payload()

    # If session provided, auto-execute hidden setup code
    if session_id:
        hidden_code = _technique_manager.get_hidden_setup_code()
        print(f"[MCP] Hidden code length: {len(hidden_code) if hidden_code else 0}", file=sys.stderr)

        if hidden_code:
            # Add preview of hidden code to stderr for debugging
            print(f"[MCP] Hidden setup code preview (first 200 chars):", file=sys.stderr)
            print(f"[MCP] {hidden_code[:200]}...", file=sys.stderr)
            print(f"[MCP] Auto-executing {len(hidden_code)} chars of hidden setup code...", file=sys.stderr)

            try:
                # Execute hidden code in the agent's kernel WITHOUT creating a visible notebook cell
                # (use internal function to avoid MCP tool wrapper issues, and hidden=True to skip notebook cell creation)
                exec_result = await _execute_code_internal(session_id, hidden_code, hidden=True)
                print(f"[MCP] Hidden setup executed successfully (hidden from notebook)", file=sys.stderr)
                print(f"[MCP] Execution result: {exec_result}", file=sys.stderr)
                result['hidden_setup_executed'] = True
                result['hidden_setup_status'] = 'success'
                result['hidden_setup_message'] = f'✅ Hidden environment setup completed ({len(hidden_code)} chars executed, not visible in notebook)'
            except Exception as e:
                print(f"[MCP] ERROR: Failed to execute hidden setup: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)
                result['hidden_setup_executed'] = False
                result['hidden_setup_status'] = 'error'
                result['hidden_setup_message'] = f'❌ Hidden setup failed: {str(e)}'
        else:
            print(f"[MCP] No hidden setup code to execute", file=sys.stderr)
            result['hidden_setup_executed'] = False
            result['hidden_setup_status'] = 'skipped'
            result['hidden_setup_message'] = 'No hidden setup code configured'
    else:
        print(f"[MCP] No session_id provided, skipping hidden setup", file=sys.stderr)
        result['hidden_setup_executed'] = False
        result['hidden_setup_status'] = 'skipped'
        result['hidden_setup_message'] = 'No session_id provided'

    print(f"[MCP] Returning init result (status={result.get('hidden_setup_status')})", file=sys.stderr)
    return result


@mcp.tool
async def list_techniques() -> Dict[str, Dict[str, Any]]:
    """List available notebook techniques."""

    return _technique_manager.list_payload()


@mcp.tool
async def describe_technique(technique_name: str) -> Dict[str, Any]:
    """Return documentation and call snippet for ``technique_name``."""

    return _technique_manager.describe_payload(technique_name)


@mcp.tool
async def shutdown_session(session_id: str) -> str:
    """
    Shutdown a kernel session gracefully.

    Note: using this tool terminates kernel state; it should typically only be used if the user
    has instructured you to do so.

    Args:  session_id: The session ID to shutdown
    """
    try:
        server_url = ensure_server_running()
        token = get_token()
        headers = {"Authorization": f"token {token}"} if token else {}

        response = requests.post(
            f"{server_url}/api/scribe/shutdown",
            json={"session_id": session_id},
            headers=headers,
        )
        response.raise_for_status()

        # Clean up session images if image saving is enabled
        global _active_sessions
        return f"Session {session_id} shut down successfully"

    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to shutdown session: {str(e)}")


@mcp.resource(
    uri="scribe://server/status",
    name="ScribeNotebookServerStatus",  # A human-readable name. If not provided, defaults to function name
    description="Get the current Scribe server status and connection information.",
)
async def server_status() -> str:
    status = get_server_status()

    # Format as a readable status report
    lines = [
        "# Scribe Server Status",
        "",
        f"**Status:** {status['status']}",
        f"**URL:** {status['url'] or 'Not available'}",
        f"**Port:** {status['port'] or 'Not available'}",
        f"**VSCode URL:** {status['vscode_url'] or 'Not available'}",
        f"**Health:** {status['health']}",
        f"**Auth Token:** {'Yes (auto-generated)' if get_token() else 'No'}",
        f"**External Server:** {'Yes' if status['is_external'] else 'No'}",
        f"**Will Shutdown on Exit:** {'Yes' if status['will_shutdown_on_exit'] else 'No'}",
    ]

    if not status["is_external"]:
        lines.extend(["", "*This server is automatically managed by the MCP server.*"])
    else:
        lines.extend(
            ["", "*This server was found via SCRIBE_PORT environment variable.*"]
        )

    return "\n".join(lines)


# Main entry point for STDIO transport
if __name__ == "__main__":
    mcp.run(transport="stdio")
