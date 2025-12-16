"""Local notebook session - runs scribe server locally without Modal."""

import subprocess
import sys
import time
import shutil
import atexit
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

from ..workspace import Workspace
from .session_base import SessionBase


@dataclass
class LocalNotebookSession(SessionBase):
    """A notebook session running locally without Modal.

    Starts the scribe Jupyter server as a local subprocess and provides
    the same interface as NotebookSession for agent interaction.
    """

    session_id: str
    jupyter_url: str
    notebook_dir: str = "./outputs"
    _server_process: Optional[subprocess.Popen] = None

    def exec(self, code: str, hidden: bool = False) -> dict:
        """Execute code in the notebook kernel."""
        response = requests.post(
            f"{self.jupyter_url}/api/scribe/exec",
            json={"session_id": self.session_id, "code": code, "hidden": hidden},
            timeout=600,
        )
        response.raise_for_status()
        result = response.json()

        for output in result.get("outputs", []):
            if output.get("output_type") == "error":
                raise RuntimeError(f"{output.get('ename')}: {output.get('evalue')}")

        return result

    def exec_file(self, file_path: str, **kwargs) -> dict:
        """Execute a Python file in the notebook kernel."""
        code = Path(file_path).read_text()
        return self.exec(code, **kwargs)

    def setup(self, workspace: "Workspace"):
        """Apply workspace configuration to notebook session."""
        # Copy local files/dirs to workspace
        for src, dest in workspace.local_dirs:
            src_path = Path(src)
            dest_path = self.workspace_path / dest.lstrip("/")
            if not src_path.exists():
                raise FileNotFoundError(f"Directory not found: {src}")
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src_path, dest_path, dirs_exist_ok=True)

        for src, dest in workspace.local_files:
            src_path = Path(src)
            dest_path = self.workspace_path / dest.lstrip("/")
            if not src_path.exists():
                raise FileNotFoundError(f"File not found: {src}")
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dest_path)

        # Install libraries
        for library in workspace.libraries:
            library.install_in(self)

        # Install skills
        for skill in workspace.skills:
            skill.install_in(self)

        # Copy skill directories
        for skill_dir in workspace.skill_dirs:
            src_path = Path(skill_dir)
            if not src_path.exists():
                raise FileNotFoundError(f"Skill directory not found: {skill_dir}")
            skills_base = self.workspace_path / ".claude" / "skills"
            dest_path = skills_base / src_path.name
            skills_base.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src_path, dest_path, dirs_exist_ok=True)

        # Run custom init code
        if workspace.custom_init_code:
            self.exec(workspace.custom_init_code, hidden=True)

    @property
    def mcp_config(self) -> dict:
        """MCP configuration for connecting agents to this notebook."""
        return {
            "notebooks": {
                "type": "stdio",
                "command": "python",
                "args": ["-m", "scribe.notebook.notebook_mcp_server"],
                "env": {
                    "SCRIBE_URL": self.jupyter_url,
                    "NOTEBOOK_OUTPUT_DIR": self.notebook_dir,
                    "SCRIBE_SESSION_ID": self.session_id,
                }
            }
        }

    @property
    def model_info_text(self) -> str:
        """No pre-loaded models in local mode."""
        return ""

    def terminate(self):
        """Stop the local scribe server."""
        if self._server_process:
            self._server_process.terminate()
            try:
                self._server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._server_process.kill()
            self._server_process = None


def _wait_for_server(url: str, timeout: int = 30) -> bool:
    """Wait for server to be ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            response = requests.get(f"{url}/api/scribe/health", timeout=2)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(0.5)
    return False


def create_local_notebook_session(
    workspace: Optional[Workspace] = None,
    name: str = "local-session",
    port: int = 8888,
    notebook_dir: str = "./outputs",
    workspace_dir: str = "./workspace",
) -> LocalNotebookSession:
    """
    Create a local notebook session without Modal.

    Starts the scribe Jupyter server locally and creates a session.
    Requires jupyter_server, ipykernel, and other notebook dependencies
    to be installed in the current Python environment.

    Args:
        workspace: Workspace configuration (optional)
        name: Session/experiment name
        port: Port for the Jupyter server (default 8888)
        notebook_dir: Directory for notebook files
        workspace_dir: Local workspace directory

    Returns:
        LocalNotebookSession ready for agent execution
    """
    jupyter_url = f"http://localhost:{port}"

    # Check if server is already running
    server_already_running = False
    try:
        response = requests.get(f"{jupyter_url}/api/scribe/health", timeout=2)
        if response.status_code == 200:
            server_already_running = True
            print(f"Using existing scribe server at {jupyter_url}")
    except requests.exceptions.RequestException:
        pass

    server_process = None
    if not server_already_running:
        print(f"Starting local scribe server on port {port}...")

        # Ensure notebooks directory exists
        notebooks_path = Path(notebook_dir).absolute()
        notebooks_path.mkdir(parents=True, exist_ok=True)

        # Start scribe server
        # Use DEVNULL to avoid buffer issues that can cause the subprocess to hang
        server_process = subprocess.Popen(
            [
                sys.executable, "-m", "scribe.notebook.notebook_server",
                f"--port={port}",
                f"--ScribeServerApp.notebooks_dir={notebooks_path}",
                "--no-browser",
                "--ServerApp.token=",  # Disable token auth for local use
                "--ServerApp.password=",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Register cleanup on exit
        def cleanup():
            if server_process and server_process.poll() is None:
                server_process.terminate()
        atexit.register(cleanup)

        # Wait for server to be ready
        if not _wait_for_server(jupyter_url, timeout=30):
            server_process.terminate()
            raise RuntimeError(
                f"Failed to start scribe server on port {port}. "
                "Ensure jupyter_server and ipykernel are installed."
            )

        print(f"Scribe server ready at {jupyter_url}")

    # Create notebook session
    print(f"Creating notebook session: {name}")
    response = requests.post(
        f"{jupyter_url}/api/scribe/start",
        json={"experiment_name": name},
        timeout=30,
    )
    response.raise_for_status()
    session_data = response.json()

    # Setup workspace path
    workspace_path = Path(workspace_dir).absolute()
    workspace_path.mkdir(parents=True, exist_ok=True)

    session = LocalNotebookSession(
        session_id=session_data["session_id"],
        jupyter_url=jupyter_url,
        workspace_path=workspace_path,
        notebook_dir=notebook_dir,
        _server_process=server_process,
    )

    # Apply workspace configuration
    if workspace:
        session.setup(workspace)

    print(f"Session ready: {session.session_id}")
    print(f"Notebook: {session_data.get('notebook_path', notebook_dir)}")
    print(f"Workspace: {workspace_path}")

    return session
