"""Notebook session management."""

from dataclasses import dataclass
from typing import TYPE_CHECKING
import requests

from ..environment.sandbox import Sandbox
from ..environment.handles import ModelHandle, RepoHandle
from .session_base import SessionBase
from .model_loader import ModelLoader
from ._session_utils import read_and_exec

if TYPE_CHECKING:
    from ..extension import Extension


@dataclass
class NotebookSession(SessionBase):
    """A live notebook session."""
    session_id: str
    jupyter_url: str
    sandbox: Sandbox
    notebook_dir: str = "./outputs"

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
        return read_and_exec(file_path, self.exec, **kwargs)

    def _execute_extension(self, extension: "Extension"):
        """Execute extension code in notebook kernel."""
        if extension.code:
            self.exec(extension.code, hidden=True)

    @property
    def mcp_config(self) -> dict:
        """MCP configuration for connecting agents."""
        config = {
            "notebooks": {
                "type": "stdio",
                "command": "python",
                "args": ["-m", "scribe.notebook.notebook_mcp_server"],
                "env": {
                    "SCRIBE_URL": self.jupyter_url,
                    "NOTEBOOK_OUTPUT_DIR": self.notebook_dir
                }
            }
        }

        for endpoint in self._mcp_endpoints:
            config.update(endpoint)

        return config


def create_notebook_session(sandbox: Sandbox, name: str = "session", notebook_dir: str = "./outputs") -> NotebookSession:
    """Create a notebook session and load models/repos into kernel."""
    if not sandbox.jupyter_url:
        raise RuntimeError("Sandbox doesn't have jupyter. Use execution_mode=ExecutionMode.NOTEBOOK.")

    print("Creating notebook session...")
    response = requests.post(
        f"{sandbox.jupyter_url}/api/scribe/start",
        json={"experiment_name": name},
        timeout=30,
    )

    if response.status_code != 200:
        raise RuntimeError(f"Failed to start session (status {response.status_code}): {response.text}")

    session = NotebookSession(
        session_id=response.json()["session_id"],
        jupyter_url=sandbox.jupyter_url,
        sandbox=sandbox,
        notebook_dir=notebook_dir,
    )

    for handle in sandbox._model_handles:
        _attach_model(session, handle)

    for handle in sandbox._repo_handles:
        _attach_repo(session, handle)

    print(f"Session ready: {session.session_id}")
    return session


def _attach_model(session: NotebookSession, handle: ModelHandle):
    """Load model into kernel namespace."""
    var_info = f" as '{handle.var_name}'" if handle.var_name != "model" else ""
    model_name = "<hidden>" if handle.hidden else handle.name
    print(f"  Loading model{var_info}: {model_name}")
    code = ModelLoader.generate_code(handle, target="namespace")
    session.exec(code, hidden=True)


def _attach_repo(session: NotebookSession, handle: RepoHandle):
    """Setup repo workspace in kernel."""
    print(f"  Setting up repo: {handle.local_path}")

    code = f'from pathlib import Path\nWORKSPACE = Path("{handle.local_path}")\n'

    if handle.container_running:
        code += f'''
import subprocess

def container_exec(cmd: str) -> str:
    """Run command in the container."""
    result = subprocess.run(
        ["docker", "exec", "{handle.container_name}", "bash", "-c", cmd],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return result.stdout
'''

    session.exec(code, hidden=True)
