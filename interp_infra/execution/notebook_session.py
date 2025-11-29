"""Simple notebook session management."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import requests

from ..environment.sandbox import Sandbox
from ..environment.handles import ModelHandle, RepoHandle


@dataclass
class NotebookSession:
    """A live notebook session."""
    session_id: str
    jupyter_url: str
    sandbox: Sandbox
    notebook_dir: str = "./outputs"

    def exec(self, code: str, hidden: bool = False) -> dict:
        """Execute code in the notebook kernel."""
        response = requests.post(
            f"{self.jupyter_url}/api/scribe/exec",
            json={
                "session_id": self.session_id,
                "code": code,
                "hidden": hidden,
            },
            timeout=600,
        )
        response.raise_for_status()
        result = response.json()

        # Check for errors
        for output in result.get("outputs", []):
            if output.get("output_type") == "error":
                raise RuntimeError(
                    f"{output.get('ename')}: {output.get('evalue')}"
                )

        return result

    @property
    def mcp_config(self) -> dict:
        """MCP config for connecting an agent."""
        return {
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


def create_notebook_session(
    sandbox: Sandbox,
    name: str = "session",
    notebook_dir: str = "./outputs",
) -> NotebookSession:
    """
    Create a notebook session and load models/repos into kernel.

    Args:
        sandbox: Started sandbox with jupyter running
        name: Session name

    Returns:
        NotebookSession ready for agent
    """
    if not sandbox.jupyter_url:
        raise RuntimeError("Sandbox doesn't have jupyter. Use notebook=True.")

    # Start kernel session
    print("Creating notebook session...")
    response = requests.post(
        f"{sandbox.jupyter_url}/api/scribe/start",
        json={"experiment_name": name},
        timeout=30,
    )

    if response.status_code != 200:
        error_detail = response.text
        raise RuntimeError(
            f"Failed to start session (status {response.status_code}): {error_detail}"
        )

    session_id = response.json()["session_id"]

    session = NotebookSession(
        session_id=session_id,
        jupyter_url=sandbox.jupyter_url,
        sandbox=sandbox,
        notebook_dir=notebook_dir,
    )

    # Load models
    for handle in sandbox._model_handles:
        _attach_model(session, handle)

    # Setup repos
    for handle in sandbox._repo_handles:
        _attach_repo(session, handle)

    print(f"  Session ready: {session_id}")
    return session


def _attach_model(session: NotebookSession, handle: ModelHandle):
    """Load model into kernel namespace."""
    print(f"  Loading model: {'<hidden>' if handle.hidden else handle.name}")

    if handle.is_peft:
        code = f'''
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

_base = AutoModelForCausalLM.from_pretrained(
    "{handle.base_model_path}",
    device_map="auto",
    torch_dtype="auto",
)
model = PeftModel.from_pretrained(_base, "{handle.volume_path}")
tokenizer = AutoTokenizer.from_pretrained("{handle.base_model_path}")
del _base
'''
    else:
        code = f'''
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "{handle.volume_path}",
    device_map="auto",
    torch_dtype="auto",
)
tokenizer = AutoTokenizer.from_pretrained("{handle.volume_path}")
'''

    if handle.hidden:
        code += '''
if hasattr(model, "config"):
    model.config.name_or_path = "model"
    if hasattr(model.config, "_name_or_path"):
        model.config._name_or_path = "model"
'''

    session.exec(code, hidden=True)


def _attach_repo(session: NotebookSession, handle: RepoHandle):
    """Setup repo workspace in kernel."""
    print(f"  Setting up repo: {handle.local_path}")

    code = f'''
from pathlib import Path
WORKSPACE = Path("{handle.local_path}")
'''

    if handle.container_running:
        code += f'''
import subprocess

def container_exec(cmd: str) -> str:
    """Run command in the container."""
    result = subprocess.run(
        ["docker", "exec", "{handle.container_name}", "bash", "-c", cmd],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return result.stdout
'''

    session.exec(code, hidden=True)
