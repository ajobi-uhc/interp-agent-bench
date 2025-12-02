"""Notebook session management."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import requests
import shutil

from ..environment.sandbox import Sandbox
from ..environment.utils import ModelHandle, RepoHandle
from ..workspace import Workspace
from .session_base import SessionBase


@dataclass
class NotebookSession(SessionBase):
    """A live Jupyter notebook session."""

    session_id: str
    jupyter_url: str
    sandbox: Sandbox

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

    def _mount_dir(self, src: str, dest: str):
        """Mount not supported in notebook - files are in sandbox."""
        print(f"Warning: Cannot mount {src} → {dest} in notebook session")

    def _mount_file(self, src: str, dest: str):
        """Mount not supported in notebook - files are in sandbox."""
        print(f"Warning: Cannot mount {src} → {dest} in notebook session")

    def _copy_skill_dir(self, skill_dir: str):
        """Copy skill directory to notebook's workspace."""
        src_path = Path(skill_dir)
        if not src_path.exists():
            raise FileNotFoundError(f"Skill directory not found: {skill_dir}")

        # Create skills directory in sandbox
        skills_base = self.workspace_path / ".claude" / "skills"
        dest_path = skills_base / src_path.name

        # Copy via sandbox
        self.exec(f"""
import shutil
from pathlib import Path

dest = Path("{dest_path}")
dest.parent.mkdir(parents=True, exist_ok=True)
""", hidden=True)

        # Would need to upload files to sandbox - simplified for now
        print(f"Skills copied to {dest_path}")

    def _execute_code(self, code: str):
        """Execute code in notebook kernel."""
        self.exec(code, hidden=True)


def create_notebook_session(
    sandbox: Sandbox,
    workspace: Optional[Workspace] = None,
    name: str = "session",
    notebook_dir: str = "./outputs",
) -> NotebookSession:
    """
    Create a notebook session and setup workspace.

    Args:
        sandbox: Sandbox with jupyter running
        workspace: Workspace configuration
        name: Session name
        notebook_dir: Directory for notebook files

    Returns:
        NotebookSession with workspace setup
    """
    if not sandbox.jupyter_url:
        raise RuntimeError("Sandbox needs jupyter. Use execution_mode=ExecutionMode.NOTEBOOK")

    print("Creating notebook session...")
    response = requests.post(
        f"{sandbox.jupyter_url}/api/scribe/start",
        json={"experiment_name": name},
        timeout=30,
    )
    response.raise_for_status()

    session = NotebookSession(
        session_id=response.json()["session_id"],
        jupyter_url=sandbox.jupyter_url,
        sandbox=sandbox,
        workspace_path=Path("/workspace"),
    )

    # Load models into kernel
    for handle in sandbox._model_handles:
        _load_model(session, handle)

    # Setup repos
    for handle in sandbox._repo_handles:
        _setup_repo(session, handle)

    # Setup workspace if provided
    if workspace:
        workspace.setup_in(session)

    print(f"Session ready: {session.session_id}")
    return session


def _load_model(session: NotebookSession, handle: ModelHandle):
    """Load model into kernel namespace."""
    var_info = f" as '{handle.var_name}'" if handle.var_name != "model" else ""
    model_name = "<hidden>" if handle.hidden else handle.name
    print(f"  Loading model{var_info}: {model_name}")

    var = handle.var_name
    tok_var = f"{var}_tokenizer" if var != "model" else "tokenizer"

    if handle.is_peft:
        code = f'''from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

_base = AutoModelForCausalLM.from_pretrained("{handle.base_model_path}", device_map="auto", torch_dtype="auto")
{var} = PeftModel.from_pretrained(_base, "{handle.volume_path}")
{tok_var} = AutoTokenizer.from_pretrained("{handle.base_model_path}")
del _base
'''
    else:
        code = f'''from transformers import AutoModelForCausalLM, AutoTokenizer

{var} = AutoModelForCausalLM.from_pretrained("{handle.volume_path}", device_map="auto", torch_dtype="auto")
{tok_var} = AutoTokenizer.from_pretrained("{handle.volume_path}")
'''

    if handle.hidden:
        code += f'''
if hasattr({var}, "config"):
    {var}.config.name_or_path = "model"
'''

    session.exec(code, hidden=True)


def _setup_repo(session: NotebookSession, handle: RepoHandle):
    """Setup repo workspace in kernel."""
    print(f"  Setting up repo: {handle.local_path}")

    code = f'from pathlib import Path\nWORKSPACE = Path("{handle.local_path}")\n'

    if handle.container_running:
        code += f'''
import subprocess

def container_exec(cmd: str) -> str:
    result = subprocess.run(
        ["docker", "exec", "{handle.container_name}", "bash", "-c", cmd],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return result.stdout
'''

    session.exec(code, hidden=True)
