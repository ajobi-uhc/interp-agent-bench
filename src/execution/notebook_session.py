"""Notebook session management."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import requests
import shutil

from ..environment.sandbox import Sandbox, ModelHandle, RepoHandle
from ..workspace import Workspace
from .session_base import SessionBase
from .model_loader import generate_model_loading_code, generate_model_verification_code
from ..harness.logging import get_logger

logger = get_logger("notebook_session")


@dataclass
class NotebookSession(SessionBase):
    """A live Jupyter notebook session."""

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
        code = Path(file_path).read_text()
        return self.exec(code, **kwargs)

    def setup(self, workspace: "Workspace"):
        """Apply workspace configuration to notebook session."""
        # Mount local files/dirs (not supported in notebook - files must be in sandbox)
        if workspace.local_dirs or workspace.local_files:
            print("Warning: local_dirs/local_files not supported in notebook sessions")
            print("  Use SandboxConfig.local_dirs/local_files to include files at build time")

        # Install libraries
        for library in workspace.libraries:
            library.install_in(self)

        # Install skills
        for skill in workspace.skills:
            skill.install_in(self)

        # Copy skill directories
        for skill_dir in workspace.skill_dirs:
            self._copy_skill_dir(skill_dir)

        # Run custom init code
        if workspace.custom_init_code:
            self.exec(workspace.custom_init_code, hidden=True)

    def _copy_skill_dir(self, skill_dir: str):
        """Copy skill directory to notebook's workspace."""
        src_path = Path(skill_dir)
        if not src_path.exists():
            raise FileNotFoundError(f"Skill directory not found: {skill_dir}")

        skills_base = self.workspace_path / ".claude" / "skills"
        dest_path = skills_base / src_path.name

        # Create skills directory in sandbox
        self.exec(f"""
import shutil
from pathlib import Path

dest = Path("{dest_path}")
dest.parent.mkdir(parents=True, exist_ok=True)
""", hidden=True)

        print(f"Skills copied to {dest_path}")

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
        """Generate formatted text describing pre-loaded models."""
        if not self.sandbox.model_handles:
            return ""

        lines = ["### Pre-loaded Models", "The following models are already loaded in the kernel:"]
        for handle in self.sandbox.model_handles:
            model_name = handle.name if not handle.hidden else "<hidden>"
            lines.append(f"- `{handle.var_name}`: {model_name}")
            tokenizer_var = f"{handle.var_name}_tokenizer" if handle.var_name != "model" else "tokenizer"
            lines.append(f"- `{tokenizer_var}`: Tokenizer")
        lines.append("\n**Do NOT reload these models - they are already available.**")
        return "\n".join(lines)


def create_notebook_session(
    sandbox: Sandbox,
    workspace: Optional[Workspace] = None,
    name: str = "session",
    notebook_dir: str = "./outputs",
) -> NotebookSession:
    """
    Create a notebook session connected to a sandbox.

    Loads sandbox models/repos into the notebook kernel, then applies workspace configuration.

    Args:
        sandbox: Sandbox with jupyter running
        workspace: Workspace configuration (optional)
        name: Session name
        notebook_dir: Directory for notebook files

    Returns:
        NotebookSession ready for agent execution
    """
    if not sandbox.jupyter_url:
        raise RuntimeError("Sandbox needs jupyter. Use execution_mode=ExecutionMode.NOTEBOOK")

    logger.info(f"Creating notebook session: {name}")
    logger.info(f"📓 [bold cyan]A notebook will be synced to:[/bold cyan] [yellow]{notebook_dir}[/yellow] [dim]for you to follow along[/dim]")
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
        notebook_dir=notebook_dir,
    )
    logger.info(f"Session created with ID: {session.session_id}")

    # Load sandbox resources into kernel
    logger.info("Loading sandbox resources into kernel...")
    _load_sandbox_resources(session, sandbox, workspace)

    # Apply workspace configuration
    if workspace:
        logger.info("Applying workspace configuration...")
        session.setup(workspace)

    logger.info(f"Notebook session ready: {sandbox.jupyter_url}/tree/notebooks")
    return session


def _load_sandbox_resources(session: NotebookSession, sandbox: Sandbox, workspace: Optional[Workspace]):
    """Load models and repos from sandbox into notebook kernel."""
    # Load models (if workspace allows)
    if workspace is None or workspace.preload_models:
        hidden = workspace.hidden_model_loading if workspace else True
        for handle in sandbox.model_handles:
            _load_model(session, handle, hidden=hidden)

    # Setup repos
    for handle in sandbox.repo_handles:
        _setup_repo(session, handle)


def _load_model(session: NotebookSession, handle: ModelHandle, hidden: bool = True):
    """Load model into kernel namespace.

    Args:
        session: The notebook session
        handle: Model handle with metadata
        hidden: Whether to hide the loading cell (default True)
    """
    var_info = f" as '{handle.var_name}'" if handle.var_name != "model" else ""
    model_name = "<hidden>" if handle.hidden else handle.name
    print(f"  Loading model{var_info}: {model_name}")

    code = generate_model_loading_code(handle) + generate_model_verification_code(handle)
    session.exec(code, hidden=hidden)


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
