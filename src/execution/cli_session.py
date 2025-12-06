"""CLI session management."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import shutil

from ..environment.sandbox import Sandbox, ModelHandle, RepoHandle
from ..workspace import Workspace
from .session_base import SessionBase
from .model_loader import generate_model_loading_code


@dataclass
class CLISession(SessionBase):
    """A CLI session for shell-based interactions."""

    sandbox: Sandbox
    session_id: str

    def exec(self, code: str, **kwargs) -> str:
        """Execute Python code in the sandbox."""
        return self.sandbox.exec_python(code)

    def exec_shell(self, cmd: str) -> str:
        """Execute a shell command in the sandbox."""
        return self.sandbox.exec(cmd)

    @property
    def mcp_config(self) -> dict:
        """MCP configuration for connecting agents to this CLI session."""
        if not self.sandbox.sandbox_id:
            raise RuntimeError("Sandbox not started - no sandbox ID available")

        return {
            "cli": {
                "type": "stdio",
                "command": "python",
                "args": ["-m", "src.mcps.cli"],
                "env": {
                    "SANDBOX_ID": self.sandbox.sandbox_id,
                }
            }
        }

    def setup(self, workspace: "Workspace"):
        """Apply workspace configuration to CLI session."""
        # Mount local files/dirs
        for src, dest in workspace.local_dirs:
            print(f"Warning: Mounting directories not fully implemented - would copy {src} → {dest}")

        for src, dest in workspace.local_files:
            src_path = Path(src)
            if src_path.exists():
                content = src_path.read_text()
                self.sandbox.write_file(dest, content)

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
            self.exec(workspace.custom_init_code)

    def _copy_skill_dir(self, skill_dir: str):
        """Copy skill directory to workspace/.claude/skills/."""
        src_path = Path(skill_dir)
        if not src_path.exists():
            raise FileNotFoundError(f"Skill directory not found: {skill_dir}")

        dest = f"{self.workspace_path}/.claude/skills/{src_path.name}"

        # Copy SKILL.md if exists
        skill_md = src_path / "SKILL.md"
        if skill_md.exists():
            content = skill_md.read_text()
            self.sandbox.write_file(f"{dest}/SKILL.md", content)


def create_cli_session(
    sandbox: Sandbox,
    workspace: Optional[Workspace] = None,
    name: str = "cli-session",
) -> CLISession:
    """
    Create a CLI session connected to a sandbox.

    Prepares sandbox models/repos as scripts, then applies workspace configuration.

    Args:
        sandbox: Sandbox in CLI mode
        workspace: Workspace configuration (optional)
        name: Session name

    Returns:
        CLISession ready for agent execution
    """
    from ..environment.sandbox import ExecutionMode

    if sandbox.config.execution_mode.value != "cli":
        raise RuntimeError("Sandbox must be in CLI mode. Use ExecutionMode.CLI")

    print(f"Creating CLI session: {name}")
    session = CLISession(
        sandbox=sandbox,
        session_id=name,
        workspace_path=Path("/workspace"),
    )

    # Prepare sandbox resources as scripts
    _prepare_sandbox_resources(session, sandbox, workspace)

    # Apply workspace configuration
    if workspace:
        session.setup(workspace)

    print(f"CLI session ready: {name}")
    return session


def _prepare_sandbox_resources(session: CLISession, sandbox: Sandbox, workspace: Optional[Workspace]):
    """Prepare models and repos from sandbox as CLI scripts."""
    # Prepare models (if workspace allows)
    if workspace is None or workspace.preload_models:
        for handle in sandbox.model_handles:
            _prepare_model(session, handle)

    # Prepare repos
    for handle in sandbox.repo_handles:
        _prepare_repo(session, handle)


def _prepare_model(session: CLISession, handle: ModelHandle):
    """Create model loading script in workspace."""
    model_name = "<hidden>" if handle.hidden else handle.name
    print(f"  Preparing model: {model_name}")

    script = "#!/usr/bin/env python3\n" + generate_model_loading_code(handle)

    script_name = "load_model.py" if handle.hidden else f"load_{handle.name.replace('/', '_')}.py"
    session.sandbox.write_file(f"/workspace/{script_name}", script)
    session.exec_shell(f"chmod +x /workspace/{script_name}")


def _prepare_repo(session: CLISession, handle: RepoHandle):
    """Create repo README and helper scripts."""
    print(f"  Preparing repo: {handle.local_path}")

    repo_name = handle.url.split("/")[-1].replace(".git", "")
    readme = f"Repository: {repo_name}\nLocation: {handle.local_path}\nURL: {handle.url}\n"

    if handle.container_running:
        readme += f"\nContainer: {handle.container_name} (running)\n"

        helper = f'''#!/usr/bin/env python3
import subprocess, sys

def container_exec(cmd):
    r = subprocess.run(["docker", "exec", "{handle.container_name}", "bash", "-c", cmd], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr)
    return r.stdout

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: exec_{handle.container_name}.py <command>")
        sys.exit(1)
    print(container_exec(" ".join(sys.argv[1:])))
'''
        session.sandbox.write_file(f"/workspace/exec_{handle.container_name}.py", helper)
        session.exec_shell(f"chmod +x /workspace/exec_{handle.container_name}.py")
        readme += f"Helper: /workspace/exec_{handle.container_name}.py\n"

    session.sandbox.write_file(f"/workspace/README_{repo_name}.txt", readme)
