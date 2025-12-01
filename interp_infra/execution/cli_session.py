"""CLI session management for shell-based interactions."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..environment.sandbox import Sandbox
from ..environment.handles import ModelHandle, RepoHandle
from .session_base import SessionBase
from .model_loader import ModelLoader
from ._session_utils import read_and_exec

if TYPE_CHECKING:
    from ..extension import Extension


@dataclass
class CLISession(SessionBase):
    """A CLI session for running shell commands."""
    sandbox: Sandbox
    session_id: str

    def exec(self, cmd: str) -> str:
        """Execute a shell command in the sandbox."""
        return self.sandbox.exec(cmd)

    def exec_python(self, code: str) -> str:
        """Execute Python code in the sandbox."""
        return self.sandbox.exec_python(code)

    def exec_file(self, file_path: str, **kwargs) -> str:
        """Execute a Python file in the sandbox."""
        return read_and_exec(file_path, self.exec_python, **kwargs)

    def _execute_extension(self, extension: "Extension"):
        """Execute extension code via exec_python."""
        if extension.code:
            self.exec_python(extension.code)

    @property
    def mcp_config(self) -> dict:
        """MCP configuration for connecting agents."""
        config = {
            "cli": {
                "type": "stdio",
                "command": "python",
                "args": ["-m", "interp_infra.mcps.cli"],
                "env": {"SANDBOX_ID": self.sandbox.sandbox_id}
            }
        }

        for endpoint in self._mcp_endpoints:
            config.update(endpoint)

        return config


def create_cli_session(sandbox: Sandbox, name: str = "cli-session") -> CLISession:
    """Create a CLI session and prepare environment with models/repos."""
    if sandbox.config.execution_mode.value != "cli":
        raise RuntimeError("Sandbox must be in CLI mode. Use ExecutionMode.CLI.")

    print(f"Creating CLI session: {name}")

    session = CLISession(sandbox=sandbox, session_id=name)

    for handle in sandbox._model_handles:
        _prepare_model(session, handle)

    for handle in sandbox._repo_handles:
        _prepare_repo(session, handle)

    print(f"CLI session ready: {name}")
    return session


def _prepare_model(session: CLISession, handle: ModelHandle):
    """Make model available in the environment."""
    model_name = "<hidden>" if handle.hidden else handle.name
    print(f"  Preparing model: {model_name}")

    load_script = ModelLoader.generate_code(handle, target="script")
    script_name = "load_model.py" if handle.hidden else f"load_{handle.name.replace('/', '_').replace('.', '_')}.py"

    session.sandbox.modal_sandbox.open(f"/workspace/{script_name}", "w").write(load_script)
    session.exec(f"chmod +x /workspace/{script_name}")


def _prepare_repo(session: CLISession, handle: RepoHandle):
    """Make repo available in the environment."""
    print(f"  Preparing repo: {handle.local_path}")

    repo_name = handle.url.split("/")[-1].replace(".git", "")
    readme_content = f"Repository: {repo_name}\nLocation: {handle.local_path}\nURL: {handle.url}\n"

    if handle.container_running:
        readme_content += f"\nContainer: {handle.container_name} (running)"

        helper_script = f'''#!/usr/bin/env python3
"""Helper to execute commands in {handle.container_name} container."""

import subprocess
import sys

def container_exec(cmd: str) -> str:
    """Run command in the container."""
    result = subprocess.run(
        ["docker", "exec", "{handle.container_name}", "bash", "-c", cmd],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return result.stdout

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: exec_{handle.container_name}.py <command>")
        sys.exit(1)
    cmd = " ".join(sys.argv[1:])
    print(container_exec(cmd))
'''
        session.sandbox.modal_sandbox.open(f"/workspace/exec_{handle.container_name}.py", "w").write(helper_script)
        session.exec(f"chmod +x /workspace/exec_{handle.container_name}.py")
        readme_content += f"\nHelper script: /workspace/exec_{handle.container_name}.py"

    session.sandbox.modal_sandbox.open(f"/workspace/README_{repo_name}.txt", "w").write(readme_content)
