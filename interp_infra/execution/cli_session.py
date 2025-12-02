"""CLI session management."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import shutil

from ..environment.sandbox import Sandbox
from ..environment.utils import ModelHandle, RepoHandle
from ..workspace import Workspace
from .session_base import SessionBase


@dataclass
class CLISession(SessionBase):
    """A CLI session for shell-based interactions."""

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
        code = Path(file_path).read_text()
        return self.exec_python(code)

    def _mount_dir(self, src: str, dest: str):
        """Mount local directory into sandbox workspace."""
        # Would need to copy to sandbox - simplified for now
        print(f"Mounting {src} → {dest} (copy to sandbox)")

    def _mount_file(self, src: str, dest: str):
        """Mount local file into sandbox workspace."""
        src_path = Path(src)
        if not src_path.exists():
            raise FileNotFoundError(f"File not found: {src}")

        content = src_path.read_text()
        with self.sandbox.modal_sandbox.open(dest, "w") as f:
            f.write(content)

    def _copy_skill_dir(self, skill_dir: str):
        """Copy skill directory to workspace/.claude/skills/."""
        src_path = Path(skill_dir)
        if not src_path.exists():
            raise FileNotFoundError(f"Skill directory not found: {skill_dir}")

        dest = f"{self.workspace_path}/.claude/skills/{src_path.name}"
        self.exec(f"mkdir -p {dest}")

        # Copy SKILL.md if exists
        skill_md = src_path / "SKILL.md"
        if skill_md.exists():
            content = skill_md.read_text()
            with self.sandbox.modal_sandbox.open(f"{dest}/SKILL.md", "w") as f:
                f.write(content)

    def _execute_code(self, code: str):
        """Execute code in sandbox."""
        self.exec_python(code)


def create_cli_session(
    sandbox: Sandbox,
    workspace: Optional[Workspace] = None,
    name: str = "cli-session",
) -> CLISession:
    """
    Create a CLI session and setup workspace.

    Args:
        sandbox: Sandbox in CLI mode
        workspace: Workspace configuration
        name: Session name

    Returns:
        CLISession with workspace setup
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

    # Prepare models
    for handle in sandbox._model_handles:
        _prepare_model(session, handle)

    # Prepare repos
    for handle in sandbox._repo_handles:
        _prepare_repo(session, handle)

    # Setup workspace if provided
    if workspace:
        workspace.setup_in(session)

    print(f"CLI session ready: {name}")
    return session


def _prepare_model(session: CLISession, handle: ModelHandle):
    """Create model loading script in workspace."""
    model_name = "<hidden>" if handle.hidden else handle.name
    print(f"  Preparing model: {model_name}")

    var = handle.var_name
    tok_var = f"{var}_tokenizer" if var != "model" else "tokenizer"

    if handle.is_peft:
        script = f'''#!/usr/bin/env python3
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

_base = AutoModelForCausalLM.from_pretrained("{handle.base_model_path}", device_map="auto", torch_dtype="auto")
{var} = PeftModel.from_pretrained(_base, "{handle.volume_path}")
{tok_var} = AutoTokenizer.from_pretrained("{handle.base_model_path}")
del _base
'''
    else:
        script = f'''#!/usr/bin/env python3
from transformers import AutoModelForCausalLM, AutoTokenizer

{var} = AutoModelForCausalLM.from_pretrained("{handle.volume_path}", device_map="auto", torch_dtype="auto")
{tok_var} = AutoTokenizer.from_pretrained("{handle.volume_path}")
'''

    if handle.hidden:
        script += f'''
if hasattr({var}, "config"):
    {var}.config.name_or_path = "model"
'''

    script_name = "load_model.py" if handle.hidden else f"load_{handle.name.replace('/', '_')}.py"
    session.sandbox.modal_sandbox.open(f"/workspace/{script_name}", "w").write(script)
    session.exec(f"chmod +x /workspace/{script_name}")


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
        session.sandbox.modal_sandbox.open(f"/workspace/exec_{handle.container_name}.py", "w").write(helper)
        session.exec(f"chmod +x /workspace/exec_{handle.container_name}.py")
        readme += f"Helper: /workspace/exec_{handle.container_name}.py\n"

    session.sandbox.modal_sandbox.open(f"/workspace/README_{repo_name}.txt", "w").write(readme)
