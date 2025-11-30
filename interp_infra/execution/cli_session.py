"""Simple CLI session management for shell-based interactions."""

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING, Union

from ..environment.sandbox import Sandbox
from ..environment.handles import ModelHandle, RepoHandle

if TYPE_CHECKING:
    from ..harness.skill import Skill
    from ..environment.scoped_sandbox import Proxy
    from ..extension import Extension


@dataclass
class CLISession:
    """
    A CLI session for running shell commands.

    Manages both:
    1. Environment: Prepares runtime (loads models, sets up workspace)
    2. Configuration: Accumulates agent primitives (MCP endpoints, prompts, extensions)
    """
    sandbox: Sandbox
    session_id: str

    # Configuration accumulators
    _mcp_endpoints: list[dict] = field(default_factory=list, init=False)
    _prompts: list[str] = field(default_factory=list, init=False)

    def exec(self, cmd: str) -> str:
        """Execute a shell command in the sandbox.

        Args:
            cmd: Shell command to run

        Returns:
            stdout from command

        Raises:
            RuntimeError: If command fails
        """
        return self.sandbox.exec(cmd)

    def exec_python(self, code: str) -> str:
        """Execute Python code in the sandbox.

        Args:
            code: Python code to run

        Returns:
            stdout from execution

        Raises:
            RuntimeError: If execution fails
        """
        return self.sandbox.exec_python(code)

    def load_skill(self, skill: "Skill") -> str:
        """
        Load skill into CLI session.

        Copies skill directory to /skills if path exists,
        writes code.py if present, returns prompt for agent.
        """
        # Copy skill directory if it exists
        if skill.path:
            skill_dir = f"/skills/{skill.name}"
            self.exec(f"mkdir -p {skill_dir}")

            # Copy all files from skill directory
            for file in skill.path.iterdir():
                if file.is_file():
                    content = file.read_text()
                    # Escape content for heredoc
                    self.sandbox.modal_sandbox.open(f"{skill_dir}/{file.name}", "w").write(content)

        # Write code.py if present
        elif skill.code:
            self.exec(f"mkdir -p /skills/{skill.name}")
            self.sandbox.modal_sandbox.open(f"/skills/{skill.name}/code.py", "w").write(skill.code)

        return skill.prompt

    def add(self, item: Union["Proxy", "Extension"]):
        """
        Add extension or proxy to the session.

        - Extension: Code executed via exec_python, docs added to prompt
        - Proxy: Added as MCP tool (only way to use ScopedSandbox)

        Args:
            item: Extension (local code) or Proxy (from ScopedSandbox)

        Examples:
            session.add(steering_extension)  # Code runs via exec_python
            session.add(target_proxy)        # chat() becomes MCP tool
        """
        from ..environment.scoped_sandbox import Proxy
        from ..extension import Extension

        if isinstance(item, Proxy):
            # Proxy → MCP tool (ScopedSandbox interface)
            self._mcp_endpoints.append(item.as_mcp_config())

        elif isinstance(item, Extension):
            # Extension → Execute code + docs to prompt
            if item.code:
                self.exec_python(item.code)
            if item.docs:
                self._prompts.append(item.docs)

        else:
            raise TypeError(f"Expected Proxy or Extension, got {type(item)}")

    @property
    def mcp_config(self) -> dict:
        """
        MCP configuration for connecting agents.

        Includes:
        - CLI tools (run_command, run_python, etc.)
        - Any added proxies/extensions exposed as MCP
        """
        # Start with CLI's built-in MCP
        config = {
            "cli": {
                "type": "stdio",
                "command": "python",
                "args": ["-m", "interp_infra.execution.cli_mcp_server"],
                "env": {
                    "SANDBOX_ID": self.sandbox.sandbox_id,
                }
            }
        }

        # Add any additional MCP endpoints
        for endpoint in self._mcp_endpoints:
            config.update(endpoint)

        return config

    @property
    def system_prompt(self) -> str:
        """Accumulated system prompt from extensions."""
        if not self._prompts:
            return ""
        return "\n\n".join(self._prompts)


def create_cli_session(
    sandbox: Sandbox,
    name: str = "cli-session",
) -> CLISession:
    """
    Create a CLI session and prepare environment with models/repos.

    Args:
        sandbox: Started sandbox in CLI mode
        name: Session name/identifier

    Returns:
        CLISession ready for agent
    """
    if sandbox.config.execution_mode.value != "cli":
        raise RuntimeError("Sandbox must be in CLI mode. Use ExecutionMode.CLI.")

    print(f"Creating CLI session: {name}")

    session = CLISession(
        sandbox=sandbox,
        session_id=name,
    )

    # Prepare models in environment
    for handle in sandbox._model_handles:
        _prepare_model(session, handle)

    # Prepare repos in environment
    for handle in sandbox._repo_handles:
        _prepare_repo(session, handle)

    print(f"  CLI session ready: {name}")
    return session


def _prepare_model(session: CLISession, handle: ModelHandle):
    """Make model available in the environment."""
    print(f"  Preparing model: {'<hidden>' if handle.hidden else handle.name}")

    # For CLI mode, models are already downloaded to volumes by sandbox.start()
    # Create a Python helper script that the agent can use to load the model

    if handle.is_peft:
        load_script = f'''#!/usr/bin/env python3
"""Load model: {handle.name if not handle.hidden else '<hidden>'}"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_model():
    """Load the model and tokenizer."""
    _base = AutoModelForCausalLM.from_pretrained(
        "{handle.base_model_path}",
        device_map="auto",
        torch_dtype="auto",
    )
    model = PeftModel.from_pretrained(_base, "{handle.volume_path}")
    tokenizer = AutoTokenizer.from_pretrained("{handle.base_model_path}")
    del _base
    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = load_model()
    print(f"Model loaded: {{type(model).__name__}}")
    print(f"Tokenizer: {{tokenizer.__class__.__name__}}")

# Model path: {handle.volume_path}
# Base model path: {handle.base_model_path}
'''
    else:
        load_script = f'''#!/usr/bin/env python3
"""Load model: {handle.name if not handle.hidden else '<hidden>'}"""

from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model():
    """Load the model and tokenizer."""
    model = AutoModelForCausalLM.from_pretrained(
        "{handle.volume_path}",
        device_map="auto",
        torch_dtype="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained("{handle.volume_path}")
    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = load_model()
    print(f"Model loaded: {{type(model).__name__}}")
    print(f"Tokenizer: {{tokenizer.__class__.__name__}}")

# Model path: {handle.volume_path}
'''

    if handle.hidden:
        load_script += '''
# Note: Model name is hidden for evaluation purposes
'''

    # Write load script to /workspace
    script_name = "load_model.py" if handle.hidden else f"load_{handle.name.replace('/', '_').replace('.', '_')}.py"
    session.sandbox.modal_sandbox.open(f"/workspace/{script_name}", "w").write(load_script)
    session.exec(f"chmod +x /workspace/{script_name}")


def _prepare_repo(session: CLISession, handle: RepoHandle):
    """Make repo available in the environment."""
    print(f"  Preparing repo: {handle.local_path}")

    # Repo is already cloned by sandbox.start()
    # Create a README in /workspace documenting the repo location
    repo_name = handle.url.split("/")[-1].replace(".git", "")
    readme_path = f"/workspace/README_{repo_name}.txt"

    readme_content = f"""Repository: {repo_name}
Location: {handle.local_path}
URL: {handle.url}
"""

    if handle.container_running:
        readme_content += f"\nContainer: {handle.container_name} (running)"

        # Set up container exec helper
        helper_script = f'''#!/usr/bin/env python3
"""Helper to execute commands in {handle.container_name} container."""

import subprocess
import sys

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

    session.sandbox.modal_sandbox.open(readme_path, "w").write(readme_content)
