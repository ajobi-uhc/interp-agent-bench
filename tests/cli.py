"""CLI session with model loading and command execution."""

from interp_infra.environment import Sandbox, SandboxConfig, ExecutionMode, ModelConfig
from interp_infra.execution import create_cli_session
from conftest import auto_cleanup

config = SandboxConfig(
    python_packages=["torch", "transformers"],
    gpu="H100",
    execution_mode=ExecutionMode.CLI,
    models=[ModelConfig(name="google/gemma-2-9b", hidden=True)],
)

sandbox = Sandbox(config)

with auto_cleanup(sandbox):
    sandbox.start(name="cli")

    session = create_cli_session(sandbox)

    # Test shell command execution
    result = session.exec("echo 'Hello from CLI'")
    print(f"Shell output: {result}")

    # Test Python execution
    result = session.exec_python("""
print("Python execution test")
import torch
print(f"PyTorch version: {torch.__version__}")
""")
    print(f"Python output: {result}")

    # Test that model load script exists
    result = session.exec("ls -la /workspace/load_model.py")
    print(f"Model script exists: {result}")

    print(f"CLI session ready: {session.session_id}")
