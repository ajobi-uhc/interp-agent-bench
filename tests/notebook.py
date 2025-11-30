"""Notebook session with model loading."""

from interp_infra.environment import Sandbox, SandboxConfig, ExecutionMode, ModelConfig
from interp_infra.execution import create_notebook_session
from conftest import auto_cleanup

config = SandboxConfig(
    python_packages=["torch", "transformers", "nnsight"],
    gpu="H100",
    execution_mode=ExecutionMode.NOTEBOOK,
    models=[ModelConfig(name="google/gemma-2-9b", hidden=True)],
)

sandbox = Sandbox(config)

with auto_cleanup(sandbox):
    sandbox.start(name="notebook")

    session = create_notebook_session(sandbox)

    # Test model access
    session.exec("""
print(f"Model: {type(model).__name__}")
print(f"Device: {next(model.parameters()).device}")
""")

    print(f"Session ready: {session.session_id}")
