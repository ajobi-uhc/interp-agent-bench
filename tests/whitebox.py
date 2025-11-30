"""Basic sandbox with model."""

from interp_infra.environment import Sandbox, SandboxConfig, ExecutionMode
from conftest import auto_cleanup

config = SandboxConfig(
    python_packages=["torch", "transformers", "nnsight"],
    gpu="H100",
    execution_mode=ExecutionMode.NOTEBOOK,
)

sandbox = Sandbox(config)
sandbox.prepare_model("google/gemma-2-9b")

with auto_cleanup(sandbox):
    sandbox.start(name="whitebox")
    print(f"Jupyter: {sandbox.jupyter_url}")
