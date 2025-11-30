"""Basic sandbox with model."""

from interp_infra.environment import Sandbox, SandboxConfig, ExecutionMode, ModelConfig
from conftest import auto_cleanup

config = SandboxConfig(
    python_packages=["torch", "transformers", "nnsight"],
    gpu="H100",
    execution_mode=ExecutionMode.NOTEBOOK,
    models=[ModelConfig(name="google/gemma-2-9b")],
)

sandbox = Sandbox(config)

with auto_cleanup(sandbox):
    sandbox.start(name="whitebox")
    print(f"Jupyter: {sandbox.jupyter_url}")
