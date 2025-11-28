"""Simple test matching the user's example."""

from interp_infra.environment.sandbox import Sandbox, SandboxConfig

# Simple whitebox model investigation
config = SandboxConfig(
    python_packages=["torch", "transformers", "nnsight"],
    gpu="H100",
    notebook=True,
)

sandbox = Sandbox(config)
model = sandbox.prepare_model("google/gemma-2-9b")
sandbox.start(name="interpretability-research")

# jupyter_url is now available
print(sandbox.jupyter_url)

# Cleanup
print("\nCleaning up...")
sandbox.terminate()
print("Done!")
