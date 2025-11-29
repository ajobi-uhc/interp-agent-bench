"""Test notebook session with repo workspace setup."""

from interp_infra.environment.sandbox import Sandbox, SandboxConfig
from interp_infra.execution.notebook_session import create_notebook_session

# Setup sandbox with a repo
config = SandboxConfig(
    python_packages=["torch", "transformers", "accelerate"],
    gpu="H100",
    notebook=True,
)

print("Creating sandbox...")
sandbox = Sandbox(config)

# Prepare model and repo
model = sandbox.prepare_model("google/gemma-2-9b", hidden=True)
repo = sandbox.prepare_repo("PalisadeResearch/shutdown_avoidance")

# Start sandbox
sandbox.start(name="test-with-repo")

print(f"Sandbox started")
print(f"Repo at: {repo.local_path}")

# Create session - loads model and sets up repo workspace
session = create_notebook_session(sandbox, name="repo-test")

# Test that WORKSPACE is available
print("\nTesting workspace access...")
result = session.exec("""
print(f"Workspace: {WORKSPACE}")
print(f"Workspace exists: {WORKSPACE.exists()}")
print(f"Files: {list(WORKSPACE.glob('*.py'))[:3]}")
""")

print("Workspace successfully set up in kernel!")

# Cleanup
print("\nCleaning up...")
sandbox.terminate()
print("Test complete!")
