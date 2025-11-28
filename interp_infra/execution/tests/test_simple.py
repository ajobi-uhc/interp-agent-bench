"""Minimal test showing basic usage."""

from interp_infra.environment.sandbox import Sandbox, SandboxConfig
from interp_infra.execution.notebook_session import create_notebook_session

# Setup sandbox (torch, transformers, accelerate included with notebook=True)
config = SandboxConfig(
    gpu="H100",
    notebook=True,
)
sandbox = Sandbox(config)
model = sandbox.prepare_model("google/gemma-2-9b", hidden=True)
sandbox.start(name="simple-test")

# Check Jupyter logs first
print("\nChecking Jupyter logs...")
try:
    logs = sandbox.exec("tail -50 /var/log/jupyter.log")
    print("Jupyter logs:")
    print(logs)
except Exception as e:
    print(f"Could not get logs: {e}")

# Create session - loads model into kernel
try:
    session = create_notebook_session(sandbox)

    # Session is ready for agent
    print("MCP Config:")
    print(session.mcp_config)
except Exception as e:
    print(f"\nError creating session: {e}")
    print("\nJupyter logs:")
    print(sandbox.exec("tail -100 /var/log/jupyter.log"))
    raise

# Cleanup
sandbox.terminate()
