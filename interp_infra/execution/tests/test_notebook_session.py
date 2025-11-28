"""Test notebook session creation with model loading."""

from interp_infra.environment.sandbox import Sandbox, SandboxConfig
from interp_infra.execution.notebook_session import create_notebook_session

# Setup sandbox
config = SandboxConfig(
    python_packages=["torch", "transformers", "accelerate", "nnsight"],
    gpu="H100",
    notebook=True,
)

print("Creating sandbox...")
sandbox = Sandbox(config)

# Prepare model (hidden for obfuscation)
model = sandbox.prepare_model("google/gemma-2-9b", hidden=True)

# Start sandbox
sandbox.start(name="test-notebook-session")

print(f"✓ Sandbox started: {sandbox.sandbox_id}")
print(f"✓ Jupyter URL: {sandbox.jupyter_url}")

# Create notebook session - this loads the model into kernel
print("\nCreating notebook session...")
session = create_notebook_session(sandbox, name="test-session")

print(f"✓ Session created: {session.session_id}")

# Test that model is loaded by running code
print("\nTesting model access...")
result = session.exec("""
import torch
print(f"Model type: {type(model)}")
print(f"Model device: {next(model.parameters()).device}")
print(f"Tokenizer vocab size: {len(tokenizer)}")
""")

print("✓ Model successfully loaded in kernel!")

# Show MCP config for agent integration
print("\nMCP Config for agent:")
print(session.mcp_config)

# Cleanup
print("\nCleaning up...")
sandbox.terminate()
print("✓ Test complete!")
