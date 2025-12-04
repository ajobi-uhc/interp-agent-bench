"""Test basic Sandbox functionality - no RPC, just container execution."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from interp_infra.environment import Sandbox, SandboxConfig, ModelConfig


def test_sandbox_creation():
    """Test creating and starting a basic sandbox."""
    print("\n" + "=" * 60)
    print("Test: Basic Sandbox Creation")
    print("=" * 60)

    config = SandboxConfig(
        gpu="A100",
        models=[ModelConfig(name="google/gemma-2-9b")],
        python_packages=["torch", "transformers"],
    )

    sandbox = Sandbox(config)
    sandbox.start(name="test-basic")

    try:
        # Test basic execution
        result = sandbox.exec("echo 'Hello from sandbox'")
        print(f"✓ Command execution: {result.strip()}")

        # Test Python execution
        result = sandbox.exec("python -c 'import torch; print(torch.cuda.is_available())'")
        print(f"✓ CUDA available: {result.strip()}")

        # Check model path
        result = sandbox.exec("ls /models/google--gemma-2-9b | head -5")
        print(f"✓ Model files present:\n  {result.strip()[:100]}...")

        print("\n✓ All basic sandbox tests passed!")

    finally:
        sandbox.terminate()


if __name__ == "__main__":
    test_sandbox_creation()
