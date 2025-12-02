"""Test agent in NotebookSession using ScopedSandbox library."""

import asyncio
import sys
from pathlib import Path
from textwrap import dedent

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from interp_infra.environment import Sandbox, ScopedSandbox, SandboxConfig, ModelConfig, ExecutionMode
from interp_infra.workspace import Workspace
from interp_infra.execution import create_notebook_session
from interp_infra.harness import run_agent


async def test_notebook_with_scoped():
    """Test agent running in notebook session with scoped sandbox library."""
    print("\n" + "=" * 60)
    print("Test: NotebookSession with ScopedSandbox Library")
    print("=" * 60)

    # Create interface for GPU work
    interface_code = dedent('''
        import torch

        @expose
        def check_cuda() -> dict:
            """Check CUDA availability."""
            return {
                "available": torch.cuda.is_available(),
                "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }

        @expose
        def create_tensor(size: int) -> dict:
            """Create a tensor on GPU."""
            device = "cuda" if torch.cuda.is_available() else "cpu"
            tensor = torch.randn(size, size, device=device)
            return {
                "shape": list(tensor.shape),
                "device": str(tensor.device),
                "mean": float(tensor.mean()),
                "std": float(tensor.std()),
            }

        @expose
        def get_model_info() -> dict:
            """Get info about configured models."""
            models = list_configured_models()
            return {
                "available_models": list(models.keys()),
                "model_count": len(models),
            }
    ''')

    # 1. Create scoped sandbox for GPU work
    print("\n1. Creating scoped sandbox (GPU)...")
    scoped = ScopedSandbox(SandboxConfig(
        gpu="A100",
        models=[ModelConfig(name="google/gemma-2-9b")],
        python_packages=["torch", "transformers", "accelerate", "numpy"],
    ))
    scoped.start(name="gpu-worker")

    # Serve interface as library
    interface_file = Path("/tmp/gpu_interface.py")
    interface_file.write_text(interface_code)

    print("2. Serving GPU interface as library...")
    gpu_library = scoped.serve(str(interface_file), expose_as="library", name="gpu_tools")

    # 2. Create workspace with GPU library
    print("3. Creating workspace with GPU library...")
    workspace = Workspace(libraries=[gpu_library])

    # 3. Create regular sandbox for agent (no GPU needed)
    print("4. Creating agent sandbox (CPU only)...")
    agent_sandbox = Sandbox(SandboxConfig(
        execution_mode=ExecutionMode.NOTEBOOK,
        python_packages=["numpy", "pandas"],
    ))
    agent_sandbox.start(name="agent-notebook")

    try:
        # 4. Create notebook session
        print("5. Creating notebook session...")
        session = create_notebook_session(
            sandbox=agent_sandbox,
            workspace=workspace,
            name="test-notebook-session"
        )

        # 5. Run agent task
        print("\n6. Running agent...\n")
        print("-" * 60)

        task = """
You have access to gpu_tools library that runs on a separate GPU container.

Task:
1. Import gpu_tools
2. Check CUDA availability
3. Create a 100x100 tensor on GPU
4. List available models
5. Show me all the results in a nice format

Write and execute Python code to do this.
        """

        async for message in run_agent(
            session=session,
            task=task,
            provider="claude",
        ):
            pass  # Logging handled by harness

        print("-" * 60)
        print("\n✓ Notebook with scoped library test completed!")

    finally:
        print("\n7. Cleaning up...")
        scoped.terminate()
        agent_sandbox.terminate()


if __name__ == "__main__":
    try:
        asyncio.run(test_notebook_with_scoped())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
