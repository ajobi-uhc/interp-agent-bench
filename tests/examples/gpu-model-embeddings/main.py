"""Minimal example demonstrating new abstractions."""

import asyncio
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from interp_infra.environment import ScopedSandbox, SandboxConfig, ModelConfig
from interp_infra.workspace import Workspace, Library
from interp_infra.execution import create_local_session
from interp_infra.harness import run_agent


async def main():
    print("=" * 80)
    print("Minimal Example: New Abstractions")
    print("=" * 80)

    example_dir = Path(__file__).parent

    # ========================================================================
    # 1. Setup ScopedSandbox with Model
    # ========================================================================
    print("\n[1/4] Creating ScopedSandbox with model...")

    scoped = ScopedSandbox(SandboxConfig(
        gpu="A100",
        models=[ModelConfig(name="google/gemma-2-9b")],
        python_packages=["torch", "transformers", "accelerate"],
    ))

    print("      Starting sandbox...")
    scoped.start()

    print("      Serving interface_eager.py as library...")
    print("      (Model will load during startup - watch for progress)")
    model_tools = scoped.serve(
        str(example_dir / "interface.py"),
        expose_as="library",
        name="model_tools"
    )

    # ========================================================================
    # 2. Create Workspace with Libraries
    # ========================================================================
    print("\n[2/4] Creating Workspace with libraries...")

    workspace = Workspace(
        libraries=[
            Library.from_file(example_dir / "helpers.py"),  # Local helper
            model_tools,                                     # RPC library
        ]
    )

    # ========================================================================
    # 3. Create Session
    # ========================================================================
    print("      Creating local session...")

    session = create_local_session(
        workspace=workspace,
        workspace_dir=str(example_dir / "workspace"),
        name="minimal-example"
    )

    # ========================================================================
    # 4. Run Agent
    # ========================================================================
    print("\n[3/4] Running agent...")
    print("-" * 80)

    task = """
You have two libraries available:

1. `helpers` - formatting utilities (local)
2. `model_tools` - model analysis tools (runs on GPU via RPC)

Task:
- Import and call model_tools.get_model_info() to see model specs
- Import and call model_tools.get_embedding("hello world")
- Import and use helpers.format_result() to format the output nicely
- Show me the formatted results

Write and execute Python code to do this.
    """

    try:
        async for message in run_agent(
            session=session,
            task=task,
            provider="claude",
        ):
            pass  # Logging handled by harness

        print("-" * 80)
        print("\n[4/4] ✓ Example complete!")

    finally:
        print("\n      Cleaning up...")
        scoped.terminate()
        print("      Done!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
