"""Local Workspace with RPC - Agent runs locally with RPC calls to GPU sandbox.

Demonstrates:
- ScopedSandbox for GPU isolation
- RPC library serving from GPU
- Local session with workspace (no MCP, direct Python execution)
"""

import asyncio
from pathlib import Path

from src.environment import ScopedSandbox, SandboxConfig, ModelConfig
from src.workspace import Workspace, Library
from src.execution import create_local_session
from src.harness import run_agent


async def main():
    example_dir = Path(__file__).parent

    scoped = ScopedSandbox(SandboxConfig(
        gpu="A100",
        models=[ModelConfig(name="google/gemma-2-9b")],
        python_packages=["torch", "transformers", "accelerate"],
        secrets=["HF_TOKEN"],
    ))

    scoped.start()

    model_tools = scoped.serve(
        str(example_dir / "interface.py"),
        expose_as="library",
        name="model_tools"
    )

    workspace = Workspace(libraries=[
        Library.from_file(example_dir / "helpers.py"),
        model_tools,
    ])

    session = create_local_session(
        workspace=workspace,
        workspace_dir=str(example_dir / "workspace"),
        name="minimal-example"
    )

    task = """
You have two libraries available:

1. `helpers` - formatting utilities (runs locally)
2. `model_tools` - model analysis tools (runs on GPU via RPC)

Task:
- Import and call model_tools.get_model_info() to see model specs
- Import and call model_tools.get_embedding("hello world")
- Import and use helpers.format_result() to format the output
- Show me the formatted results

Write and execute Python code to do this.
"""

    try:
        async for message in run_agent(
            prompt=task,
            mcp_config={},  # LocalSession uses default tools only
            provider="claude"
        ):
            pass

        print(f"\n✓ Session: {session.name}")

    finally:
        scoped.terminate()


if __name__ == "__main__":
    asyncio.run(main())
