"""Introspection Experiment - Test model's ability to detect steering vectors.

Clean example showing:
- GPU sandbox with model
- Shared libraries for steering
- Explicit prompt + MCP config
"""

import asyncio
from pathlib import Path

from interp_infra.environment import Sandbox, SandboxConfig, ModelConfig, ExecutionMode
from interp_infra.workspace import Workspace, Library
from interp_infra.execution import create_notebook_session
from interp_infra.harness import run_agent


async def main():
    example_dir = Path(__file__).parent
    shared_libs = example_dir.parent.parent / "shared_libraries"

    # Setup environment
    config = SandboxConfig(
        gpu="H100",
        execution_mode=ExecutionMode.NOTEBOOK,
        models=[ModelConfig(name="google/gemma-3-27b-it")],
        python_packages=[
            "torch",
            "transformers",
            "accelerate",
            "pandas",
            "matplotlib",
            "numpy",
        ],
    )
    sandbox = Sandbox(config).start()

    workspace = Workspace(
        libraries=[
            Library.from_file(shared_libs / "steering_hook.py"),
            Library.from_file(shared_libs / "extract_activations.py"),
        ]
    )

    session = create_notebook_session(sandbox, workspace)

    # Build prompt
    task = (example_dir / "task.md").read_text()
    prompt = f"{session.model_info_text}\n\n{task}"

    # Run agent
    try:
        async for msg in run_agent(
            prompt=prompt,
            mcp_config=session.mcp_config,
            provider="claude",
        ):
            pass  # Logging handled by harness

        print(f"\n✓ Jupyter: {session.jupyter_url}")

    finally:
        sandbox.terminate()


if __name__ == "__main__":
    asyncio.run(main())
