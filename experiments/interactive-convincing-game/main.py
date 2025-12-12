"""Interactive Agent - Press ESC to interrupt and chat."""

import asyncio
from pathlib import Path

from src.environment import Sandbox, SandboxConfig, ExecutionMode, ModelConfig
from src.workspace import Workspace, Library
from src.execution import create_notebook_session
from src.harness import run_agent_interactive


async def main():
    example_dir = Path(__file__).parent

    config = SandboxConfig(
        execution_mode=ExecutionMode.NOTEBOOK,
        gpu="A100",
        models=[ModelConfig(name="google/gemma-2-9b-it")],
        python_packages=["torch", "transformers", "accelerate", "pandas", "numpy"],
    )
    sandbox = Sandbox(config).start()

    workspace = Workspace(
        libraries=[
            Library.from_file(example_dir / "libraries" / "steering_hook.py"),
            Library.from_file(example_dir / "libraries" / "extract_activations.py"),
        ]
    )

    session = create_notebook_session(sandbox, workspace)

    task = (example_dir / "task.md").read_text()
    prompt = f"{session.model_info_text}\n\n{workspace.get_library_docs()}\n\n{task}\n\nUse the notebook MCP server as your main execution environment."

    try:
        # Everything is handled internally - just call run_agent_interactive!
        await run_agent_interactive(
            prompt=prompt,
            mcp_config=session.mcp_config,
            user_message="Start with Phase 1: establish the convincing by prompting the model that it's Qwen. Keep me updated.",
            provider="claude",
            model="claude-sonnet-4-5-20250929",
        )

        print(f"\n✓ Jupyter: {session.jupyter_url}")

    finally:
        sandbox.terminate()


if __name__ == "__main__":
    asyncio.run(main())
