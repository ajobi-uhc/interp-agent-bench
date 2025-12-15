"""Notebook Intro - Learn the basics of working with models in a Jupyter notebook.

Demonstrates:
- GPU sandbox setup with a simple model
- Jupyter notebook execution environment
- Basic model interactions and exploration
"""

import asyncio
from pathlib import Path

from src.environment import Sandbox, SandboxConfig, ExecutionMode, ModelConfig
from src.workspace import Workspace
from src.execution import create_notebook_session
from src.harness import run_agent


async def main():
    example_dir = Path(__file__).parent

    config = SandboxConfig(
        gpu="A100",
        execution_mode=ExecutionMode.NOTEBOOK,
        models=[ModelConfig(name="google/gemma-2-2b-it")],
        python_packages=["torch", "transformers", "accelerate"],
        secrets=["HF_TOKEN"],
    )
    sandbox = Sandbox(config).start()

    workspace = Workspace(libraries=[])
    session = create_notebook_session(sandbox, workspace)

    task = (example_dir / "task.md").read_text()
    prompt = f"{session.model_info_text}\n\n{task}"

    try:
        async for msg in run_agent(prompt=prompt, mcp_config=session.mcp_config, provider="claude"):
            pass

        print(f"\n✓ Jupyter: {session.jupyter_url}")

    finally:
        sandbox.terminate()


if __name__ == "__main__":
    asyncio.run(main())
