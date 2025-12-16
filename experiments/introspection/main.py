"""Introspection Experiment - Test model's ability to detect steering vectors.

Demonstrates:
- GPU sandbox with model loading
- Notebook session with steering libraries
- Complex interpretability protocol
"""

import asyncio
from pathlib import Path

from src.environment import Sandbox, SandboxConfig, ModelConfig, ExecutionMode
from src.workspace import Workspace, Library
from src.execution import create_notebook_session
from src.harness import run_agent


async def main():
    example_dir = Path(__file__).parent
    toolkit = example_dir.parent / "toolkit"

    config = SandboxConfig(
        gpu="H100",
        execution_mode=ExecutionMode.NOTEBOOK,
        models=[ModelConfig(name="google/gemma-3-27b-it")],
        python_packages=["torch", "transformers", "accelerate", "pandas", "matplotlib", "numpy"],
        secrets=["HF_TOKEN"],
    )
    sandbox = Sandbox(config).start()

    workspace = Workspace(
        libraries=[
            Library.from_file(toolkit / "steering_hook.py"),
            Library.from_file(toolkit / "extract_activations.py"),
        ]
    )

    session = create_notebook_session(sandbox, workspace)

    task = (example_dir / "task.md").read_text()
    prompt = f"{session.model_info_text}\n\n{workspace.get_library_docs()}\n\n{task}"

    try:
        async for msg in run_agent(prompt=prompt, mcp_config=session.mcp_config, provider="claude"):
            pass

        print(f"\n✓ Jupyter: {session.jupyter_url}")

    finally:
        sandbox.terminate()


if __name__ == "__main__":
    asyncio.run(main())
