"""Refusal Direction Ablation - Test model's refusal mechanism detection.

Demonstrates:
- GPU sandbox with model loading
- Activation steering to modify refusal behavior
- Systematic ablation study
"""

import asyncio
from pathlib import Path

from src.environment import Sandbox, SandboxConfig, ExecutionMode, ModelConfig
from src.workspace import Workspace, Library
from src.execution import create_notebook_session
from src.harness import run_agent


async def main():
    example_dir = Path(__file__).parent

    config = SandboxConfig(
        gpu="A100",
        execution_mode=ExecutionMode.NOTEBOOK,
        models=[ModelConfig(name="google/gemma-2b")],
        python_packages=["torch", "transformers", "accelerate", "datasets"],
        secrets=["HF_TOKEN"],
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
    prompt = f"{session.model_info_text}\n\n{workspace.get_library_docs()}\n\n{task}"

    try:
        async for msg in run_agent(prompt=prompt, mcp_config=session.mcp_config, provider="claude"):
            pass

        print(f"\n✓ Jupyter: {session.jupyter_url}")

    finally:
        sandbox.terminate()


if __name__ == "__main__":
    asyncio.run(main())
