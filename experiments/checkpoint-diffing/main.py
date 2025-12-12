"""Checkpoint Diffing - Compare model checkpoints using SAE-based analysis.

Demonstrates:
- GPU sandbox with SAE inference
- External repo cloning (interp_embed)
- API access for data generation
- Complex analysis pipeline
"""

import asyncio
from pathlib import Path

from src.environment import Sandbox, SandboxConfig, ExecutionMode, RepoConfig
from src.workspace import Workspace, Library
from src.execution import create_notebook_session
from src.harness import run_agent


async def main():
    example_dir = Path(__file__).parent

    config = SandboxConfig(
        gpu="A100",
        execution_mode=ExecutionMode.NOTEBOOK,
        repos=[RepoConfig(url="nickjiang2378/interp_embed")],
        system_packages=["git"],
        python_packages=[
            "torch", "transformers", "accelerate", "pandas", "numpy", "scipy",
            "google-generativeai", "datasets", "matplotlib", "seaborn", "google-genai",
            "goodfire", "python-dotenv", "scikit-learn", "tqdm", "openai",
            "sae-lens", "transformer-lens", "huggingface-hub", "pydantic", "bitsandbytes",
        ],
        secrets=["GEMINI_API_KEY", "OPENAI_KEY", "OPENROUTER_API_KEY", "HF_TOKEN"],
        timeout=7200,
    )
    sandbox = Sandbox(config).start()

    workspace = Workspace(libraries=[Library.from_file(example_dir / "openrouter_client.py")])

    session = create_notebook_session(sandbox, workspace)

    task = (example_dir / "task.md").read_text()
    prompt = f"{session.model_info_text}\n\n{workspace.get_library_docs()}\n\n{task}\n\nUse the notebook MCP server as your main execution environment."

    try:
        async for msg in run_agent(prompt=prompt, mcp_config=session.mcp_config, provider="claude"):
            pass

        print(f"\n✓ Jupyter: {session.jupyter_url}")

    finally:
        sandbox.terminate()


if __name__ == "__main__":
    asyncio.run(main())
