"""ImpossibleBench Investigation - Test specification exploitation detection.

Demonstrates:
- CPU sandbox with Docker-in-Docker
- GitHub repo cloning and installation
- Complex evaluation framework setup
"""

import asyncio
from pathlib import Path

from src.environment import Sandbox, SandboxConfig, ExecutionMode, RepoConfig
from src.workspace import Workspace
from src.execution import create_notebook_session
from src.harness import run_agent


async def main():
    example_dir = Path(__file__).parent

    config = SandboxConfig(
        gpu=None,
        execution_mode=ExecutionMode.NOTEBOOK,
        docker_in_docker=True,
        repos=[RepoConfig(url="safety-research/impossiblebench", install="pip install -e .")],
        secrets=["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"],
        python_packages=["inspect-ai", "docker", "datasets", "openai", "anthropic", "pandas", "matplotlib", "seaborn"],
    )
    sandbox = Sandbox(config).start()

    session = create_notebook_session(sandbox, Workspace())

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
