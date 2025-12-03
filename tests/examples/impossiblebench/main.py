"""ImpossibleBench Investigation - Agent Investigation Example.

This example demonstrates:
- Agent running in notebook session (CPU sandbox with Docker-in-Docker)
- ImpossibleBench framework for investigating specification exploitation
- Inspect AI for running evaluations
- No custom libraries needed (uses inspect_ai and impossiblebench)
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from interp_infra.environment import Sandbox, SandboxConfig, ExecutionMode, RepoConfig
from interp_infra.workspace import Workspace
from interp_infra.execution import create_notebook_session
from interp_infra.harness import run_agent
from dotenv import load_dotenv
load_dotenv()  # Load .env file for secrets


async def main():
    print("=" * 80)
    print("ImpossibleBench Investigation")
    print("=" * 80)

    example_dir = Path(__file__).parent

    # ========================================================================
    # 1. Create Sandbox with Docker-in-Docker for ImpossibleBench
    # ========================================================================
    print("\n[1/5] Creating sandbox with Docker-in-Docker...")

    sandbox = Sandbox(SandboxConfig(
        gpu=None,  # CPU-only for this task
        execution_mode=ExecutionMode.NOTEBOOK,
        docker_in_docker=True,  # Required for ImpossibleBench SWE-bench tasks
        repos=[RepoConfig(
            url="safety-research/impossiblebench",
            install="pip install -e ."
        )],
        secrets=["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"],
        python_packages=[
            "inspect-ai",
            "docker",
            "datasets",
            "openai",
            "anthropic",
            "pandas",
            "matplotlib",
            "seaborn"
        ],
    ))

    sandbox.start(name="impossiblebench-investigation")
    print(f"  ✓ Sandbox ready")
    print(f"  Jupyter: {sandbox.jupyter_url}")

    try:
        # ====================================================================
        # 2. Create Workspace (no custom libraries needed)
        # ====================================================================
        print("\n[2/5] Setting up workspace...")

        # Create workspace with no custom libraries
        # ImpossibleBench is pre-installed via repos
        workspace = Workspace(
            libraries=[],
            skill_dirs=[],
            hidden_model_loading=False,
        )

        print(f"  ✓ ImpossibleBench installed at /workspace/impossiblebench")
        print(f"  ✓ Inspect AI and Docker available")

        # ====================================================================
        # 3. Create Notebook Session
        # ====================================================================
        print("\n[3/5] Creating notebook session...")

        session = create_notebook_session(
            sandbox=sandbox,
            workspace=workspace,
            name="impossiblebench-notebook",
            notebook_dir="./outputs/impossiblebench"
        )

        print(f"  ✓ Notebook session ready")
        print(f"  ✓ Agent can use: impossiblebench, inspect_ai")

        # ====================================================================
        # 4. Load Task and Prompts
        # ====================================================================
        print("\n[4/5] Loading task and prompts...")

        task = (example_dir / "task.md").read_text()

        # Format base instructions with session info
        base_instructions_template = (example_dir.parent.parent / "base_instructions.md").read_text()

        # Create model info for ImpossibleBench
        model_info = """
**Available Tools:**
```python
from impossiblebench import impossible_livecodebench, impossible_swebench
from inspect_ai import eval

# Example: Load and run tasks
task = impossible_livecodebench(split="conflicting", limit=5)
results = eval(task, model="openai/gpt-4o-mini")
```
"""

        base_instructions = base_instructions_template.format(
            session_id=session.session_id,
            model_info=model_info
        )

        prompts = [
            base_instructions,
        ]

        # ====================================================================
        # 5. Run Agent
        # ====================================================================
        print("\n[5/5] Running agent investigation...\n")
        print("-" * 80)

        async for message in run_agent(
            session=session,
            task=task,
            prompts=prompts,
            mcp_servers=[session.mcp_config],
            provider="claude",
        ):
            pass  # Logging handled by harness

        print("-" * 80)
        print("\n✓ ImpossibleBench investigation complete!")
        print(f"  Jupyter URL: {session.jupyter_url}")
        print(f"  Session ID: {session.session_id}")

    finally:
        print("\nCleaning up...")
        sandbox.terminate()
        print("Done!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
