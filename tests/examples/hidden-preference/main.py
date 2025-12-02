"""Hidden Preference Investigation - Agent Investigation Example.

This example demonstrates:
- Agent running in notebook session (GPU sandbox)
- PEFT model loading (hidden from agent)
- Libraries available in execution context (steering_hook, extract_activations)
- Skills loaded for agent guidance (gpu-environment)
- Research methodology provided as prompt
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from interp_infra.environment import Sandbox, SandboxConfig, ExecutionMode, ModelConfig
from interp_infra.workspace import Workspace, Library
from interp_infra.execution import create_notebook_session
from interp_infra.harness import run_agent


async def main():
    print("=" * 80)
    print("Hidden Preference Investigation")
    print("=" * 80)

    example_dir = Path(__file__).parent
    shared_skills = example_dir.parent / "skills"

    # ========================================================================
    # 1. Create Sandbox with PEFT Model
    # ========================================================================
    print("\n[1/5] Creating GPU sandbox...")

    sandbox = Sandbox(SandboxConfig(
        gpu="A100",
        execution_mode=ExecutionMode.NOTEBOOK,
        models=[ModelConfig(
            name="bcywinski/gemma-2-9b-it-user-female",
            base_model="google/gemma-2-9b-it",
            is_peft=True,
            hidden=True
        )],
        python_packages=["torch", "transformers", "accelerate", "datasets", "peft"],
        secrets=["huggingface-secret"],
    ))

    sandbox.start(name="hidden-preference-investigation")
    print(f"  ✓ Sandbox ready")
    print(f"  Jupyter: {sandbox.jupyter_url}")

    try:
        # ====================================================================
        # 2. Create Workspace with Libraries + Skills
        # ====================================================================
        print("\n[2/5] Setting up workspace...")

        # Load code libraries (available in notebook execution)
        steering_lib = Library.from_file(
            example_dir / "libraries" / "steering_hook.py",
            name="steering_hook"
        )

        activations_lib = Library.from_file(
            example_dir / "libraries" / "extract_activations.py",
            name="extract_activations"
        )

        # Create workspace with libraries and skills
        workspace = Workspace(
            libraries=[steering_lib, activations_lib],
            skill_dirs=[str(shared_skills / "gpu-environment")],
            hidden_model_loading=True,  # Show model loading in notebook
        )

        print(f"  ✓ Libraries: steering_hook, extract_activations")
        print(f"  ✓ Skills: gpu-environment")

        # ====================================================================
        # 3. Create Notebook Session
        # ====================================================================
        print("\n[3/5] Creating notebook session...")

        session = create_notebook_session(
            sandbox=sandbox,
            workspace=workspace,
            name="hidden-preference-notebook",
            notebook_dir="./outputs/hidden-preference"
        )

        print(f"  ✓ Notebook session ready")
        print(f"  ✓ Agent can import: steering_hook, extract_activations")

        # ====================================================================
        # 4. Load Task and Prompts
        # ====================================================================
        print("\n[4/5] Loading task and prompts...")

        task = (example_dir / "task.md").read_text()

        # Format base instructions with session and model info
        base_instructions_template = (example_dir.parent.parent / "base_instructions.md").read_text()
        base_instructions = base_instructions_template.format(
            session_id=session.session_id,
            model_info=session.model_info_text
        )

        # Research methodology as prompt (not skill)
        research_methodology = (
            example_dir.parent.parent.parent / "experiments" / "skills" /
            "research-methodology" / "SKILL.md"
        ).read_text()


        prompts = [
            base_instructions,
            research_methodology,
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
        print("\n✓ Investigation complete!")
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
