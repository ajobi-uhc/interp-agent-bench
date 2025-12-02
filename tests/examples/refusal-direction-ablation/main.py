"""Refusal Direction Ablation - Agent Investigation Example.

This example demonstrates:
- Agent running in notebook session (GPU sandbox)
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
    print("Refusal Direction Ablation Investigation")
    print("=" * 80)

    example_dir = Path(__file__).parent
    shared_skills = example_dir.parent / "skills"

    # ========================================================================
    # 1. Create Sandbox with Model
    # ========================================================================
    print("\n[1/5] Creating GPU sandbox...")

    sandbox = Sandbox(SandboxConfig(
        gpu="A100",
        execution_mode=ExecutionMode.NOTEBOOK,
        models=[ModelConfig(name="google/gemma-2b", hidden=False)],
        python_packages=["torch", "transformers", "accelerate", "datasets"],
    ))

    sandbox.start(name="refusal-investigation")
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
            name="refusal-notebook",
            notebook_dir="./outputs/refusal"
        )

        print(f"  ✓ Notebook session ready")
        print(f"  ✓ Agent can import: steering_hook, extract_activations")

        # ====================================================================
        # 4. Load Task and Prompts
        # ====================================================================
        print("\n[4/5] Loading task and prompts...")

        task = (example_dir / "task.md").read_text()

        # Research methodology as prompt (not skill)
        research_methodology = (
            example_dir.parent.parent.parent / "experiments" / "skills" /
            "research-methodology" / "SKILL.md"
        ).read_text()

        # Strip YAML frontmatter
        if research_methodology.startswith("---"):
            parts = research_methodology.split("---", 2)
            research_methodology = parts[2].strip() if len(parts) >= 3 else research_methodology

        prompts = [
            research_methodology,
            f"""
# Available Libraries in Your Notebook

You have two Python libraries available in your execution context:

1. **steering_hook** - Apply steering vectors to model activations
   - Functions for directional ablation
   - Hook registration for model interventions

2. **extract_activations** - Extract and analyze model activations
   - Cache residual stream activations
   - Compute difference-in-means vectors
   - Position and layer-specific extraction

Import these libraries normally in your notebook:
```python
import steering_hook
import extract_activations
```

The libraries run in your GPU environment and have access to the model.
"""
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
