"""Unknown entity experiment - steering with Gemma Scope SAEs."""

import asyncio
import sys
import signal
from pathlib import Path

# Add parent directory to path so we can import interp_infra
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from interp_infra.environment import Sandbox, SandboxConfig, ExecutionMode, ModelConfig
from interp_infra.execution import create_notebook_session
from interp_infra.harness import run_agent
from interp_infra import Extension

# Global sandbox reference for cleanup
sandbox_instance = None

def cleanup_handler(signum, frame):
    """Handle Ctrl+C and ensure cleanup runs."""
    print("\n\nReceived interrupt signal. Cleaning up...")
    if sandbox_instance:
        try:
            sandbox_instance.terminate()
            print("Cleanup complete!")
        except Exception as e:
            print(f"Error during cleanup: {e}")
    sys.exit(0)

async def main():
    global sandbox_instance

    # Load task and system prompt
    script_dir = Path(__file__).parent
    task_text = (script_dir / "task.md").read_text()
    system_prompt = (script_dir.parent / "base_instructions.md").read_text()

    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, cleanup_handler)

    # Create sandbox
    print("Setting up environment...")
    sandbox = Sandbox(
        SandboxConfig(
            gpu="A100",
            python_packages=["sae-lens"],
            gpu_count=1,
            execution_mode=ExecutionMode.NOTEBOOK,
            models=[
                ModelConfig(name="google/gemma-2-2b-it", var_name="model_it", hidden=False),
                ModelConfig(name="google/gemma-2-2b", var_name="model_base", hidden=False),
            ],
        )
    )
    sandbox_instance = sandbox

    # Start sandbox
    sandbox.start(name="entity-unknown-steering")

    print(f"Sandbox ready: {sandbox.sandbox_id}")
    print(f"Jupyter: {sandbox.jupyter_url}")

    # Create notebook session (loads models into kernel)
    session = create_notebook_session(sandbox, name="entity_driver", notebook_dir="./outputs/entity")

    # Generate model info for system prompt
    model_info = """**Models:**
- `model_it` (google/gemma-2-2b-it) - Instruction-tuned model for chat/generation
  - Tokenizer: `model_it_tokenizer`
- `model_base` (google/gemma-2-2b) - Base model for latent discovery
  - Tokenizer: `model_base_tokenizer`

**Libraries:**
- `sae_lens` - Library for loading Gemma Scope SAEs (already installed)"""

    # Inject session info into system prompt
    system_prompt = system_prompt.format(
        session_id=session.session_id,
        jupyter_url=session.jupyter_url,
        model_info=model_info
    )

    # Load extensions
    print("\nLoading extensions...")
    skills_dir = script_dir.parent / "skills"
    extensions = [
        Extension.from_dir(skills_dir / "gpu-environment"),
        Extension.from_dir(skills_dir / "research-methodology"),
        Extension.from_dir(skills_dir / "steering-hook"),
    ]

    for ext in extensions:
        session.add(ext)

    # Save prompts to outputs directory
    outputs_dir = Path("./outputs/entity")
    outputs_dir.mkdir(exist_ok=True, parents=True)

    full_system_prompt = system_prompt + "\n\n" + session.system_prompt
    (outputs_dir / "system_prompt.md").write_text(full_system_prompt)
    (outputs_dir / "task.md").write_text(task_text)
    print(f"\nSaved prompts to {outputs_dir}/")

    # Run agent
    print("\nRunning agent...\n")
    try:
        async for message in run_agent(
            mcp_config=session.mcp_config,
            system_prompt=full_system_prompt,
            task=task_text,
            provider="claude",
        ):
            pass  # Logging handled by harness

        print("\nExperiment complete!")

    except KeyboardInterrupt:
        print("\n\nReceived Ctrl+C during execution...")
    finally:
        # Cleanup always runs
        print("\nCleaning up...")
        try:
            sandbox.terminate()
            print("Done!")
        except Exception as e:
            print(f"Error during cleanup: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
