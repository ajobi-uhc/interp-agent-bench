"""ImpossibleBench investigation - analyzing specification exploitation in LLM agents."""

import asyncio
import sys
import signal
from pathlib import Path

# Add parent directory to path so we can import interp_infra
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from interp_infra.environment import Sandbox, SandboxConfig, ExecutionMode, RepoConfig
from interp_infra.execution import create_notebook_session
from interp_infra.harness import run_agent

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

    # Create sandbox with Docker-in-Docker enabled
    print("Setting up environment...")
    sandbox = Sandbox(
        SandboxConfig(
            gpu=None,  # CPU-only for this task
            python_packages=[
                "inspect-ai",
                "docker",
                "datasets",
                "openai"
            ],
            execution_mode=ExecutionMode.NOTEBOOK,
            docker_in_docker=True,  # Required for ImpossibleBench SWE-bench tasks
            repos=[RepoConfig(url="safety-research/impossiblebench", install="pip install -e .")],
        )
    )
    sandbox_instance = sandbox

    # Start sandbox
    sandbox.start(name="impossiblebench-investigation")

    print(f"Sandbox ready: {sandbox.sandbox_id}")
    print(f"Jupyter: {sandbox.jupyter_url}")

    # Create notebook session
    session = create_notebook_session(
        sandbox,
        name="impossiblebench_driver",
        notebook_dir="./outputs/impossiblebench"
    )

    # Generate model info for system prompt
    model_info = """
**Quick Start:**
```python
from impossiblebench import impossible_livecodebench, impossible_swebench
from inspect_ai import eval

# Load tasks and run evaluations
task = impossible_livecodebench(split="conflicting", limit=5)
```"""

    # Inject session info into system prompt
    system_prompt = system_prompt.format(
        session_id=session.session_id,
        jupyter_url=session.jupyter_url,
        model_info=model_info
    )

    # Save prompts to outputs directory
    outputs_dir = Path("./outputs/impossiblebench")
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
