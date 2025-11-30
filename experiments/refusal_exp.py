"""Refusal is mediated by single direction - Simple Python version."""

import asyncio
import sys
import signal
from pathlib import Path

# Add parent directory to path so we can import interp_infra
sys.path.insert(0, str(Path(__file__).parent.parent))

from interp_infra.environment import Sandbox, SandboxConfig, ExecutionMode
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
    task_text = Path("experiments/task.md").read_text()
    system_prompt = Path("experiments/base_instructions.md").read_text()

    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, cleanup_handler)

    # Create sandbox
    print("Setting up environment...")
    sandbox = Sandbox(
        SandboxConfig(
            gpu="A100",
            gpu_count=1,
            execution_mode=ExecutionMode.NOTEBOOK,
        )
    )
    sandbox_instance = sandbox

    # Prepare model
    sandbox.prepare_model(
        "google/gemma-2b",
        hidden=False,
    )

    # Start sandbox
    sandbox.start(name="refusal-single-dir")

    print(f"Sandbox ready: {sandbox.sandbox_id}")
    print(f"Jupyter: {sandbox.jupyter_url}")

    # Create notebook session (loads model into kernel)
    session = create_notebook_session(sandbox, name="driver")

    # Inject session info into system prompt
    system_prompt = system_prompt.format(
        session_id=session.session_id,
        jupyter_url=session.jupyter_url
    )

    # Load extensions
    print("\nLoading extensions...")
    extensions = [
        Extension.from_file("experiments/skills/gpu-environment.md"),
        Extension.from_file("experiments/skills/research-methodology.md"),
        Extension.from_file("experiments/skills/extract-activations.md"),
        Extension.from_file("experiments/skills/steering-hook.md"),
    ]

    for ext in extensions:
        session.add(ext)

    # Run agent
    print("\nRunning agent...\n")
    try:
        async for message in run_agent(
            mcp_config=session.mcp_config,
            system_prompt=system_prompt + "\n\n" + session.system_prompt,
            task=task_text,
            provider="claude",
            verbose=True,
        ):
            # Print content
            if hasattr(message, 'content'):
                for block in message.content:
                    if hasattr(block, 'name') and hasattr(block, 'input'):
                        tool_name = block.name.replace('mcp__notebooks__', '')
                        print(f"\n🔧 Tool: {tool_name}")
                        if hasattr(block, 'id'):
                            print(f"   Tool ID: {block.id}")
                    elif hasattr(block, 'text'):
                        print(block.text, end="", flush=True)
                    elif hasattr(block, 'type') and block.type == 'extended_thinking':
                        # Extended thinking block - show full content
                        thinking_text = getattr(block, 'thinking', '') or getattr(block, 'text', '')
                        if thinking_text:
                            print(f"\n💭 [Thinking...]\n{thinking_text}\n", flush=True)
                    elif hasattr(block, 'tool_use_id'):
                        # Tool result - show FULL content (no truncation)
                        if hasattr(block, 'is_error') and block.is_error:
                            print(f"\n❌ Tool Error: {block.content}")
                        elif hasattr(block, 'content'):
                            # Show full content without truncation
                            print(f"\n✅ Tool Result: {block.content}")

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
