#!/usr/bin/env python3
"""
Run a Claude agent with GPU deployment via interp-infra.

Usage:
    python run_agent_new.py configs/my_experiment.yaml
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


async def run_gpu_agent(config_path: Path, verbose: bool = False):
    """Run agent with GPU deployment."""
    global deployment_ref

    from interp_infra.deploy import deploy_experiment
    from interp_infra.config.parser import load_config
    from interp_infra.sandbox_state import save_sandbox_state, SandboxState
    from prompt_builder import build_agent_prompts
    from agent_providers import create_agent_provider, AgentOptions
    from datetime import datetime

    # Load config
    config = load_config(config_path)
    print(f"📋 Experiment: {config.name}")

    # Create agent workspace
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    workspace_name = f"{config.name}_{timestamp}"
    agent_workspace = Path.cwd() / "notebooks" / workspace_name
    agent_workspace.mkdir(parents=True, exist_ok=True)

    # Deploy GPU infrastructure on Modal
    print("\n🚀 Deploying GPU infrastructure on Modal...")
    deployment = deploy_experiment(config_path=config_path)
    deployment_ref = deployment  # Store globally for interrupt handler

    # Save deployment state immediately
    state = SandboxState(
        sandbox_id=deployment.sandbox_id,
        jupyter_url=deployment.jupyter_url,
        jupyter_port=deployment.jupyter_port,
        jupyter_token=deployment.jupyter_token,
        experiment_name=config.name,
        config_path=str(config_path.absolute()),
        workspace_path=str(agent_workspace.absolute()),
        timestamp=timestamp,
    )
    save_sandbox_state(state)

    # Configure MCP server to connect to Modal Jupyter via public tunnel
    venv_python = Path.cwd() / ".venv" / "bin" / "python"
    mcp_servers = {
        "notebooks": {
            "type": "stdio",
            "command": str(venv_python),
            "args": ["-m", "scribe.notebook.notebook_mcp_server"],
            "env": {
                "SCRIBE_URL": deployment.jupyter_url,  # Public HTTPS tunnel (no auth needed)
                "NOTEBOOK_OUTPUT_DIR": str(agent_workspace),
            }
        }
    }

    # Build prompts from config
    prompts = build_agent_prompts(
        task=config.task,
        needs_gpu=True,
        agent_provider="claude",
        session_id=deployment.session_id,  # Pass pre-warmed session ID
    )

    # Callback to display MCP server logs
    def stderr_callback(line: str):
        if line.strip():
            print(f"[MCP] {line.rstrip()}", flush=True)

    # Configure agent options
    options = AgentOptions(
        system_prompt=prompts.system_prompt,
        workspace_path=agent_workspace,
        mcp_config=mcp_servers,
        allowed_tools=[
            "mcp__notebooks__attach_to_session",  # NEW: Attach to pre-warmed session
            "mcp__notebooks__start_new_session",
            "mcp__notebooks__execute_code",
            "mcp__notebooks__add_markdown",
            "mcp__notebooks__edit_cell",
            "mcp__notebooks__shutdown_session",
        ],
        stderr_callback=stderr_callback,
        verbose=verbose,
    )

    # Save prompts
    prompts.save_to_workspace(agent_workspace)

    print("==" * 35)
    print(f"🚀 Starting agent with Modal GPU deployment")
    print(f"📂 Workspace: {agent_workspace}")
    print(f"🔗 Sandbox: {deployment.sandbox_id}")
    print(f"🔗 Jupyter: {deployment.jupyter_url}")
    print("==" * 35)

    # Run agent
    try:
        async with create_agent_provider("claude", options) as client:
            import time
            start_time = time.time()

            await client.query(prompts.user_prompt)

            total_tokens = 0
            message_count = 0

            async for message in client.receive_response():
                # Track tokens
                if hasattr(message, 'usage') and message.usage:
                    input_tokens = getattr(message.usage, 'input_tokens', 0)
                    output_tokens = getattr(message.usage, 'output_tokens', 0)
                    if input_tokens > 0 or output_tokens > 0:
                        total_tokens += input_tokens + output_tokens
                        message_count += 1
                        print(f"\n📊 Token Usage (Message #{message_count}):")
                        print(f"   Input: {input_tokens:,} | Output: {output_tokens:,}")
                        print(f"   Total: {total_tokens:,}")

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
                        elif hasattr(block, 'tool_use_id'):
                            # Tool result - show if there's an error
                            if hasattr(block, 'is_error') and block.is_error:
                                print(f"\n❌ Tool Error: {block.content}")
                            elif hasattr(block, 'content'):
                                print(f"\n✅ Tool Result: {block.content[:200]}..." if len(str(block.content)) > 200 else f"\n✅ Tool Result: {block.content}")

            elapsed = time.time() - start_time
            print(f"\n\n⏱️  Total time: {elapsed:.2f}s ({elapsed/60:.2f} min)")
            print(f"📈 Total tokens: {total_tokens:,}")

    finally:
        # Deployment cleanup is handled by interrupt handler or normal exit
        pass


deployment_ref = None  # Global reference for interrupt handler


def main():
    parser = argparse.ArgumentParser(description="Run agent with GPU deployment")
    parser.add_argument("config", nargs="?", type=Path, help="Path to experiment YAML config or experiment name to resume")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    parser.add_argument("--list", "-l", action="store_true", help="List paused sandboxes")
    parser.add_argument("--resume", "-r", metavar="NAME", help="Resume agent from paused sandbox")
    parser.add_argument("--connect", "-c", metavar="NAME", help="Get Jupyter URL for human access")

    args = parser.parse_args()

    # Handle --list flag
    if args.list:
        list_paused_sandboxes()
        sys.exit(0)

    # Handle --connect flag
    if args.connect:
        show_connection_info(args.connect)
        sys.exit(0)

    # Handle --resume flag
    if args.resume:
        print("⚠️  Resume functionality not yet implemented")
        print(f"   Use: python run_agent.py --connect {args.resume}")
        sys.exit(1)

    # Normal agent run
    if not args.config:
        parser.print_help()
        sys.exit(1)

    if not args.config.exists():
        print(f"❌ Config not found: {args.config}")
        sys.exit(1)

    try:
        asyncio.run(run_gpu_agent(args.config, args.verbose))
        # Normal completion - terminate sandbox
        if deployment_ref:
            deployment_ref.close()
            print("\n✅ Agent completed - sandbox terminated")
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        choice = handle_interrupt()

        if choice == "1":  # Pause
            from interp_infra.config.parser import load_config
            config = load_config(args.config)
            jupyter_url = deployment_ref.jupyter_url if deployment_ref else 'N/A'
            notebooks_url = f"{jupyter_url}/tree/notebooks" if jupyter_url != 'N/A' else 'N/A'
            print(f"\n💤 Sandbox paused")
            print(f"   📂 Notebooks: {notebooks_url}")
            print(f"   💡 Open in browser to view/edit notebooks")
            print(f"   💡 Resume: python run_agent.py --resume {config.name}")
        elif choice == "2":  # Terminate
            if deployment_ref:
                deployment_ref.close()
                print("\n🔌 Sandbox terminated")
        elif choice == "3":  # Continue
            print("\n▶️  Continuing agent execution...")
            # TODO: Resume agent execution
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        # Cleanup on error
        if deployment_ref:
            deployment_ref.close()
        sys.exit(1)


def list_paused_sandboxes():
    """List all paused sandboxes."""
    from interp_infra.sandbox_state import list_saved_sandboxes

    sandboxes = list_saved_sandboxes()
    if not sandboxes:
        print("📭 No paused sandboxes found")
        return

    print("\n📦 Paused Sandboxes:")
    print("="*70)
    for state in sandboxes:
        print(f"\n  🔹 {state.experiment_name}")
        print(f"     Jupyter: {state.jupyter_url}")
        print(f"     Created: {state.timestamp}")
    print("="*70)


def show_connection_info(experiment_name: str):
    """Show connection info for human access."""
    from interp_infra.sandbox_state import load_sandbox_state

    state = load_sandbox_state(experiment_name)
    if not state:
        print(f"❌ No paused sandbox: {experiment_name}")
        print("\nRun: python run_agent.py --list")
        sys.exit(1)

    notebooks_url = f"{state.jupyter_url}/tree/notebooks"

    print("\n" + "="*70)
    print(f"🔗 Sandbox: {experiment_name}")
    print("="*70)
    print(f"\n📂 Notebooks: {notebooks_url}")
    print(f"📁 Local workspace: {state.workspace_path}")
    print("\n💡 Variables available in notebooks:")
    print("   - model: The loaded language model")
    print("   - tokenizer: The tokenizer")
    print("="*70)


def handle_interrupt():
    """Handle keyboard interrupt with options to pause or terminate."""
    import select

    print("\n" + "="*70)
    print("⚠️  Agent interrupted! What would you like to do?")
    print()
    print("  1. Pause   - Keep sandbox running, save state for later")
    print("  2. Terminate - Shutdown sandbox and cleanup")
    print("  3. Continue  - Resume agent execution")
    print()
    print("⏱️  Auto-terminating in 10 seconds if no input...")
    print("="*70)
    print("\nChoice [1/2/3]: ", end="", flush=True)

    # Timeout for auto-terminate
    timeout = 10
    ready, _, _ = select.select([sys.stdin], [], [], timeout)

    if ready:
        choice = sys.stdin.readline().strip()
    else:
        print("\n\n⏱️  Timeout - auto-terminating sandbox")
        choice = "2"

    return choice


if __name__ == "__main__":
    main()
