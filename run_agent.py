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
    from interp_infra.deploy import deploy_experiment
    from interp_infra.config.parser import load_config
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
        # Cleanup deployment
        deployment.close()
        print("\n🔌 Modal sandbox terminated")


def main():
    parser = argparse.ArgumentParser(description="Run agent with GPU deployment")
    parser.add_argument("config", type=Path, help="Path to experiment YAML config")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if not args.config.exists():
        print(f"❌ Config not found: {args.config}")
        sys.exit(1)

    try:
        asyncio.run(run_gpu_agent(args.config, args.verbose))
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
