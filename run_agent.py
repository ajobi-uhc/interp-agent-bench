#!/usr/bin/env python3
"""
Run a Claude agent with Jupyter notebook MCP server from YAML configuration.

Usage:
    python run_agent.py configs/gemma_secret_extraction.yaml
"""

import argparse
import asyncio
import sys
from pathlib import Path

import yaml
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions


def load_config(config_path: Path) -> dict:
    """Load experiment configuration from YAML file.

    Supports loading hidden system prompts from external files for security.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load hidden system prompt from external file if specified
    model_config = config.get('model', {})
    if 'hidden_system_prompt_file' in model_config:
        prompt_file = Path(model_config['hidden_system_prompt_file'])

        # Support relative paths from config directory
        if not prompt_file.is_absolute():
            prompt_file = config_path.parent / prompt_file

        if prompt_file.exists():
            config['model']['hidden_system_prompt'] = prompt_file.read_text().strip()
            print(f"🔒 Loaded hidden system prompt from: {prompt_file}")
            # Remove file reference to keep it hidden
            del config['model']['hidden_system_prompt_file']
        else:
            print(f"⚠️  Warning: Hidden prompt file not found: {prompt_file}")

    return config


async def run_notebook_agent(config_path: Path):
    """Run an agent with access to the notebook MCP server."""

    # Load configuration
    config = load_config(config_path)
    print(f"📋 Loading config: {config['experiment_name']}")
    print(f"   {config['description']}")

    # Extract model configuration
    model_config = config.get('model', {})
    execution_mode = model_config.get('execution_mode', 'modal')

    # Pre-deploy ModelService if model is configured and using Modal
    if 'model' in config and model_config.get('name'):
        if execution_mode == 'modal':
            from deploy_model import deploy
            print(f"🚀 Deploying model to Modal GPU...")
            deploy(model_config, config['experiment_name'], config.get('techniques'))
        elif execution_mode == 'local':
            print(f"🔧 Local execution mode - skipping Modal deployment")
        else:
            print(f"❌ Error: Invalid execution_mode '{execution_mode}'. Must be 'modal' or 'local'")
            sys.exit(1)

    # Get Python from virtual environment
    venv_python = Path.cwd() / ".venv" / "bin" / "python"
    if not venv_python.exists():
        print(f"❌ Error: Python not found at {venv_python}")
        print("Make sure you're in the project root with an active virtual environment")
        sys.exit(1)

    # Create agent workspace with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    agent_workspace = Path.cwd() / "notebooks" / f"{config['experiment_name']}_{timestamp}"
    agent_workspace.mkdir(parents=True, exist_ok=True)

    # Use model_config from earlier
    selected_techniques = config.get('techniques', [])

    # Configure the MCP server for notebooks with model info
    mcp_env = {
        "NOTEBOOK_OUTPUT_DIR": str(agent_workspace),
        "PATH": str(Path.cwd() / ".venv" / "bin"),
        "EXPERIMENT_NAME": config['experiment_name'],
        "MODEL_NAME": model_config['name'],
        "MODEL_IS_PEFT": "true" if model_config.get('is_peft', False) else "false",
        "MODEL_BASE": model_config.get('base_model', ''),
        "TOKENIZER_NAME": model_config.get('tokenizer', model_config.get('base_model', model_config['name'])),
        "GPU_TYPE": model_config.get('gpu_type', 'A10G'),
        "SELECTED_TECHNIQUES": ",".join(selected_techniques) if selected_techniques else "",
        "OBFUSCATE_MODEL_NAME": "true",  # Always obfuscate
        "EXECUTION_MODE": execution_mode,
        "DEVICE": model_config.get('device', 'auto'),
        "HIDDEN_SYSTEM_PROMPT": model_config.get('hidden_system_prompt', ''),
    }

    mcp_servers = {
        "notebooks": {
            "type": "stdio",
            "command": str(venv_python),
            "args": ["-m", "scribe.notebook.notebook_mcp_server"],
            "env": mcp_env
        }
    }

    # Build system prompt with technique source code
    agent_md_path = Path(__file__).parent / "AGENT.md"
    system_prompt = agent_md_path.read_text()

    # Add technique source code to prompt (if model is configured)
    if 'model' in config and config['model'].get('name'):
        from scribe.notebook.technique_loader import load_technique_methods
        techniques_dir = Path(__file__).parent / "techniques"
        all_techniques = load_technique_methods(techniques_dir)

        # Filter to selected techniques
        if selected_techniques:
            techniques = {name: method for name, method in all_techniques.items()
                         if name in selected_techniques}
        else:
            techniques = all_techniques

        # Append technique source code to prompt
        if techniques:
            system_prompt += "\n\n## Available Techniques\n\n"
            system_prompt += "The following techniques are pre-loaded on `model_service`. Here is their source code:\n\n"
            for name, method in techniques.items():
                system_prompt += f"### `{name}`\n\n"
                system_prompt += f"**Description**: {method.description}\n\n"
                system_prompt += "**Source code**:\n```python\n"
                system_prompt += method.code.strip() + "\n```\n\n"

    print(f"📝 System prompt built with technique source code")

    # Log system prompt to file
    prompt_log_path = agent_workspace / "system_prompt.md"
    with open(prompt_log_path, 'w') as f:
        f.write(system_prompt)
    print(f"💾 System prompt saved to: {prompt_log_path}")

    # Configure agent options
    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        mcp_servers=mcp_servers,
        permission_mode="bypassPermissions",
        add_dirs=[str(agent_workspace)],  # Allow access to workspace without changing cwd
        allowed_tools=[
            # Notebook session management
            "mcp__notebooks__start_new_session",
            "mcp__notebooks__start_session_resume_notebook",
            "mcp__notebooks__start_session_continue_notebook",

            # Notebook operations
            "mcp__notebooks__execute_code",
            "mcp__notebooks__edit_cell",
            "mcp__notebooks__add_markdown",
            "mcp__notebooks__shutdown_session",

            # Technique management
            "mcp__notebooks__init_session",
            "mcp__notebooks__list_techniques",
            "mcp__notebooks__describe_technique",
        ]
    )

    print("=" * 70)
    print("🚀 Starting Claude agent with notebook MCP server")
    print(f"📂 Agent workspace: {agent_workspace}")
    print(f"🎯 Techniques: {', '.join(selected_techniques) if selected_techniques else 'all'}")
    print("=" * 70)

    # Use task from config
    experiment_prompt = config['task']

    # Use ClaudeSDKClient for continuous conversation
    async with ClaudeSDKClient(options=options) as client:

        # Send initial query
        await client.query(experiment_prompt)

        # Process response
        async for message in client.receive_response():
            if hasattr(message, 'subtype') and message.subtype == "init":
                # Show MCP connection info
                if hasattr(message, 'data') and 'mcp_servers' in message.data:
                    print("\n📡 MCP Server Status:")
                    for server in message.data['mcp_servers']:
                        status_icon = "✅" if server.get('status') == 'connected' else "❌"
                        print(f"  {status_icon} {server.get('name')}: {server.get('status')}")

                        if server.get('tools'):
                            print(f"      Tools: {len(server['tools'])}")
                            for tool in server['tools']:
                                print(f"        - {tool.get('name')}: {tool.get('description', 'No description')[:60]}...")

                        if server.get('status') == 'failed':
                            print(f"      ⚠️  Error: {server.get('error', 'Unknown error')}")
                    print()

            # Print assistant messages
            if hasattr(message, 'content'):
                for block in message.content:
                    if hasattr(block, 'text'):
                        print(block.text, end="", flush=True)

        print("\n\n" + "=" * 70)
        print("✅ Agent completed")
        print("=" * 70)


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Run Claude agent with Jupyter notebook MCP server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_agent.py configs/gemma_secret_extraction.yaml
  python run_agent.py configs/example_gpt2_test.yaml
        """
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to YAML configuration file"
    )

    args = parser.parse_args()

    if not args.config.exists():
        print(f"❌ Error: Config file not found: {args.config}")
        sys.exit(1)

    try:
        asyncio.run(run_notebook_agent(args.config))
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
