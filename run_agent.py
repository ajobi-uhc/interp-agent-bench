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
from typing import Any

import yaml
from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    AssistantMessage,
    ToolUseBlock,
    ToolResultBlock,
    TextBlock,
    ThinkingBlock,
    SystemMessage,
    ResultMessage,
    HookContext,
)
from dotenv import load_dotenv
load_dotenv()


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


def format_tool_input(tool_input: dict[str, Any], verbose: bool = False) -> str:
    """Format tool input for display."""
    if not verbose:
        # Compact display for non-verbose mode
        if 'code' in tool_input:
            first_line = tool_input['code'].split('\n')[0][:60]
            return f"code: {first_line}..."
        elif 'command' in tool_input:
            return f"command: {tool_input['command'][:60]}..."
        else:
            # Show first few key-value pairs
            items = list(tool_input.items())[:3]
            return ", ".join(f"{k}: {str(v)[:40]}..." for k, v in items)
    else:
        # Full display in verbose mode
        import json
        return json.dumps(tool_input, indent=2)


def format_tool_result(content: Any, verbose: bool = False) -> str:
    """Format tool result for display."""
    if not verbose:
        # Compact display - truncate at 500 chars
        if isinstance(content, str):
            return content[:500] + ("..." if len(content) > 500 else "")
        elif isinstance(content, list):
            return str(content)[:500]
        else:
            return str(content)[:500]
    else:
        # Full display in verbose mode
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            import json
            return json.dumps(content, indent=2)
        else:
            return str(content)


def create_log_user_prompt_hook(workspace_path: Path):
    """Factory function to create a hook that logs input messages with workspace context."""
    from datetime import datetime
    
    async def log_user_prompt_hook(
        input_data: dict[str, Any],
        tool_use_id: str | None,
        context: HookContext
    ) -> dict[str, Any]:
        """Hook to log all input messages being sent to Claude."""
        prompt = input_data.get('prompt', '')
        timestamp = datetime.now().isoformat()
        
        # Log to console
        print("\n" + "="*70)
        print("📨 INPUT MESSAGE TO CLAUDE:")
        print("="*70)
        print(prompt)
        print("="*70 + "\n")
        
        # Log to a file in the workspace
        log_file = workspace_path / "input_messages.log"
        with open(log_file, 'a') as f:
            f.write(f"\n{'='*70}\n")
            f.write(f"TIMESTAMP: {timestamp}\n")
            f.write(f"{'='*70}\n")
            f.write(prompt)
            f.write(f"\n{'='*70}\n\n")
        
        # Return unmodified input_data (or modify if needed)
        return input_data
    
    return log_user_prompt_hook


async def run_notebook_agent(config_path: Path, run_id: int = None, verbose: bool = False):
    """Run an agent with access to the notebook MCP server.

    Args:
        config_path: Path to the configuration file
        run_id: Optional run identifier for parallel runs (e.g., 1, 2, 3)
        verbose: Enable verbose logging with detailed token usage and tool I/O
    """

    # Load configuration
    config = load_config(config_path)

    # Validate configuration
    from config_validator import validate_config, print_validation_errors
    errors = validate_config(config, config_path)
    if errors:
        print_validation_errors(errors, config_path)
        sys.exit(1)

    run_suffix = f" (Run {run_id})" if run_id is not None else ""
    print(f"📋 Loading config: {config['experiment_name']}{run_suffix}")
    print(f"   {config.get('description', '')}")

    # Extract model configuration
    model_config = config.get('model', {})
    execution_mode = model_config.get('execution_mode', 'modal')

    if execution_mode not in ['modal', 'local']:
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
    workspace_name = f"{config['experiment_name']}_{timestamp}"
    if run_id is not None:
        workspace_name += f"_run{run_id}"
    agent_workspace = Path.cwd() / "notebooks" / workspace_name
    agent_workspace.mkdir(parents=True, exist_ok=True)

    # Use model_config from earlier
    selected_techniques = config.get('techniques', [])

    # Configure the MCP server for notebooks with model info
    import os as os_module

    # Get API provider if not using GPU (determine before building env)
    api_provider = None
    if not ('model' in config and config['model'].get('name')):
        api_provider = model_config.get('api_provider', 'anthropic')  # Default to anthropic

    mcp_env = {
        "NOTEBOOK_OUTPUT_DIR": str(agent_workspace),
        "PATH": str(Path.cwd() / ".venv" / "bin"),
        "EXPERIMENT_NAME": config['experiment_name'],
        "MODEL_NAME": model_config.get('name', ''),
        "MODEL_IS_PEFT": "true" if model_config.get('is_peft', False) else "false",
        "MODEL_BASE": model_config.get('base_model', ''),
        "TOKENIZER_NAME": model_config.get('tokenizer', model_config.get('base_model', model_config.get('name', ''))),
        "GPU_TYPE": model_config.get('gpu_type', 'A10G'),
        "SELECTED_TECHNIQUES": ",".join(selected_techniques) if selected_techniques else "",
        "EXECUTION_MODE": execution_mode,
        "DEVICE": model_config.get('device', 'auto'),
        "HIDDEN_SYSTEM_PROMPT": model_config.get('hidden_system_prompt', ''),
        "API_PROVIDER": api_provider or '',
    }

    # Load specific API key if using API mode
    if api_provider:
        api_key_map = {
            'anthropic': 'ANTHROPIC_API_KEY',
            'openai': 'OPENAI_API_KEY',
            'google': 'GOOGLE_API_KEY',
        }
        key_name = api_key_map.get(api_provider)
        if key_name:
            mcp_env[key_name] = os_module.environ.get(key_name, '')
            if not mcp_env[key_name]:
                print(f"❌ Error: {key_name} not found in environment")
                print(f"\nTo use API mode with {api_provider}, set the API key:")
                print(f"  export {key_name}=\"your-key-here\"")
                print(f"  python run_agent.py {config_path}")
                sys.exit(1)

    mcp_servers = {
        "notebooks": {
            "type": "stdio",
            "command": str(venv_python),
            "args": ["-m", "scribe.notebook.notebook_mcp_server"],
            "env": mcp_env
        }
    }

    # Build prompts using clean prompt builder
    from prompt_builder import build_agent_prompts

    # Determine if GPU access is needed
    needs_gpu = 'model' in config and config['model'].get('name')

    # Check if research tips should be included
    include_research_tips = 'research_tips_file' in config

    # Build prompts (api_provider already determined above)
    prompts = build_agent_prompts(
        task=config['task'],
        needs_gpu=needs_gpu,
        selected_techniques=selected_techniques if needs_gpu else None,
        include_research_tips=include_research_tips,
        api_provider=api_provider,
    )

    system_prompt = prompts.system_prompt
    experiment_prompt = prompts.user_prompt

    # Callback to display MCP server logs
    def stderr_callback(line: str):
        """Display stderr output from MCP server."""
        if line.strip():
            print(f"[MCP] {line.rstrip()}", flush=True)

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
        ],
        include_partial_messages=True,  # Enable partial message streaming
        stderr=stderr_callback,  # Forward MCP server logs to terminal
        hooks={
            "UserPromptSubmit": [create_log_user_prompt_hook(agent_workspace)],  # Log all input messages (must be list)
        },
    )

    # Save prompts to workspace
    prompts.save_to_workspace(agent_workspace)

    print("=" * 70)
    print("🚀 Starting Claude agent with notebook MCP server")
    print(f"📂 Agent workspace: {agent_workspace}")
    print(f"🎯 Mode: {execution_mode}")
    print(f"🔬 Techniques: {', '.join(selected_techniques) if selected_techniques else 'agent will define as needed'}")
    print(f"📝 Input logging: enabled (saving to {agent_workspace / 'input_messages.log'})")
    if verbose:
        print(f"🔍 Verbose mode: ENABLED (full tool I/O, detailed token usage, thinking blocks)")
    print("=" * 70)

    # Use ClaudeSDKClient for continuous conversation
    async with ClaudeSDKClient(options=options) as client:

        # Send initial query
        import time
        query_start_time = time.time()
        await client.query(experiment_prompt)

        # Track cumulative token usage
        total_input_tokens = 0
        total_output_tokens = 0
        message_count = 0
        response_start_time = time.time()

        # Track agent completion status
        completion_status = None
        
        # Process response
        async for message in client.receive_response():
            # Check for completion/error
            if hasattr(message, 'subtype'):
                if message.subtype == "success":
                    completion_status = "success"
                    print("\n✅ Agent completed successfully")
                elif message.subtype == "error":
                    completion_status = "error"
                    error_msg = getattr(message, 'error', 'Unknown error')
                    print(f"\n❌ Agent encountered error: {error_msg}")
                elif message.subtype == "init":
                    # Show MCP connection info
                    pass  # Will be handled below
            
            if hasattr(message, 'subtype') and message.subtype == "init":
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

            # Log token usage when available
            if hasattr(message, 'usage') and message.usage:
                input_tokens = getattr(message.usage, 'input_tokens', 0)
                output_tokens = getattr(message.usage, 'output_tokens', 0)

                # Only log if there are actual tokens (skip empty usage reports)
                if input_tokens > 0 or output_tokens > 0:
                    # Try to get cache metrics (available in some API responses)
                    cache_creation_tokens = getattr(message.usage, 'cache_creation_input_tokens', 0)
                    cache_read_tokens = getattr(message.usage, 'cache_read_input_tokens', 0)

                    total_input_tokens += input_tokens
                    total_output_tokens += output_tokens
                    message_count += 1

                    # Calculate timing
                    message_time = time.time()
                    elapsed_since_query = message_time - query_start_time
                    elapsed_since_response_start = message_time - response_start_time

                    print(f"\n📊 Token Usage (Message #{message_count}):")
                    print(f"   Input:  {input_tokens:,} tokens (context size)")
                    print(f"   Output: {output_tokens:,} tokens")

                    # Only show cache info if present
                    if cache_creation_tokens > 0 or cache_read_tokens > 0:
                        if cache_creation_tokens > 0:
                            print(f"   Cache Write: {cache_creation_tokens:,} tokens")
                        if cache_read_tokens > 0:
                            print(f"   Cache Read: {cache_read_tokens:,} tokens")

                    print(f"   Cumulative - Input: {total_input_tokens:,} | Output: {total_output_tokens:,}")
                    print(f"   Response Time: {elapsed_since_response_start:.2f}s")

                    # Verbose mode: Show detailed metadata
                    if verbose:
                        if hasattr(message, 'id'):
                            print(f"   [VERBOSE] Message ID: {message.id}")
                        if hasattr(message, 'stop_reason'):
                            print(f"   [VERBOSE] Stop Reason: {message.stop_reason}")
                        if hasattr(message, 'model'):
                            print(f"   [VERBOSE] Model: {message.model}")

                        # Show the full usage object
                        if hasattr(message.usage, '__dict__'):
                            print(f"   [VERBOSE] Full usage object: {message.usage.__dict__}")

                    print()

                    # Reset response timer for next message
                    response_start_time = time.time()

            # Print assistant messages
            if hasattr(message, 'content'):
                for block in message.content:
                    # Show tool uses
                    if hasattr(block, 'name') and hasattr(block, 'input'):
                        # ToolUseBlock
                        tool_name = block.name.replace('mcp__notebooks__', '')
                        print(f"\n🔧 Tool: {tool_name}", flush=True)

                        if verbose:
                            # Verbose mode: Show full tool input
                            print(f"   [VERBOSE] Tool ID: {block.id}", flush=True)
                            print(f"   [VERBOSE] Full Input:", flush=True)
                            formatted_input = format_tool_input(block.input, verbose=True)
                            for line in formatted_input.split('\n'):
                                print(f"   {line}", flush=True)
                        else:
                            # Compact mode: Show key inputs
                            if hasattr(block, 'input') and isinstance(block.input, dict):
                                for key, value in block.input.items():
                                    if key in ['session_id', 'experiment_name', 'notebook_path']:
                                        print(f"   {key}: {value}", flush=True)
                                    elif key == 'code' and value:
                                        # Show first line of code
                                        first_line = value.split('\n')[0][:60]
                                        print(f"   code: {first_line}...", flush=True)

                    # Show tool results
                    elif hasattr(block, 'content') and hasattr(block, 'tool_use_id'):
                        # ToolResultBlock
                        if verbose:
                            # Verbose mode: Show full result with metadata
                            print(f"   ✅ Tool completed", flush=True)
                            print(f"   [VERBOSE] Tool Use ID: {block.tool_use_id}", flush=True)
                            if hasattr(block, 'is_error') and block.is_error:
                                print(f"   [VERBOSE] ⚠️  Error Result", flush=True)
                            print(f"   [VERBOSE] Full Response:", flush=True)
                            formatted_result = format_tool_result(block.content, verbose=True)
                            # Truncate extremely long results even in verbose mode
                            if len(formatted_result) > 5000:
                                print(f"   {formatted_result[:5000]}... (truncated, total: {len(formatted_result)} chars)", flush=True)
                            else:
                                for line in formatted_result.split('\n'):
                                    print(f"   {line}", flush=True)
                        else:
                            # Compact mode: Only show result if it's an error or short
                            if hasattr(block, 'is_error') and block.is_error:
                                print(f"   ❌ Error:", flush=True)
                                if isinstance(block.content, str):
                                    print(f"   {block.content[:500]}", flush=True)
                            # Skip normal tool results in compact mode - they're usually not interesting

                    # Show thinking blocks (extended thinking)
                    elif hasattr(block, 'thinking'):
                        # ThinkingBlock
                        if verbose:
                            print(f"\n💭 [VERBOSE] Extended Thinking:", flush=True)
                            print(f"   {block.thinking}", flush=True)
                            if hasattr(block, 'signature'):
                                print(f"   Signature: {block.signature}", flush=True)

                    # Show text
                    elif hasattr(block, 'text'):
                        print(block.text, end="", flush=True)

            # Show system messages in verbose mode (skip init messages as they're already shown)
            if verbose and isinstance(message, SystemMessage) and message.subtype not in ['init', 'success', 'error']:
                print(f"\n🔔 [VERBOSE] System Message: {message.subtype}", flush=True)
                if message.data:
                    import json
                    # Only show if data is interesting (not empty)
                    if message.data and len(str(message.data)) > 2:
                        print(f"   Data: {json.dumps(message.data, indent=2)}", flush=True)

        # Calculate final statistics
        total_elapsed_time = time.time() - query_start_time
        
        print("\n\n" + "=" * 70)
        if completion_status == "success":
            print("✅ Agent completed successfully")
        elif completion_status == "error":
            print("❌ Agent completed with errors")
        else:
            print("⚠️  Agent completed (status unknown)")
        print("=" * 70)
        print(f"\n📈 Final Statistics:")
        print(f"   Status:              {completion_status or 'unknown'}")
        print(f"   Total Messages:      {message_count}")
        print(f"   Total Input Tokens:  {total_input_tokens:,}")
        print(f"   Total Output Tokens: {total_output_tokens:,}")
        print(f"   Grand Total:         {total_input_tokens + total_output_tokens:,}")
        print(f"   Total Time:          {total_elapsed_time:.2f}s ({total_elapsed_time/60:.2f} min)")
        print("=" * 70)


async def run_parallel_agents(config_path: Path, num_runs: int, verbose: bool = False):
    """Run multiple agents in parallel on the same task.

    Args:
        config_path: Path to the configuration file
        num_runs: Number of parallel agents to run
        verbose: Enable verbose logging with detailed token usage and tool I/O
    """
    print("=" * 70)
    print(f"🚀 Running {num_runs} agents in PARALLEL")
    print("=" * 70)

    # Create tasks for all runs
    tasks = [
        run_notebook_agent(config_path, run_id=i+1, verbose=verbose)
        for i in range(num_runs)
    ]

    # Run all agents concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Check for failures
    failures = [i+1 for i, r in enumerate(results) if isinstance(r, Exception)]

    print("\n" + "=" * 70)
    if failures:
        print(f"⚠️  {len(failures)} run(s) failed: {failures}")
        print(f"✅ {num_runs - len(failures)} run(s) succeeded")
    else:
        print(f"✅ All {num_runs} runs completed successfully")
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
  python run_agent.py configs/gemma_secret_extraction.yaml --verbose

Config options:
  num_parallel_runs: 5  # Run 5 agents in parallel on the same task
        """
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging (full tool inputs/outputs, detailed token usage)"
    )

    args = parser.parse_args()

    if not args.config.exists():
        print(f"❌ Error: Config file not found: {args.config}")
        sys.exit(1)

    try:
        # Load config to check for parallel runs
        config = load_config(args.config)
        num_parallel_runs = config.get('num_parallel_runs', 1)

        if num_parallel_runs > 1:
            asyncio.run(run_parallel_agents(args.config, num_parallel_runs, verbose=args.verbose))
        else:
            asyncio.run(run_notebook_agent(args.config, verbose=args.verbose))
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
