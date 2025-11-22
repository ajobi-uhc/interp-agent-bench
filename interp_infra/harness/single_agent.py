"""Single agent harness - one agent with full notebook access."""

import time
from pathlib import Path
from datetime import datetime
from typing import Optional

from .base import Harness
from . import toolkit
from .prompts import build_agent_prompts


class SingleAgentHarness(Harness):
    """
    Single agent orchestration pattern.

    Creates one agent with full notebook access and runs it to completion.
    This is the standard pattern for most interpretability experiments.
    """

    def __init__(self, deployment, config, verbose: bool = False):
        """
        Initialize single agent harness.

        Args:
            deployment: Deployment from Stage 1+2
            config: ExperimentConfig
            verbose: Enable verbose logging
        """
        super().__init__(deployment, config)
        self.verbose = verbose

        # Create workspace for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        workspace_name = f"{config.name}_{timestamp}"
        self.workspace = Path.cwd() / "notebooks" / workspace_name
        self.workspace.mkdir(parents=True, exist_ok=True)

        # Store workspace on deployment for toolkit access
        self.deployment.workspace = self.workspace

    async def run(self) -> dict:
        """
        Run single agent to completion.

        Returns:
            Dictionary with outputs, token usage, and statistics
        """
        print(f"📋 Experiment: {self.config.name}")

        # Build prompts using helper
        prompts = build_agent_prompts(
            task=self.config.task,
            agent_provider="claude",  # TODO: Make configurable
            session_id=self.deployment.session_id,
            skills=self.config.execution.skills,
        )

        # Save prompts to workspace
        prompts.save_to_workspace(self.workspace)

        print("==" * 35)
        print(f"🚀 Starting single agent")
        print(f"📂 Workspace: {self.workspace}")
        print(f"🔗 Sandbox: {self.deployment.sandbox_id}")
        print(f"🔗 Jupyter: {self.deployment.jupyter_url}")
        print("==" * 35)

        # Create agent using toolkit
        agent = toolkit.create_agent(
            self.deployment,
            system_prompt=prompts.system_prompt,
            user_prompt=prompts.user_prompt,
            provider="claude",  # TODO: Make configurable
            verbose=self.verbose,
        )

        # Run agent with streaming output
        start_time = time.time()
        total_tokens = 0
        message_count = 0
        last_activity = time.time()

        print("\n⏳ Waiting for response...", flush=True)

        async with agent:
            await agent.query(prompts.user_prompt)

            async for message in agent.receive_response():
                # Update last activity timestamp
                current_time = time.time()
                if current_time - last_activity > 30:  # Show heartbeat every 30s
                    elapsed_mins = (current_time - start_time) / 60
                    print(f"\n⏱️  Still running... ({elapsed_mins:.1f} min elapsed)", flush=True)
                last_activity = current_time

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

        elapsed = time.time() - start_time
        print(f"\n\n⏱️  Total time: {elapsed:.2f}s ({elapsed/60:.2f} min)")
        print(f"📈 Total tokens: {total_tokens:,}")

        # Return results
        return {
            'total_tokens': total_tokens,
            'message_count': message_count,
            'elapsed_seconds': elapsed,
            'workspace': str(self.workspace),
        }
