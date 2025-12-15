# Harness

The harness runs the agent and connects it to a session. Seer provides a default harness, but it's designed to be swapped out. The session provides an mcp config for any harness/agent to connect to

## Basic usage
```python
async for msg in run_agent(
    prompt=task,
    mcp_config=session.mcp_config,
):
    print(msg)
```

The harness:
1. Connects the agent to the session via MCP
2. Sends the prompt
3. Streams messages back
4. Handles tool calls automatically

## Providers
```python
provider="claude"   # Claude (default)
```

## Interactive mode

Chat with the agent in your terminal. Press ESC to interrupt mid-response.
```python
await run_agent_interactive(
    prompt=prompt,
    mcp_config=session.mcp_config,
    user_message="Start by exploring the model's hidden preferences.",
)
```

Full example:
```python
import asyncio
from pathlib import Path

from src.environment import Sandbox, SandboxConfig, ExecutionMode, ModelConfig
from src.workspace import Workspace, Library
from src.execution import create_notebook_session
from src.harness import run_agent_interactive


async def main():
    config = SandboxConfig(
        execution_mode=ExecutionMode.NOTEBOOK,
        gpu="A100",
        models=[ModelConfig(name="google/gemma-2-9b-it")],
    )
    sandbox = Sandbox(config).start()

    workspace = Workspace(
        libraries=[
            Library.from_file("libraries/steering_hook.py"),
            Library.from_file("libraries/extract_activations.py"),
        ]
    )

    session = create_notebook_session(sandbox, workspace)

    prompt = f"{session.model_info_text}\n\n{workspace.get_library_docs()}\n\nInvestigate the model."

    try:
        await run_agent_interactive(
            prompt=prompt,
            mcp_config=session.mcp_config,
            user_message="Start with Phase 1.",
        )
    finally:
        sandbox.terminate()


asyncio.run(main())
```

## Multi-agent

For multi-agent setups, run multiple agents with different (or the same!) configs:
```python
auditor = run_agent(auditor_prompt, mcp_config=auditor_tools)
investigator = run_agent(investigator_prompt, mcp_config=investigator_tools)
judge = run_agent(judge_prompt, mcp_config={})
```

See [Petri Harness](../experiments/06-petri-harness.md) for a working example.

## Custom harnesses

The harness is just scaffolding around the agent. You can:

- Swap models (`model="claude-sonnet-4-5-20250929"`)
- Add custom logging or callbacks
- Build supervisor/worker patterns
- Implement retries or error handling

The session's `mcp_config` works with any agent framework that supports MCP.