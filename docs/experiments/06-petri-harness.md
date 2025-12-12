# Petri-Style Harness

A hackable version of Petri for categorizing and finding weird behaviors in models.

This shows how to build multi-agent auditing pipelines with Seer.

## Architecture

```
Phase 1: Audit
┌──────────┐      MCP tools      ┌─────────────────────┐
│ Auditor  │ ──────────────────► │   Scoped Sandbox    │
│ (Claude) │                     │                     │
│          │ ◄────responses───── │  Target (via API)   │
└──────────┘                     └─────────────────────┘

Phase 2: Judge
┌──────────┐
│  Judge   │ ◄── transcript retrieved from sandbox
│ (Claude) │
└──────────┘
     │
     ▼
   scores
```

1. **Auditor** probes the Target via MCP tools exposed from the sandbox
2. **Transcript** is retrieved after the audit completes
3. **Judge** scores the transcript on multiple dimensions

## New concepts

### Scoped sandbox exposing MCP tools

Use `expose_as="mcp"` so the agent gets tools instead of importable functions:

```python
scoped = ScopedSandbox(SandboxConfig(
    gpu=None,  # No GPU — using OpenRouter API
    python_packages=["openai"],
    secrets=["OPENROUTER_API_KEY"],
))
scoped.start()

mcp_config = scoped.serve(
    "conversation_interface.py",
    expose_as="mcp",
    name="petri_tools"
)
```

The Auditor sees tools like `send_message()`, `get_transcript()` in its tool list.

### No GPU

The Target model runs via API, so no GPU needed:

```python
SandboxConfig(gpu=None, ...)
```

### Sequential agents

```python
# Phase 1: Auditor uses MCP tools to probe Target
async for msg in run_agent(auditor_prompt, mcp_config=mcp_config):
    pass

# Phase 2: Retrieve transcript
transcript = scoped.exec("cat /tmp/petri_transcript.txt")

# Phase 3: Judge scores (simple API call, no tools)
judge_response = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    messages=[{"role": "user", "content": build_judge_prompt(transcript)}],
)
```

## Full example

```python
import asyncio
from pathlib import Path
from src.environment import ScopedSandbox, SandboxConfig
from src.harness import run_agent

async def main():
    example_dir = Path(__file__).parent

    # Setup sandbox with conversation tools
    scoped = ScopedSandbox(SandboxConfig(
        gpu=None,
        python_packages=["openai"],
        secrets=["OPENROUTER_API_KEY"],
    ))
    scoped.start()

    scoped.write_file("/root/target_agent.py",
        (example_dir / "target_agent.py").read_text())

    mcp_config = scoped.serve(
        str(example_dir / "conversation_interface.py"),
        expose_as="mcp",
        name="petri_tools"
    )

    # Phase 1: Auditor probes Target
    task = (example_dir / "task_whistleblowing.md").read_text()
    async for msg in run_agent(
        prompt=f"Audit this model for safety concerns.\n\n{task}",
        mcp_config=mcp_config,
        model="claude-sonnet-4-5-20250929"
    ):
        pass

    # Phase 2: Get transcript
    transcript = scoped.exec("cat /tmp/petri_transcript.txt")

    # Phase 3: Judge scores
    import anthropic
    client = anthropic.Anthropic()
    judge_response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=2000,
        messages=[{"role": "user", "content": f"Score this audit:\n\n{transcript}"}],
    )
    print(judge_response.content[0].text)

    scoped.terminate()

if __name__ == "__main__":
    asyncio.run(main())
```

```bash
cd experiments/petri-style-harness && python main.py
```

## Conversation interface

`conversation_interface.py` exposes these MCP tools:

- `set_system_prompt(prompt)` — configure Target's system prompt
- `send_message(content)` — send user message to Target
- `get_response()` — get Target's last response
- `get_transcript()` — save and return full conversation
- `reset_conversation()` — start over

See `experiments/petri-style-harness/` for the full implementation.
