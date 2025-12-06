# Harness

The harness runs the agent.

## Basic Usage

```python
async for msg in run_agent(
    prompt=task,
    mcp_config=session.mcp_config,
    provider="claude"
):
    print(msg)  # Stream agent output
```

## What It Does

1. Connects agent to session via MCP
2. Sends initial prompt
3. Streams agent messages back
4. Handles tool calls (MCP)

## Providers

```python
provider="claude"    # Claude Sonnet (default)
provider="openai"    # GPT-4
```

## MCP Config

Session provides `mcp_config` with available tools:

**Notebook mode:** `execute_code`, `add_markdown`, `edit_cell`, etc.
**Local mode:** Standard Python execution
**Scoped mode:** Custom tools from `@expose` functions

## Multi-Agent

For multi-agent setups (like Petri harness), use custom orchestration:

```python
# Auditor (Claude)
auditor = run_agent(auditor_prompt, mcp_config=auditor_tools)

# Target (GPT-4)
target = run_agent(target_prompt, mcp_config=target_tools)

# Judge (Claude)
judge = run_agent(judge_prompt, mcp_config={})
```

See `experiments/petri-style-harness/` for implementation.
