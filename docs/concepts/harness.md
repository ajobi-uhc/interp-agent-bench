# Harness

The harness runs the agent and connects it to the session.

```python
async for msg in run_agent(
    prompt=task,
    mcp_config=session.mcp_config,
    provider="claude"  # or "openai"
):
    print(msg)
```

## Providers

```python
provider="claude"   # Claude (default)
provider="openai"   # GPT-4
```

## What it does

1. Connects agent to session via MCP
2. Sends prompt
3. Streams agent messages back
4. Handles tool calls

## Multi-agent

For multi-agent setups, run multiple agents with different configs:

```python
auditor = run_agent(auditor_prompt, mcp_config=auditor_tools)
target = run_agent(target_prompt, mcp_config=target_tools)
judge = run_agent(judge_prompt, mcp_config={})
```

See `experiments/petri-style-harness/` for an example.
