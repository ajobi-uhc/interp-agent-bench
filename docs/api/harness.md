# Harness API

## run_agent

```python
async for msg in run_agent(
    prompt: str,
    mcp_config: dict = {},
    provider: str = "claude",
    model: str = None,
    user_message: str = None,
):
    print(msg)
```

Run agent with task prompt. Streams messages.

**Parameters:**

- `prompt` — system prompt / task description
- `mcp_config` — from session (or empty dict)
- `provider` — "claude", "openai", or "gemini"
- `model` — specific model (optional)
- `user_message` — initial user message (optional)

**Example:**

```python
async for msg in run_agent(
    prompt="Explore this model's behavior",
    mcp_config=session.mcp_config,
    provider="claude"
):
    pass
```

## run_agent_interactive

```python
await run_agent_interactive(
    mcp_config: dict,
    provider: str = "claude",
    model: str = None,
)
```

Interactive chat session with agent. For debugging or manual exploration.
