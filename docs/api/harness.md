# Harness API

## run_agent

```python
from src.harness import run_agent

async for message in run_agent(
    prompt: str,
    mcp_config: dict,
    provider: str = "claude",
    model: str = None,
):
    print(message)
```

Run an agent with the given task prompt. Streams messages as the agent works.

**Parameters:**
- `prompt` - Task description for the agent
- `mcp_config` - MCP server configuration (from session)
- `provider` - Agent provider ("claude", "openai")
- `model` - Specific model to use (optional, uses default for provider)

**Returns:** Async iterator of message strings

**Example:**

```python
session = create_notebook_session(sandbox, workspace)

async for msg in run_agent(
    prompt="Explore the model architecture",
    mcp_config=session.mcp_config,
    provider="claude"
):
    print(msg)
```

## run_agent_interactive

```python
from src.harness import run_agent_interactive

await run_agent_interactive(
    mcp_config: dict,
    provider: str = "claude",
    model: str = None,
)
```

Start an interactive agent session where you can chat with the agent.

**Parameters:**
- `mcp_config` - MCP server configuration (from session)
- `provider` - Agent provider ("claude", "openai")
- `model` - Specific model to use (optional)

**Use case:** For debugging or manual exploration of the sandbox environment.
