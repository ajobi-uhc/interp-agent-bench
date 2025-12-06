# Core Concepts

Three components. They chain.

## Sandbox

Provisions GPU compute. Downloads models. Installs packages. Returns a running environment.

```python
config = SandboxConfig(gpu="A100", models=[...], python_packages=[...])
sandbox = Sandbox(config).start()
```

## Session

Defines how agents interact with the sandbox. Three modes: Notebook (Jupyter), Local (direct), Scoped (RPC).

```python
session = create_notebook_session(sandbox, workspace)
# Returns: mcp_config, jupyter_url, model_info_text
```

## Harness

Runs the agent. Connects to session via MCP. Streams results.

```python
async for msg in run_agent(prompt=task, mcp_config=session.mcp_config):
    pass
```

## Flow

```
SandboxConfig → Sandbox.start() → create_session() → run_agent()
```

That's it.
