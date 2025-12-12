# Core Concepts

```
┌──────────────────────────────────────────────────────────────┐
│                         Your Machine                         │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                        Harness                          │ │
│  │  run_agent(prompt, mcp_config, provider="claude")       │ │
│  └───────────────────────────┬─────────────────────────────┘ │
│                              │ MCP                           │
│                              ▼                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                        Session                          │ │
│  │  Notebook: agent works in Jupyter                       │ │
│  │  Local: agent runs locally, calls GPU via RPC           │ │
│  └───────────────────────────┬─────────────────────────────┘ │
└──────────────────────────────┼───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                      Modal (Remote GPU)                      │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                       Sandbox                           │ │
│  │  - GPU (A100, H100, etc.)                               │ │
│  │  - Models (cached on Modal volumes)                     │ │
│  │  - Workspace libraries                                  │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

## Sandbox

GPU environment with models loaded. [Details →](environment.md)

```python
sandbox = Sandbox(SandboxConfig(
    gpu="A100",
    models=[ModelConfig(name="google/gemma-2-9b")],
)).start()
```

Two types:
- **Sandbox** — agent has full access (notebook mode)
- **ScopedSandbox** — agent can only call functions you expose (local mode)

## Workspace

Libraries the agent can import. [Details →](workspaces.md)

```python
workspace = Workspace(libraries=[
    Library.from_file("steering_hook.py"),
])
```

## Session

How the agent connects to the sandbox. [Details →](execution.md)

```python
session = create_notebook_session(sandbox, workspace)  # Full access
# or
session = create_local_session(workspace, workspace_dir)  # Scoped access
```

## Harness

Runs the agent. [Details →](harness.md)

```python
async for msg in run_agent(prompt, mcp_config=session.mcp_config):
    pass
```

## Putting it together

```python
# 1. Sandbox
config = SandboxConfig(gpu="A100", models=[...])
sandbox = Sandbox(config).start()

# 2. Workspace
workspace = Workspace(libraries=[...])

# 3. Session
session = create_notebook_session(sandbox, workspace)

# 4. Harness
async for msg in run_agent(prompt, mcp_config=session.mcp_config):
    pass

# 5. Cleanup
sandbox.terminate()
```
