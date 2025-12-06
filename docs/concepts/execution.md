# Execution

Execution defines the interface between agent and sandbox.

## Three Modes

### Notebook

Agent gets Jupyter notebook via MCP tools. Can write/execute cells.

```python
session = create_notebook_session(sandbox, workspace)
# Returns: session.mcp_config, session.jupyter_url
```

Use when: Agent needs iterative exploration, plotting, model experimentation.

### Local

Agent runs locally, calls GPU functions via RPC.

```python
session = create_local_session(workspace, workspace_dir)
```

Use when: Agent logic runs locally, only model inference needs GPU.

### Scoped

Hybrid. Sandbox serves specific functions, agent calls them remotely.

```python
scoped = ScopedSandbox(config).start()
model_tools = scoped.serve("interface.py", expose_as="library")
```

Use when: Maximum isolation, clear API boundary between agent and GPU.

## Workspace

Defines what libraries/tools the agent has access to.

```python
workspace = Workspace(libraries=[
    Library.from_file("helpers.py"),      # Local library
    Library.from_file("steering_hook.py") # Remote library (runs on GPU)
])
```

Libraries in workspace get injected as importable modules.

## Session Object

```python
session.mcp_config    # Pass to run_agent()
session.jupyter_url   # For monitoring (Notebook mode)
session.model_info    # Model details for agent prompt
```
