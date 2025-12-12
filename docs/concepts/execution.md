# Execution

Sessions define how the agent connects to the sandbox.

## Notebook mode

Agent gets a Jupyter notebook on the GPU.

```python
session = create_notebook_session(sandbox, workspace)
```

Returns:
- `session.mcp_config` — pass to `run_agent`
- `session.jupyter_url` — view notebook in browser
- `session.model_info_text` — model details for agent prompt

Use when: exploratory research, iterative probing, visualization.

## Local mode

Agent runs locally, calls GPU functions via RPC.

```python
session = create_local_session(workspace, workspace_dir, name)
```

Use when: you want explicit control over what GPU functions are available.

Requires `ScopedSandbox` with interface file. See [Scoped Sandbox](scoped-sandbox.md).
