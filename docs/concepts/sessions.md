# Sessions

Sessions define how the agent connects to the sandbox.

| Sandbox type | Session type | Agent experience |
|-------------|--------------|------------------|
| `Sandbox` | Notebook | Full Jupyter access on GPU |
| `ScopedSandbox` | Local | Runs locally, calls exposed functions via RPC |

## Notebook session

Agent gets a Jupyter notebook running on the sandbox.
```python
session = create_notebook_session(sandbox, workspace)
```

Returns:
- `session.mcp_config` — pass to `run_agent`  
- `session.jupyter_url` — view notebook in browser
- `session.model_info_text` — model details for agent prompt

Use when: exploratory research, iterative probing, visualization.

## Local session

Agent runs on your machine. GPU access is through the functions you exposed.
```python
session = create_local_session(workspace, workspace_dir, name)
```

Returns the same `mcp_config` interface, but execution happens locally.

Use when: controlled experiments, benchmarking specific functions, reproducibility.

Requires `ScopedSandbox` with interface file. See [Scoped Sandbox](scoped-sandbox.md).