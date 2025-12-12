# Execution API

## create_notebook_session

```python
session = create_notebook_session(sandbox, workspace, name="notebook")
```

Agent gets Jupyter notebook on GPU.

**Returns NotebookSession:**

- `mcp_config` — pass to run_agent
- `jupyter_url` — view notebook in browser
- `model_info_text` — model details for prompt
- `session_id` — unique identifier

## create_local_session

```python
session = create_local_session(workspace, workspace_dir, name="local")
```

Agent runs locally. Use with ScopedSandbox for GPU access via RPC.

**Returns LocalSession:**

- `mcp_config` — pass to run_agent (empty dict)
- `name` — session name
- `workspace_dir` — local workspace path

## create_cli_session

```python
session = create_cli_session(sandbox, workspace, name="cli")
```

Agent gets shell interface to sandbox.

**Returns CLISession:**

- `mcp_config` — pass to run_agent
- `session_id` — unique identifier
- `exec(code)` — execute Python in sandbox
- `exec_shell(cmd)` — execute shell command
