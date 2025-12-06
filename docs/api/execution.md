# Execution API

Session functions create execution environments for agents.

## create_notebook_session

```python
from src.execution import create_notebook_session

session = create_notebook_session(
    sandbox: Sandbox,
    workspace: Workspace,
    name: str = "notebook",
)
```

Agent gets Jupyter notebook with full GPU access. Best for exploratory research.

**Returns:** `NotebookSession` with:
- `mcp_config` - MCP config for agent connection
- `jupyter_url` - Jupyter notebook URL
- `model_info_text` - Text describing loaded models
- `session_id` - Unique session identifier

## create_cli_session

```python
from src.execution import create_cli_session

session = create_cli_session(
    sandbox: Sandbox,
    workspace: Workspace,
    name: str = "cli",
)
```

Agent gets shell interface to GPU sandbox. For command-line workflows.

**Returns:** `CLISession` with:
- `mcp_config` - MCP config for agent connection
- `session_id` - Unique session identifier
- `exec(code)` - Execute Python code in sandbox
- `exec_shell(cmd)` - Execute shell command in sandbox

## create_local_session

```python
from src.execution import create_local_session

session = create_local_session(
    workspace: Workspace,
    workspace_dir: str,
    name: str = "local",
)
```

Agent runs locally but can call GPU functions via RPC. Use with `ScopedSandbox.serve()`.

**Returns:** `LocalSession` with:
- `mcp_config` - MCP config for agent connection (empty dict, uses default tools)
- `name` - Session name
- `workspace_dir` - Path to local workspace
