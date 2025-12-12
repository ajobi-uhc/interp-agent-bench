# Environment API

## SandboxConfig

```python
SandboxConfig(
    gpu: str = None,                    # "A100", "H100", "A10G", or None for CPU
    execution_mode: ExecutionMode = ExecutionMode.CLI,
    models: list[ModelConfig] = [],
    repos: list[RepoConfig] = [],
    python_packages: list[str] = [],
    system_packages: list[str] = [],
    secrets: list[str] = [],            # Modal secret names
    timeout: int = 3600,                # Seconds (default 1 hour)
    local_files: list[tuple] = [],      # [(local_path, sandbox_path), ...]
    local_dirs: list[tuple] = [],       # [(local_path, sandbox_path), ...]
    env: dict[str, str] = {},           # Environment variables
)
```

## ModelConfig

```python
ModelConfig(
    name: str,                          # HuggingFace model ID
    var_name: str = "model",            # Variable name in model info
    hidden: bool = False,               # Hide model name from agent
    is_peft: bool = False,              # Is a PEFT adapter
    base_model: str = None,             # Base model ID if PEFT
)
```

## RepoConfig

```python
RepoConfig(
    url: str,                           # GitHub repo (e.g., "user/repo")
    dockerfile: str = None,             # Optional Dockerfile path
    install: bool = False,              # Run pip install
)
```

## ExecutionMode

```python
ExecutionMode.NOTEBOOK  # Jupyter notebook on GPU
ExecutionMode.CLI       # Shell interface
```

## Sandbox

```python
sandbox = Sandbox(config).start()
```

**Methods:**

- `start()` → Sandbox — provision GPU, download models, return running sandbox
- `terminate()` — shutdown sandbox
- `exec(cmd: str)` → str — execute shell command
- `exec_python(code: str)` → str — execute Python code

**Properties:**

- `model_handles` — list of ModelHandle for loaded models
- `repo_handles` — list of RepoHandle for cloned repos

## ScopedSandbox

```python
scoped = ScopedSandbox(config)
scoped.start()

lib = scoped.serve(
    "interface.py",
    expose_as="library",  # or "mcp"
    name="model_tools"
)
```

**Methods:**

- `start()` — provision sandbox
- `serve(file, expose_as, name)` → Library | dict — serve file as RPC library or MCP tools
- `write_file(path, content)` — write file to sandbox
- `exec(cmd)` → str — execute shell command
- `terminate()` — shutdown sandbox

**expose_as options:**

- `"library"` — returns Library, agent imports it
- `"mcp"` — returns MCP config dict, agent sees tools
