# Environment API

## Sandbox

```python
from src.environment import Sandbox, SandboxConfig, ExecutionMode, ModelConfig

config = SandboxConfig(
    gpu="A100",
    execution_mode=ExecutionMode.NOTEBOOK,
    models=[ModelConfig(name="google/gemma-2-2b-it")],
    python_packages=["torch", "transformers"],
)
sandbox = Sandbox(config).start()
```

**Methods:**
- `start()` - Provision GPU, download models, return running sandbox
- `terminate()` - Shutdown sandbox
- `exec(cmd: str)` - Execute shell command in sandbox
- `exec_python(code: str)` - Execute Python code in sandbox

## ScopedSandbox

```python
from src.environment import ScopedSandbox, SandboxConfig, ModelConfig

scoped = ScopedSandbox(SandboxConfig(
    gpu="A100",
    models=[ModelConfig(name="google/gemma-2-9b")],
    python_packages=["torch", "transformers"],
))
scoped.start()

# Serve a Python file as an RPC library
model_tools = scoped.serve(
    "interface.py",
    expose_as="library",
    name="model_tools"
)
```

**Methods:**
- `start()` - Provision GPU sandbox
- `serve(file_path, expose_as="library", name)` - Serve Python file as RPC library
- `terminate()` - Shutdown sandbox

## SandboxConfig

```python
SandboxConfig(
    gpu: str = "A100",
    execution_mode: ExecutionMode = ExecutionMode.NOTEBOOK,
    models: list[ModelConfig] = [],
    repos: list[RepoConfig] = [],
    python_packages: list[str] = [],
    provider: str = "modal",
)
```

**Fields:**
- `gpu` - GPU type ("A100", "H100", "A10G")
- `execution_mode` - How agent interacts (NOTEBOOK, CLI, LOCAL)
- `models` - List of models to download
- `repos` - List of git repos to clone
- `python_packages` - Python packages to install
- `provider` - Infrastructure provider ("modal", "runpod")

## ModelConfig

```python
ModelConfig(
    name: str,
    revision: str = "main",
)
```

**Fields:**
- `name` - HuggingFace model ID (e.g., "google/gemma-2-2b-it")
- `revision` - Git revision/branch (default: "main")

## ExecutionMode

```python
from src.environment import ExecutionMode

ExecutionMode.NOTEBOOK  # Jupyter notebook with GPU access
ExecutionMode.CLI       # Shell interface to GPU sandbox
ExecutionMode.LOCAL     # Agent runs locally, calls GPU via RPC
```
