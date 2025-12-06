# Workspaces

A workspace is a set of libraries available to the agent.

## Structure

```python
workspace = Workspace(libraries=[
    Library.from_file("local_helper.py"),
    Library.from_file("gpu_tools.py"),
])
```

## How It Works

Libraries become importable modules in the agent's environment.

**Example:**

`helpers.py`:
```python
def format_result(data: dict) -> str:
    return f"Result: {data}"
```

Agent can:
```python
from helpers import format_result
result = format_result({"key": "value"})
```

## Local vs Remote

**Local libraries:** Run in agent's process (notebook/local execution)
**Remote libraries:** Run on GPU (scoped execution with `@expose` decorator)

```python
# Remote library (GPU)
@expose
def get_embedding(text: str) -> dict:
    # Runs on GPU, returns to agent
    ...
```

## Shared Libraries

Common interpretability tools live in `experiments/shared_libraries/`:
- `extract_activations.py` - Layer activation extraction
- `steering_hook.py` - Activation steering via hooks

Import them:
```python
workspace = Workspace(libraries=[
    Library.from_file(shared_libs / "steering_hook.py"),
    Library.from_file(shared_libs / "extract_activations.py"),
])
```
