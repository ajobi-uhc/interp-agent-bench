# Workspace API

## Workspace

```python
from src.workspace import Workspace, Library

workspace = Workspace(
    libraries: list[Library] = [],
    local_dirs: list[tuple[str, str]] = [],
    skill_dirs: list[str] = [],
)
```

Defines files, libraries, and skills available to the agent.

**Fields:**
- `libraries` - Python libraries to inject into agent's environment
- `local_dirs` - Local directories to mount: `[(src_path, dest_path), ...]`
- `skill_dirs` - Directories containing agent skills/tools

## Library

```python
from src.workspace import Library

# From local file
lib = Library.from_file("path/to/helpers.py")

# From remote GPU (via ScopedSandbox.serve())
model_tools = scoped_sandbox.serve("interface.py", name="model_tools")
```

Represents a Python library available to the agent.

**Methods:**
- `Library.from_file(path: str)` - Load library from local file
- `Library.from_code(name: str, code: str)` - Create library from code string

**Note:** Libraries from `ScopedSandbox.serve()` are automatically wrapped as remote RPC libraries.
