# Scoped Sandbox

Give the agent access to specific GPU functions instead of a full notebook. See [Scoped Sandbox concept](../concepts/scoped-sandbox.md) for more on interface files.

## When to use this

- **Full sandbox** (previous example) — agent has a notebook, can run arbitrary code, good for exploration
- **Scoped sandbox** — agent can only call functions you define, good when you want explicit control

## 1. Configure the scoped sandbox

```python
from src.environment import ScopedSandbox, SandboxConfig, ModelConfig

scoped = ScopedSandbox(SandboxConfig(
    gpu="A100",
    models=[ModelConfig(name="google/gemma-2-9b")],
    python_packages=["torch", "transformers", "accelerate"],
))

scoped.start()
```

No `execution_mode` — the agent doesn't run in the sandbox. Instead, you serve specific functions from it.

## 2. Define GPU functions

Create an interface file with functions that run on the GPU:

```python
# interface.py
from transformers import AutoModel, AutoTokenizer
import torch

model_path = get_model_path("google/gemma-2-9b")  # injected by RPC server
model = AutoModel.from_pretrained(model_path, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)

@expose
def get_model_info() -> dict:
    """Get basic model information."""
    return {
        "num_layers": model.config.num_hidden_layers,
        "hidden_size": model.config.hidden_size,
        "vocab_size": model.config.vocab_size,
    }

@expose
def get_embedding(text: str) -> dict:
    """Get text embedding from model."""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    embedding = outputs.hidden_states[-1].mean(dim=1).squeeze()
    return {"embedding": embedding.tolist()}
```

- `@expose` marks functions the agent can call — everything else is hidden
- Functions must return JSON-serializable types (use `.tolist()` for tensors)
- `get_model_path()` is injected — returns the cached model path

## 3. Serve the interface

```python
model_tools = scoped.serve(
    str(example_dir / "interface.py"),
    expose_as="library",
    name="model_tools"
)
```

Loads `interface.py` on the GPU and creates an RPC server.

`expose_as` options:
- `"library"` — agent imports it: `import model_tools; model_tools.get_embedding("hello")`
- `"mcp"` — agent sees them as MCP tools

## 4. Add local helpers (optional)

```python
# helpers.py
def format_result(data: dict) -> str:
    """Format a result dict as markdown."""
    lines = ["## Result"]
    for key, value in data.items():
        lines.append(f"- **{key}**: {value}")
    return "\n".join(lines)
```

Local files run on your machine, not the GPU. No `@expose` needed.

## 5. Create workspace and session

```python
from src.workspace import Workspace, Library
from src.execution import create_local_session

workspace = Workspace(libraries=[
    Library.from_file(example_dir / "helpers.py"),  # local
    model_tools,  # remote (from scoped.serve)
])

session = create_local_session(
    workspace=workspace,
    workspace_dir=str(example_dir / "workspace"),
    name="scoped-example"
)
```

`create_local_session` runs the agent locally. When it calls `model_tools.*`, the call goes to the GPU via RPC. See [Workspaces](../concepts/workspaces.md) for more on libraries.

## 6. Run the agent

```python
from src.harness import run_agent

task = """
You have access to:
- `model_tools` — GPU functions (get_model_info, get_embedding)
- `helpers` — local utilities (format_result)

Get the model info and an embedding for "hello world", then format the results.
"""

async for msg in run_agent(
    prompt=task,
    mcp_config={},
    provider="claude"
):
    pass

scoped.terminate()
```

## Full example

```python
import asyncio
from pathlib import Path
from src.environment import ScopedSandbox, SandboxConfig, ModelConfig
from src.workspace import Workspace, Library
from src.execution import create_local_session
from src.harness import run_agent

async def main():
    example_dir = Path(__file__).parent

    scoped = ScopedSandbox(SandboxConfig(
        gpu="A100",
        models=[ModelConfig(name="google/gemma-2-9b")],
        python_packages=["torch", "transformers", "accelerate"],
    ))
    scoped.start()

    model_tools = scoped.serve(
        str(example_dir / "interface.py"),
        expose_as="library",
        name="model_tools"
    )

    workspace = Workspace(libraries=[
        Library.from_file(example_dir / "helpers.py"),
        model_tools,
    ])

    session = create_local_session(
        workspace=workspace,
        workspace_dir=str(example_dir / "workspace"),
        name="scoped-example"
    )

    task = """
    You have access to:
    - `model_tools` — GPU functions (get_model_info, get_embedding)
    - `helpers` — local utilities (format_result)

    Get the model info and an embedding for "hello world", then format the results.
    """

    try:
        async for msg in run_agent(prompt=task, mcp_config={}, provider="claude"):
            pass
    finally:
        scoped.terminate()

if __name__ == "__main__":
    asyncio.run(main())
```

```bash
cd experiments/scoped-sandbox-intro && python main.py
```
