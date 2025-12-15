# Scoped Sandbox usage

In the [previous tutorial](01-sandbox-intro.md), the agent had a full Jupyter notebook - it could run any code. That's great for exploration, but sometimes you want more control.

**Scoped sandbox** lets you expose specific functions to the agent. Instead of arbitrary code execution, the agent can only call functions you define. This is useful when you want to give the agent access to model operations without letting it do anything else. See [Scoped Sandbox](../concepts/scoped-sandbox.md) for the full concept.

**When to use which:**
- **Full sandbox** - Open-ended exploration, agent writes its own code
- **Scoped sandbox** - Controlled access, agent can only call functions you define

## Step 1: Configure the scoped sandbox

```python
from src.environment import ScopedSandbox, SandboxConfig, ModelConfig

scoped = ScopedSandbox(SandboxConfig(
    gpu="A100",
    models=[ModelConfig(name="google/gemma-2-9b")],
    python_packages=["torch", "transformers", "accelerate"],
    secrets=["HF_TOKEN"],
))

scoped.start()
```

Note: no `execution_mode`. The scoped sandbox serves functions - how the agent runs (notebook, CLI, local) is decided later.

## Step 2: Define functions to expose

Create `interface.py` with functions you want the agent to call:

```python
# interface.py
from transformers import AutoModel, AutoTokenizer
import torch

model_path = get_model_path("google/gemma-2-9b")  # injected by Seer
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

Key points:
- `@expose` marks functions the agent can call. Everything else stays hidden.
- `get_model_path()` is injected by Seer - returns path to the cached model.
- Return values must be JSON-serializable (use `.tolist()` for tensors).

## Step 3: Serve the interface

```python
model_tools = scoped.serve(
    str(example_dir / "interface.py"),
    expose_as="library",
    name="model_tools"
)
```

This loads your interface on the sandbox and creates an RPC server. The returned `model_tools` is a Library the agent can import.

`expose_as` options:
- `"library"` - Agent imports it: `model_tools.get_embedding("hello")`
- `"mcp"` - Agent sees functions as MCP tools

## Step 4: Create workspace and session

A [Workspace](../concepts/workspaces.md) defines what code the agent can import.

```python
from src.workspace import Workspace, Library
from src.execution import create_local_session

workspace = Workspace(libraries=[
    model_tools,  # from scoped.serve()
])

session = create_local_session(
    workspace=workspace,
    workspace_dir=str(example_dir / "workspace"),
    name="scoped-example"
)
```

Here we use `create_local_session` - the agent runs locally and calls the exposed functions via RPC. You could also use `create_notebook_session` if you wanted the agent to work in a notebook while still having access to the scoped functions. See [Sessions](../concepts/sessions.md) for more on session types.

## Step 5: Run the agent

```python
from src.harness import run_agent

task = """
You have access to `model_tools` with these functions:
- get_model_info() - returns model architecture info
- get_embedding(text) - returns embedding for input text

Get the model info and an embedding for "hello world".
"""

async for msg in run_agent(prompt=task, mcp_config={}, provider="claude"):
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
        secrets=["HF_TOKEN"],
    ))
    scoped.start()

    model_tools = scoped.serve(
        str(example_dir / "interface.py"),
        expose_as="library",
        name="model_tools"
    )

    workspace = Workspace(libraries=[model_tools])

    session = create_local_session(
        workspace=workspace,
        workspace_dir=str(example_dir / "workspace"),
        name="scoped-example"
    )

    task = """
    You have access to `model_tools` with these functions:
    - get_model_info() - returns model architecture info
    - get_embedding(text) - returns embedding for input text

    Get the model info and an embedding for "hello world".
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
cd experiments/scoped-sandbox-intro
python main.py
```

## Next steps

- [Hidden Preference](03-hidden-preference.md) - Give the agent interpretability tools and see if it can discover a model's hidden bias
