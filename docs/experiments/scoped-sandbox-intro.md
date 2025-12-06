# Scoped Sandbox Intro

**Goal**: Run the agent locally while calling specific GPU functions via RPC to minimize costs

In the sandbox-intro example, the agent runs on the GPU in a Jupyter notebook. This is great for exploration, but you pay for GPU time even while the agent is thinking. `ScopedSandbox` solves this by:

- Agent runs on your local machine (cheap)
- GPU sandbox only runs specific functions you define (expensive only when called)
- Communication happens via RPC (Remote Procedure Call)

Use this pattern when you have well-defined model operations and want to minimize GPU costs.

## ScopedSandbox Configuration

Instead of `Sandbox`, use `ScopedSandbox`:

```python
from src.environment import ScopedSandbox, SandboxConfig, ModelConfig

scoped = ScopedSandbox(SandboxConfig(
    gpu="A100",
    models=[ModelConfig(name="google/gemma-2-9b")],
    python_packages=["torch", "transformers", "accelerate"],
))

scoped.start()
```

Key differences from regular `Sandbox`:

- No `execution_mode` parameter - the agent doesn't run in the sandbox
- You'll "serve" specific functions from the sandbox instead of giving the agent full notebook access

## GPU Interface File

Create a file that defines which functions run on the GPU:

experiments/scoped-sandbox-intro/interface.py

```python
from transformers import AutoModel, AutoTokenizer
import torch

# get_model_path is injected by the RPC server
model_path = get_model_path("google/gemma-2-9b")
model = AutoModel.from_pretrained(model_path, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)

@expose
def get_model_info() -> dict:
    """Get basic model information."""
    config = model.config
    return {
        "num_layers": config.num_hidden_layers,
        "hidden_size": config.hidden_size,
        "vocab_size": config.vocab_size,
        "device": str(model.device),
    }

@expose
def get_embedding(text: str) -> dict:
    """Get text embedding from model."""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    embedding = outputs.hidden_states[-1].mean(dim=1).squeeze()

    return {
        "text": text,
        "embedding": embedding.tolist(),
        "preview": embedding[:10].tolist(),
    }
```

The `@expose` decorator:

- Marks functions that can be called via RPC from your local machine
- Only `@expose` decorated functions are accessible - this gives you explicit control
- Functions must return JSON-serializable types (dict, list, str, int, etc.) - not PyTorch tensors
- That's why we use `.tolist()` to convert tensors to lists before returning

## Serving the Interface

"Serving" means hosting the interface file on the GPU sandbox and creating an API the agent can call:

```python
model_tools = scoped.serve(
    str(example_dir / "interface.py"),
    expose_as="library",
    name="model_tools"
)
```

What this does:

- Loads `interface.py` in the GPU sandbox
- Finds all `@expose` decorated functions
- Creates an RPC server that the agent can call
- Returns a `Library` object that you add to the workspace

The `expose_as` parameter determines how the agent accesses these functions:

### `expose_as="library"` (Python imports)

The agent imports it like a Python module:

```python
import model_tools
result = model_tools.get_embedding("hello world")
```

When the agent calls `model_tools.get_embedding()`, the library:

1. Serializes the arguments to JSON
2. Sends them via RPC to the GPU sandbox
3. Runs the function on the GPU
4. Sends the result back
5. Returns it to the agent

### `expose_as="mcp"` (MCP tools)

The agent gets MCP tools it can call:

```python
model_tools = scoped.serve(
    str(example_dir / "interface.py"),
    expose_as="mcp",
    name="model_tools"
)
```

The agent sees them as MCP tools in its tool list and calls them like:

```
Tool: get_embedding
Parameters: {"text": "hello world"}
```

**When to use each:**

- Use `expose_as="library"` when:
  - Agent needs to write Python code that imports and uses functions
  - You want the agent to treat it like a normal Python library
  - You're using `create_local_session` (agent executes Python directly)

- Use `expose_as="mcp"` when:
  - You want functions available as tools in the agent's tool list
  - Building multi-agent systems where one agent controls another via tools
  - You need the agent to see function signatures and descriptions as tools

## Local Helper Library

You can also have local libraries that run on your machine (no GPU):

experiments/scoped-sandbox-intro/helpers.py

```python
def format_result(data: dict) -> str:
    """Format a result dict as markdown."""
    lines = ["## Result", ""]
    for key, value in data.items():
        lines.append(f"- **{key}**: {value}")
    return "\n".join(lines)
```

This is just a regular Python file. No `@expose` decorator needed because it runs locally.

## Workspace with Mixed Libraries

Now create a workspace with both local and remote libraries:

```python
from src.workspace import Workspace, Library

workspace = Workspace(libraries=[
    Library.from_file(example_dir / "helpers.py"),  # Local library
    model_tools,  # Remote library (from scoped.serve())
])
```

The agent can import both:

- `helpers` runs instantly on your machine
- `model_tools` makes RPC calls to the GPU

## Local Session

Instead of `create_notebook_session`, use `create_local_session`:

```python
from src.execution import create_local_session

session = create_local_session(
    workspace=workspace,
    workspace_dir=str(example_dir / "workspace"),
    name="minimal-example"
)
```

Key differences from notebook session:

- Agent runs on your local machine, not in a Jupyter notebook on the GPU
- `workspace_dir` is where the agent can save files locally
- No `session.jupyter_url` - there's no notebook to view
- The agent executes Python directly instead of through Jupyter

## Running the Agent

```python
from src.harness import run_agent

task = """
You have two libraries available:

1. `helpers` - formatting utilities (runs locally)
2. `model_tools` - model analysis tools (runs on GPU via RPC)

Task:
- Import and call model_tools.get_model_info() to see model specs
- Import and call model_tools.get_embedding("hello world")
- Import and use helpers.format_result() to format the output
- Show me the formatted results
"""

async for message in run_agent(
    prompt=task,
    mcp_config={},  # Empty - local session doesn't use MCP
    provider="claude"
):
    pass

print(f"\n✓ Session: {session.name}")
```

The agent will import both libraries and use them. GPU only charges during `model_tools` function calls, not during agent reasoning.

## Full Example

experiments/scoped-sandbox-intro/main.py

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
        name="minimal-example"
    )

    task = """
    You have two libraries available:

    1. `helpers` - formatting utilities (runs locally)
    2. `model_tools` - model analysis tools (runs on GPU via RPC)

    Task:
    - Import and call model_tools.get_model_info() to see model specs
    - Import and call model_tools.get_embedding("hello world")
    - Import and use helpers.format_result() to format the output
    - Show me the formatted results
    """

    try:
        async for message in run_agent(
            prompt=task,
            mcp_config={},
            provider="claude"
        ):
            pass

        print(f"\n✓ Session: {session.name}")

    finally:
        scoped.terminate()

if __name__ == "__main__":
    asyncio.run(main())
```

## Running

```bash
cd experiments/scoped-sandbox-intro
python main.py
```

The agent will import both libraries and make RPC calls to the GPU only when calling `model_tools` functions.

## When to Use This Pattern

Use `ScopedSandbox` + RPC when:

- You have well-defined model operations
- You want to minimize GPU costs
- The workflow is predictable

Use `Sandbox` + Notebook when:

- Research is exploratory
- You're not sure what operations you'll need
- You want the agent to experiment freely
