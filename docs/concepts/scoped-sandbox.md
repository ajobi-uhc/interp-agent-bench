# Scoped Sandbox

A `ScopedSandbox` lets you serve specific functions via RPC for an agent to use instead of giving the agent full access.

## When to use

- **Sandbox** — agent has full notebook access, good for exploration
- **ScopedSandbox** — agent can only call functions you expose, good for controlled experiments

## Writing interface files

An interface file defines what GPU functions the agent can call.

```python
# interface.py
from transformers import AutoModel, AutoTokenizer
import torch

model_path = get_model_path("google/gemma-2-9b")  # injected
model = AutoModel.from_pretrained(model_path, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)

@expose
def get_embedding(text: str) -> dict:
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    embedding = outputs.hidden_states[-1].mean(dim=1).squeeze()
    return {"embedding": embedding.tolist()}
```

Rules:
- `@expose` marks functions the agent can call
- Must return JSON-serializable types (use `.tolist()` for tensors)
- `get_model_path()` is injected — returns cached model path
- Load models at module level, not inside functions

## Serving the interface

```python
scoped = ScopedSandbox(SandboxConfig(
    gpu="A100",
    models=[ModelConfig(name="google/gemma-2-9b")],
)).start()

model_tools = scoped.serve(
    "interface.py",
    expose_as="library",  # or "mcp"
    name="model_tools"
)
```

`expose_as` options:
- `"library"` — agent imports it: `import model_tools`
- `"mcp"` — agent sees functions as MCP tools

## Using with local session

```python
workspace = Workspace(libraries=[model_tools])
session = create_local_session(workspace, workspace_dir)

async for msg in run_agent(prompt, mcp_config={}):
    pass
```

The agent runs locally. When it calls `model_tools.*`, the call goes to the GPU via RPC.
