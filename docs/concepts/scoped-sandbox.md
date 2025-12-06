# Scoped Sandbox & RPC Interface Files

## What is a Scoped Sandbox?

A `ScopedSandbox` is a GPU environment that serves specific Python functions via RPC. Unlike a regular sandbox where the agent has full access, a scoped sandbox only exposes the functions you explicitly mark with `@expose`.

**Use when:** You want tight control over what GPU operations are available, or when you want the agent to run locally but call GPU functions remotely.

## Architecture

```
┌─────────────┐         RPC          ┌──────────────┐
│ Local Agent │ ◄──────────────────► │ GPU Sandbox  │
│             │                      │              │
│ workspace/  │                      │ interface.py │
└─────────────┘                      └──────────────┘
```

The agent runs locally. GPU functions run on the sandbox. Communication happens via HTTP RPC calls.

## Writing Interface Files

An interface file is a Python script that runs on the GPU and defines functions the agent can call remotely.

### Basic Structure

```python
# interface.py
from transformers import AutoModel, AutoTokenizer
import torch

# get_model_path() is injected by the RPC server
model_path = get_model_path("google/gemma-2-9b")
model = AutoModel.from_pretrained(model_path, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)

@expose
def get_embedding(text: str) -> dict:
    """Get embedding for text."""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    embedding = outputs.hidden_states[-1].mean(dim=1).squeeze()
    return {"embedding": embedding.tolist()}
```

### The `@expose` Decorator

Functions marked with `@expose` become callable via RPC:

```python
@expose
def my_function(arg1: str, arg2: int) -> dict:
    # This function can be called by the agent
    return {"result": "value"}
```

**Requirements:**
- Must have type hints for all parameters and return type
- Must return JSON-serializable types (dict, list, str, int, float, bool, None)
- Cannot return complex objects (PyTorch tensors, models, etc.)

### Injected Globals

The RPC server injects these globals into your interface file:

#### `get_model_path(model_id: str) -> str`

Returns the cached path to a model on the Modal volume.

```python
model_path = get_model_path("google/gemma-2-9b")
# Returns: "/models/google--gemma-2-9b"
```

#### `@expose`

Decorator to mark functions as RPC-callable.

```python
@expose
def my_function():
    pass
```

## Complete Example

```python
# experiments/my-experiment/interface.py
from transformers import AutoModel, AutoTokenizer
import torch

# Load model (happens once when RPC server starts)
model_path = get_model_path("google/gemma-2-9b")
model = AutoModel.from_pretrained(model_path, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)

@expose
def get_model_info() -> dict:
    """Return basic model information."""
    return {
        "num_layers": model.config.num_hidden_layers,
        "hidden_size": model.config.hidden_size,
        "vocab_size": model.config.vocab_size,
    }

@expose
def generate_text(prompt: str, max_tokens: int = 50) -> dict:
    """Generate text from prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"prompt": prompt, "generated": text}

@expose
def compare_texts(text1: str, text2: str) -> dict:
    """Compare similarity of two texts."""
    # Get embeddings
    inputs1 = tokenizer(text1, return_tensors="pt").to(model.device)
    inputs2 = tokenizer(text2, return_tensors="pt").to(model.device)

    with torch.no_grad():
        emb1 = model(**inputs1, output_hidden_states=True).hidden_states[-1].mean(dim=1)
        emb2 = model(**inputs2, output_hidden_states=True).hidden_states[-1].mean(dim=1)

    # Compute cosine similarity
    similarity = torch.nn.functional.cosine_similarity(emb1, emb2).item()

    return {
        "text1": text1,
        "text2": text2,
        "similarity": similarity
    }
```

## Using in Experiments

```python
# experiments/my-experiment/main.py
from src.environment import ScopedSandbox, SandboxConfig, ModelConfig
from src.workspace import Workspace, Library
from src.execution import create_local_session
from src.harness import run_agent

# Create scoped sandbox
scoped = ScopedSandbox(SandboxConfig(
    gpu="A100",
    models=[ModelConfig(name="google/gemma-2-9b")],
    python_packages=["torch", "transformers", "accelerate"],
))
scoped.start()

# Serve interface.py as RPC library
model_tools = scoped.serve(
    str(Path(__file__).parent / "interface.py"),
    expose_as="library",
    name="model_tools"
)

# Create workspace with RPC library
workspace = Workspace(libraries=[model_tools])

# Create local session (agent runs locally)
session = create_local_session(
    workspace=workspace,
    workspace_dir="./workspace",
    name="my-experiment"
)

# Agent can now call model_tools functions
task = """
Import model_tools and:
1. Call get_model_info() to see model specs
2. Call generate_text("Once upon a time") to generate text
3. Call compare_texts("hello", "hi") to compare similarity
"""

async for msg in run_agent(prompt=task, mcp_config=session.mcp_config):
    pass
```

## Best Practices

### Return JSON-Serializable Data

Don't return tensors or complex objects:

```python
# BAD
@expose
def get_embedding(text: str) -> torch.Tensor:
    return embedding  # Tensor can't serialize

# GOOD
@expose
def get_embedding(text: str) -> dict:
    return {"embedding": embedding.tolist()}  # List is JSON-serializable
```

### Handle Errors Gracefully

```python
@expose
def generate_text(prompt: str) -> dict:
    if not prompt:
        return {"error": "prompt cannot be empty"}

    try:
        # ... generation code ...
        return {"text": result}
    except Exception as e:
        return {"error": str(e)}
```

### Keep Functions Focused

Each function should do one thing:

```python
# GOOD - focused functions
@expose
def get_embedding(text: str) -> dict:
    ...

@expose
def compute_similarity(text1: str, text2: str) -> dict:
    ...

# BAD - do-everything function
@expose
def do_everything(text1: str, text2: str, mode: str) -> dict:
    if mode == "embed":
        ...
    elif mode == "similarity":
        ...
```

### Load Models Once

Models load when the RPC server starts (at module level), not inside functions:

```python
# GOOD - load once at module level
model = AutoModel.from_pretrained(model_path)

@expose
def use_model(text: str) -> dict:
    # Model already loaded
    return model(...)

# BAD - reloading on every call
@expose
def use_model(text: str) -> dict:
    model = AutoModel.from_pretrained(...)  # Slow!
    return model(...)
```

## Debugging

Test your interface file locally before using it in RPC:

```python
# test_interface.py
import sys
sys.path.insert(0, "experiments/my-experiment")

# Mock the injected globals
def get_model_path(model_id):
    return f"/path/to/{model_id}"

class expose:
    def __init__(self, fn):
        self.fn = fn
    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

# Now import and test
import interface
result = interface.get_model_info()
print(result)
```
