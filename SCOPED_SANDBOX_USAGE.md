# ScopedSandbox Usage Guide

## Overview

ScopedSandbox serves Python functions over RPC. Your interface code is **completely self-contained** - you control all imports, setup, and state.

## Minimal Example

### 1. Write Your Interface (Runs in Sandbox)

```python
# interface.py - completely self-contained

from transformers import AutoModel, AutoTokenizer
import torch

# Setup: Load models explicitly
model = AutoModel.from_pretrained("/models/gemma-2b", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("/models/gemma-2b")

# State: Module-level variables work
cache = {}

# Interface: Mark functions to expose
@expose  # <-- Only thing injected
def get_activations(prompt: str, layer: int) -> dict:
    """Extract activations at a specific layer."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    activations = outputs.hidden_states[layer].mean(dim=0)

    return {
        "activations": activations.tolist(),
        "shape": list(activations.shape),
        "layer": layer
    }

@expose
def list_cache() -> dict:
    """List cached items."""
    return {"cache": list(cache.keys())}

# Private helpers (not exposed)
def _internal_helper():
    pass
```

### 2. Serve the Interface

```python
from interp_infra.environment import ScopedSandbox, SandboxConfig, ModelConfig

# Create sandbox (models downloaded to volumes)
sandbox = ScopedSandbox(SandboxConfig(
    gpu="A100",
    models=[ModelConfig(name="google/gemma-2-9b")],  # Downloads to /models/
))

# Start and serve
sandbox.start()
library = sandbox.serve("interface.py", expose_as="library")

# Use in agent workspace
workspace = Workspace(libraries=[library])
```

## Using Configured Models

If you specify models in `SandboxConfig`, they're downloaded and available via environment variables:

### Option 1: Direct Environment Variables

```python
# interface.py
import os
from transformers import AutoModel

# Explicit path from env var set by SandboxConfig
model_path = os.environ["MODEL_GOOGLE_GEMMA_2_9B_PATH"]
model = AutoModel.from_pretrained(model_path)

@expose
def my_function():
    return model.generate(...)
```

### Option 2: Injected Helper Function

```python
# interface.py
from transformers import AutoModel

# get_model_path is injected by RPC server
model_path = get_model_path("google/gemma-2-9b")
model = AutoModel.from_pretrained(model_path)

@expose
def my_function():
    return model.generate(...)
```

### Option 3: List Available Models

```python
# interface.py

# list_configured_models is injected by RPC server
@expose
def show_models():
    """Show all configured models."""
    return list_configured_models()
```

## What Gets Injected?

**Three things:**
1. `@expose` - Decorator to mark functions for exposure
2. `get_model_path(name)` - Helper to get configured model paths
3. `list_configured_models()` - Helper to list all configured models

**Everything else** is explicitly in your code:
- Imports
- Model loading
- Setup code
- State management
- Helper functions

## Environment Variables

When you configure models in `SandboxConfig`:

```python
SandboxConfig(
    models=[
        ModelConfig(name="google/gemma-2-9b"),
        ModelConfig(name="meta-llama/Llama-2-7b"),
    ]
)
```

These environment variables are set:
- `MODEL_GOOGLE_GEMMA_2_9B_PATH=/volumes/google_gemma-2-9b`
- `MODEL_META_LLAMA_LLAMA_2_7B_PATH=/volumes/meta-llama_Llama-2-7b`

## Expose Modes

### As Library (Agent Imports)

```python
library = sandbox.serve("interface.py", expose_as="library")
workspace = Workspace(libraries=[library])

# Agent can:
# from interface import get_activations
# result = get_activations("hello", layer=10)
```

### As MCP Tools

```python
mcp_config = sandbox.serve("interface.py", expose_as="mcp")
run_agent(mcp_servers=[mcp_config], ...)

# Agent gets MCP tools for each @expose function
```

### As Prompt Documentation

```python
prompt = sandbox.serve("interface.py", expose_as="prompt")
run_agent(prompts=[prompt], ...)

# Agent sees function signatures and docstrings
```

## Complete Example

```python
# main.py
from interp_infra.environment import ScopedSandbox, SandboxConfig, ModelConfig
from interp_infra.workspace import Workspace
from interp_infra.execution import create_local_session
from interp_infra.harness import run_agent

# 1. Create scoped sandbox
sandbox = ScopedSandbox(SandboxConfig(
    gpu="A100",
    models=[ModelConfig(name="google/gemma-2-9b")],
))

# 2. Start and serve interface
sandbox.start()
tools = sandbox.serve("interface.py", expose_as="mcp")

# 3. Run agent with tools
session = create_local_session()
async for msg in run_agent(
    session=session,
    mcp_servers=[tools],
    task="Analyze model activations",
):
    print(msg)

# 4. Cleanup
sandbox.terminate()
```

## Key Principles

1. **Explicit over magic** - Your code shows exactly what it does
2. **Self-contained** - Interface file has all its dependencies
3. **Testable** - Can test locally before deploying to sandbox
4. **Flexible** - Full control over imports, setup, state
5. **Zero surprises** - Only `@expose` decorator is injected
