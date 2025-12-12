# Seer Documentation (LLM Context)

Copy this entire page to give an LLM context about Seer.

---

## What is Seer?

Seer is a framework for having agents conduct interpretability work and investigations. The core mechanism involves launching a remote sandbox hosted on a remote GPU or CPU. The agent operates an IPython kernel and notebook on this remote host.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                         Your Machine                         │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                        Harness                          │ │
│  │  run_agent(prompt, mcp_config, provider="claude")       │ │
│  └───────────────────────────┬─────────────────────────────┘ │
│                              │ MCP                           │
│                              ▼                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                        Session                          │ │
│  │  Notebook: agent works in Jupyter                       │ │
│  │  Local: agent runs locally, calls GPU via RPC           │ │
│  └───────────────────────────┬─────────────────────────────┘ │
└──────────────────────────────┼───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                      Modal (Remote GPU)                      │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                       Sandbox                           │ │
│  │  - GPU (A100, H100, etc.)                               │ │
│  │  - Models (cached on Modal volumes)                     │ │
│  │  - Workspace libraries                                  │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

---

## API Reference

### SandboxConfig

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
    local_dirs: list[tuple] = [],
    env: dict[str, str] = {},           # Environment variables
)
```

### ModelConfig

```python
ModelConfig(
    name: str,                          # HuggingFace model ID
    var_name: str = "model",            # Variable name in model info
    hidden: bool = False,               # Hide model name from agent
    is_peft: bool = False,              # Is a PEFT adapter
    base_model: str = None,             # Base model ID if PEFT
)
```

### RepoConfig

```python
RepoConfig(
    url: str,                           # GitHub repo (e.g., "user/repo")
    dockerfile: str = None,
    install: bool = False,              # Run pip install
)
```

### Sandbox

```python
sandbox = Sandbox(config).start()
```

Methods:
- `start()` → Sandbox — provision GPU, download models
- `terminate()` — shutdown sandbox
- `exec(cmd: str)` → str — execute shell command
- `exec_python(code: str)` → str — execute Python code

### ScopedSandbox

```python
scoped = ScopedSandbox(config).start()
lib = scoped.serve("interface.py", expose_as="library", name="model_tools")
```

Methods:
- `start()` — provision sandbox
- `serve(file, expose_as, name)` → Library | dict — serve file as RPC
- `write_file(path, content)` — write file to sandbox
- `exec(cmd)` → str — execute shell command
- `terminate()` — shutdown sandbox

expose_as options:
- `"library"` — agent imports it
- `"mcp"` — agent sees MCP tools

### Workspace

```python
Workspace(
    libraries: list[Library] = [],
    skills: list[Skill] = [],
    local_dirs: list[tuple] = [],
    custom_init_code: str = None,
)
```

Methods:
- `get_library_docs()` → str — combined docs for agent prompt

### Library

```python
lib = Library.from_file("helpers.py")
lib = Library.from_code("utils", "def foo(): ...")
lib = scoped.serve("interface.py", expose_as="library", name="tools")
```

### Sessions

```python
# Notebook mode — agent works in Jupyter on GPU
session = create_notebook_session(sandbox, workspace)
# Returns: mcp_config, jupyter_url, model_info_text, session_id

# Local mode — agent runs locally, calls GPU via RPC
session = create_local_session(workspace, workspace_dir, name)
# Returns: mcp_config, name, workspace_dir
```

### Harness

```python
async for msg in run_agent(
    prompt: str,
    mcp_config: dict = {},
    provider: str = "claude",  # or "openai", "gemini"
    model: str = None,
    user_message: str = None,
):
    print(msg)
```

---

## Common Patterns

### Basic notebook experiment

```python
from src.environment import Sandbox, SandboxConfig, ExecutionMode, ModelConfig
from src.workspace import Workspace, Library
from src.execution import create_notebook_session
from src.harness import run_agent

config = SandboxConfig(
    gpu="A100",
    execution_mode=ExecutionMode.NOTEBOOK,
    models=[ModelConfig(name="google/gemma-2-9b-it")],
    python_packages=["torch", "transformers", "accelerate"],
)
sandbox = Sandbox(config).start()

workspace = Workspace(libraries=[
    Library.from_file("experiments/toolkit/steering_hook.py"),
    Library.from_file("experiments/toolkit/extract_activations.py"),
])

session = create_notebook_session(sandbox, workspace)

async for msg in run_agent(prompt=task, mcp_config=session.mcp_config):
    pass

sandbox.terminate()
```

### Scoped sandbox with RPC

```python
from src.environment import ScopedSandbox, SandboxConfig, ModelConfig
from src.workspace import Workspace
from src.execution import create_local_session

scoped = ScopedSandbox(SandboxConfig(
    gpu="A100",
    models=[ModelConfig(name="google/gemma-2-9b")],
)).start()

model_tools = scoped.serve("interface.py", expose_as="library", name="model_tools")

workspace = Workspace(libraries=[model_tools])
session = create_local_session(workspace, workspace_dir="./workspace")

async for msg in run_agent(prompt=task, mcp_config={}):
    pass

scoped.terminate()
```

### Interface file for scoped sandbox

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
- Must return JSON-serializable types
- `get_model_path()` is injected
- Load models at module level

### PEFT adapter (hidden)

```python
ModelConfig(
    name="user/adapter",
    base_model="google/gemma-2-9b-it",
    is_peft=True,
    hidden=True  # Hide adapter name from agent
)
```

### External repos

```python
SandboxConfig(
    repos=[RepoConfig(url="user/repo", install=True)],
    system_packages=["git"],
)
```

---

## Toolkit

Reusable interpretability tools in `experiments/toolkit/`:

- `extract_activations.py` — `extract_activation(model, tokenizer, text, layer_idx)`
- `steering_hook.py` — `create_steering_hook(model, layer_idx, vector, strength)`
- `generate_response.py` — `generate_response(model, tokenizer, prompt)`

```python
from extract_activations import extract_activation
from steering_hook import create_steering_hook

# Extract concept direction
act1 = extract_activation(model, tokenizer, "I feel happy", layer_idx=15)
act2 = extract_activation(model, tokenizer, "I feel sad", layer_idx=15)
direction = act1 - act2

# Steer generation
with create_steering_hook(model, layer_idx=15, vector=direction, strength=2.0):
    output = model.generate(...)
```
