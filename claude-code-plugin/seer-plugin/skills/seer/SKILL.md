# Seer - Sandboxed Environments for Interpretability Research

## Overview

Seer provides GPU-accelerated sandboxed environments for running interpretability experiments. You can set up remote environments with models pre-loaded, connect to Jupyter notebooks, and run experiments interactively.

## When to Use This Skill

Use this skill when the user wants to:
- Run interpretability experiments on models (steering, activation analysis, probing, etc.)
- Work with models that need GPU acceleration (A100, H100, etc.)
- Set up reproducible experiment environments with pre-loaded models
- Run code in isolated sandboxes with specific dependencies
- Expose GPU sandbox functions to local code via RPC

---

## Complete API Reference

### Core Classes

#### SandboxConfig

Configuration for a Modal sandbox environment.

```python
@dataclass
class SandboxConfig:
    gpu: Optional[str] = None              # "A100", "H100", "T4", None for CPU
    execution_mode: ExecutionMode = ExecutionMode.NOTEBOOK  # NOTEBOOK or CLI
    models: List[ModelConfig] = []         # Models to pre-load
    python_packages: List[str] = []        # pip packages
    system_packages: List[str] = []        # apt packages
    secrets: List[str] = []                # Modal secret names
    repos: List[RepoConfig] = []           # Git repos to clone
    env: Dict[str, str] = {}               # Environment variables
    timeout: int = 3600                    # Timeout in seconds
    local_files: List[Tuple[str, str]] = [] # (local_path, sandbox_path)
```

**Example:**
```python
config = SandboxConfig(
    gpu="A100",
    execution_mode=ExecutionMode.NOTEBOOK,
    models=[ModelConfig(name="google/gemma-2-9b-it")],
    python_packages=["torch", "transformers", "accelerate", "matplotlib"],
    system_packages=["git"],
    secrets=["HF_TOKEN", "OPENAI_API_KEY"],
    env={"CUSTOM_VAR": "value"},
    timeout=7200,  # 2 hours
)
```

#### ModelConfig

Configuration for a model to load in the sandbox.

```python
@dataclass
class ModelConfig:
    name: str                              # HuggingFace model ID
    var_name: str = "model"                # Variable name in notebook
    hidden: bool = False                   # Hide details from agent
    is_peft: bool = False                  # Is PEFT adapter
    base_model: Optional[str] = None       # Base model for PEFT
```

**Examples:**

Basic model:
```python
ModelConfig(name="google/gemma-2-9b-it")
```

Multiple models with custom names:
```python
models=[
    ModelConfig(name="google/gemma-2-9b-it", var_name="model_a"),
    ModelConfig(name="meta-llama/Llama-2-7b-hf", var_name="model_b"),
]
```

PEFT adapter (for investigating fine-tuned models):
```python
ModelConfig(
    name="user/gemma-adapter",
    base_model="google/gemma-2-9b-it",
    is_peft=True,
    hidden=True  # Hide the adapter name from the agent
)
```

#### RepoConfig

Configuration for cloning Git repositories.

```python
@dataclass
class RepoConfig:
    url: str                               # Git URL (e.g., "owner/repo" or full URL)
    dockerfile: Optional[str] = None       # Path to Dockerfile in repo
    install: bool = False                  # pip install -e the repo
```

**Example:**
```python
repos=[RepoConfig(url="anthropics/circuits", install=True)]
```

### Session Types

#### Sandbox (Regular Notebook Mode)

Standard GPU sandbox with Jupyter notebook. **Use this for most experiments.**

```python
from src.environment import Sandbox, SandboxConfig, ExecutionMode, ModelConfig
from src.workspace import Workspace, Library
from src.execution import create_notebook_session
import json

config = SandboxConfig(
    execution_mode=ExecutionMode.NOTEBOOK,
    gpu="A100",
    models=[ModelConfig(name="google/gemma-2-9b-it")],
    python_packages=["torch", "transformers", "accelerate"],
)

sandbox = Sandbox(config).start()

workspace = Workspace(
    libraries=[
        Library.from_file("path/to/helper.py"),
    ]
)

session = create_notebook_session(sandbox, workspace)

# Output connection info as JSON
print(json.dumps({
    "session_id": session.session_id,
    "jupyter_url": session.jupyter_url,
    "model_info": session.model_info_text,
}))
```

#### ScopedSandbox (RPC Interface Mode)

Isolated GPU sandbox with RPC interface. **Use when you need to expose GPU functions to local code.**

```python
from src.environment import ScopedSandbox, SandboxConfig, ModelConfig
from src.workspace import Workspace
from src.execution import create_local_session

config = SandboxConfig(
    gpu="A100",
    models=[ModelConfig(name="google/gemma-2-9b")],
    python_packages=["torch", "transformers"],
)

scoped = ScopedSandbox(config)
scoped.start()

# Serve an interface file via RPC
interface_lib = scoped.serve(
    "path/to/interface.py",
    expose_as="library",  # or "mcp" for MCP server
    name="model_tools"
)

workspace = Workspace(libraries=[interface_lib])

session = create_local_session(
    workspace=workspace,
    workspace_dir="./workspace",
    name="experiment"
)

# Now the interface functions are available via RPC
```

---

## How Models Work (CRITICAL!)

**Models are PRE-LOADED into the notebook kernel namespace.**

When you run `create_notebook_session()`, the library:
1. Downloads models to Modal volumes (cached for future runs)
2. Loads them into memory on the GPU
3. Injects them as Python variables: `model`, `tokenizer` (or custom var_name)
4. Returns `session.model_info_text` describing what's available

### Model Information Text

The `session.model_info_text` contains critical information:

```
### Pre-loaded Models

The following models are already loaded in the kernel:

**model** (google/gemma-2-9b-it)
- Type: Gemma2ForCausalLM
- Device: cuda:0
- Parameters: 9.24B

**tokenizer** (google/gemma-2-9b-it)
- Type: GemmaTokenizerFast
- Vocab size: 256,000

**IMPORTANT:** Do NOT reload these models. They are already loaded and ready to use.
```

### ✅ Correct Usage

```python
execute_code(session_id, """
import torch

# Models are ALREADY loaded - just use them!
print(f"Model device: {model.device}")
print(f"Model type: {type(model).__name__}")

# Use directly
inputs = tokenizer("Hello world", return_tensors="pt").to(model.device)
outputs = model(**inputs)
print(f"Logits shape: {outputs.logits.shape}")
""")
```

### ❌ WRONG - Don't Do This

```python
# ❌ Don't load models manually!
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(...)  # WRONG!

# The models are already loaded. Just use `model` and `tokenizer` directly.
```

### Multiple Models

When you load multiple models:

```python
config = SandboxConfig(
    models=[
        ModelConfig(name="google/gemma-2-9b-it", var_name="gemma"),
        ModelConfig(name="meta-llama/Llama-2-7b-hf", var_name="llama"),
    ]
)
```

Both will be pre-loaded and available:

```python
execute_code(session_id, """
# Use both models
gemma_output = gemma.generate(**gemma_inputs)
llama_output = llama.generate(**llama_inputs)
""")
```

---

## Interface Files for RPC (Scoped Sandbox)

When using `ScopedSandbox`, you create an interface file that exposes functions via RPC. This lets you run functions on the GPU while your main code runs locally.

### Basic Interface Structure

```python
"""interface.py - Functions that run on GPU via RPC"""

from transformers import AutoModel, AutoTokenizer
import torch

# get_model_path() is injected by the RPC server
model_path = get_model_path("google/gemma-2-9b")
model = AutoModel.from_pretrained(model_path, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)


@expose  # @expose decorator is also injected by RPC server
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
def analyze_text(text: str) -> dict:
    """Analyze text using the model."""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Return simple types (dict, list, str, int, float, bool)
    return {
        "text": text,
        "num_tokens": len(inputs.input_ids[0]),
        "logits_shape": list(outputs.logits.shape),
    }
```

### Key Points for Interfaces

1. **`get_model_path(model_name)`**: Injected helper to get local model path
2. **`@expose`**: Decorator injected by RPC server to expose functions
3. **Return simple types**: dict, list, str, int, float, bool (no tensors!)
4. **Load models at module level**: They'll be loaded once when interface starts
5. **Type hints**: Recommended for clarity

### Advanced Interface with State

```python
"""interface.py - Stateful interface with conversation history"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = get_model_path("google/gemma-2-9b-it")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# State (persists across calls)
conversation_history = []


@expose
def send_message(message: str, max_tokens: int = 512) -> dict:
    """Send message and get response."""
    global conversation_history

    # Add to history
    conversation_history.append({"role": "user", "content": message})

    # Format prompt
    prompt = tokenizer.apply_chat_template(
        conversation_history,
        tokenize=False,
        add_generation_prompt=True
    )

    # Generate
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=0.7,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just the new response
    response = response[len(prompt):].strip()

    # Add to history
    conversation_history.append({"role": "assistant", "content": response})

    return {
        "response": response,
        "history_length": len(conversation_history),
    }


@expose
def reset_conversation() -> str:
    """Reset conversation history."""
    global conversation_history
    conversation_history = []
    return "Conversation reset"


@expose
def get_history() -> list:
    """Get full conversation history."""
    return conversation_history.copy()
```

### Complex Interface Example (MCP-style)

For advanced use cases, you can create sophisticated interfaces with multiple functions and state:

```python
"""conversation_interface.py - Full conversation management via RPC"""

from typing import Optional
import asyncio

# Import other local files (they're all in /root/ in the sandbox)
from target_agent import TargetAgent

_target: Optional[TargetAgent] = None

def get_target(model: str = "openai/gpt-4o-mini") -> TargetAgent:
    """Get or create singleton target."""
    global _target
    if _target is None:
        _target = TargetAgent(model=model)
    return _target


@expose
def initialize_target(system_message: str) -> str:
    """Initialize target with system prompt."""
    target = get_target()
    asyncio.run(target.initialize(system_message))
    return "Target initialized"


@expose
def send_to_target(message: str) -> dict:
    """Send message to target and get response."""
    target = get_target()
    response = asyncio.run(target.send_message(message))

    return {
        "type": response["type"],
        "content": response.get("content", ""),
        "tool_calls": response.get("tool_calls", []),
    }
```

See `experiments/petri-style-harness/conversation_interface.py` for a complete example.

---

## Libraries and Workspaces

### Creating Libraries

Libraries are Python files that get injected into the sandbox environment.

**From file:**
```python
from src.workspace import Library

lib = Library.from_file("path/to/helper.py")
```

**From code string:**
```python
code = '''
def my_helper(x):
    return x * 2
'''

lib = Library.from_code(code, name="helpers")
```

**From RPC interface (ScopedSandbox only):**
```python
interface_lib = scoped.serve(
    "path/to/interface.py",
    expose_as="library",
    name="model_tools"
)
```

### Using Workspaces

```python
from src.workspace import Workspace, Library

workspace = Workspace(
    libraries=[
        Library.from_file("experiments/toolkit/steering_hook.py"),
        Library.from_file("experiments/toolkit/extract_activations.py"),
        Library.from_file("custom_helpers.py"),
    ]
)

session = create_notebook_session(sandbox, workspace)
```

Now these libraries are importable in the notebook:

```python
execute_code(session_id, """
from steering_hook import create_steering_hook
from extract_activations import get_layer_activations
import custom_helpers

# Use the libraries
with create_steering_hook(model, layer_idx=20, vector=vec, strength=2.0):
    outputs = model.generate(...)
""")
```

---

## Complete Workflow Examples

### Example 1: Basic Interpretability Experiment

```python
"""Basic steering vectors experiment on Gemma."""

import asyncio
from pathlib import Path
from src.environment import Sandbox, SandboxConfig, ExecutionMode, ModelConfig
from src.workspace import Workspace, Library
from src.execution import create_notebook_session
import json

async def main():
    # Configure sandbox
    config = SandboxConfig(
        execution_mode=ExecutionMode.NOTEBOOK,
        gpu="A100",
        models=[ModelConfig(name="google/gemma-2-9b-it")],
        python_packages=["torch", "transformers", "accelerate", "matplotlib", "numpy"],
    )

    # Start sandbox (takes ~5min first time, <1min after)
    sandbox = Sandbox(config).start()

    # Add helper libraries
    workspace = Workspace(
        libraries=[
            Library.from_file("experiments/toolkit/steering_hook.py"),
            Library.from_file("experiments/toolkit/extract_activations.py"),
        ]
    )

    # Create notebook session
    session = create_notebook_session(sandbox, workspace)

    # Output connection info
    print(json.dumps({
        "session_id": session.session_id,
        "jupyter_url": session.jupyter_url,
        "model_info": session.model_info_text,
    }))

if __name__ == "__main__":
    asyncio.run(main())
```

After running this script, parse the JSON output and connect:

```python
attach_to_session(
    session_id="<from output>",
    jupyter_url="<from output>"
)
```

Then execute experiments:

```python
execute_code(session_id, """
import torch
from steering_hook import create_steering_hook

# Model is already loaded
print(f"Model: {type(model).__name__} on {model.device}")

# Extract steering vector from contrast pair
from extract_activations import get_layer_activations

positive_text = "I strongly agree with your perspective."
negative_text = "I disagree with your perspective."

pos_acts = get_layer_activations(model, tokenizer, positive_text, layer=20)
neg_acts = get_layer_activations(model, tokenizer, negative_text, layer=20)

steering_vector = pos_acts - neg_acts

# Test steering
test_prompt = "What do you think about this idea?"
inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

# Generate with steering
with create_steering_hook(model, layer_idx=20, vector=steering_vector, strength=2.0):
    outputs = model.generate(**inputs, max_new_tokens=50)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Steered response: {response}")
""")
```

### Example 2: Hidden Preference Investigation (PEFT)

```python
"""Investigate hidden preferences in a fine-tuned model."""

import asyncio
from pathlib import Path
from src.environment import Sandbox, SandboxConfig, ExecutionMode, ModelConfig
from src.workspace import Workspace, Library
from src.execution import create_notebook_session
import json

async def main():
    example_dir = Path(__file__).parent
    toolkit = example_dir.parent / "toolkit"

    # Configure with PEFT adapter (hidden from agent)
    config = SandboxConfig(
        gpu="A100",
        execution_mode=ExecutionMode.NOTEBOOK,
        models=[ModelConfig(
            name="user/gemma-adapter-secret-preference",
            base_model="google/gemma-2-9b-it",
            is_peft=True,
            hidden=True  # Agent won't know which adapter
        )],
        python_packages=["torch", "transformers", "accelerate", "datasets", "peft"],
        secrets=["HF_TOKEN"],
    )

    sandbox = Sandbox(config).start()

    workspace = Workspace(
        libraries=[
            Library.from_file(toolkit / "steering_hook.py"),
            Library.from_file(toolkit / "extract_activations.py"),
            Library.from_file(toolkit / "generate_response.py"),
        ]
    )

    session = create_notebook_session(sandbox, workspace)

    # Include research methodology
    task = (example_dir / "task.md").read_text()
    methodology = (toolkit / "research_methodology.md").read_text()

    print(json.dumps({
        "session_id": session.session_id,
        "jupyter_url": session.jupyter_url,
        "model_info": session.model_info_text,
        "task": task,
        "methodology": methodology,
    }))

if __name__ == "__main__":
    asyncio.run(main())
```

### Example 3: Scoped Sandbox with RPC Interface

```python
"""Expose GPU functions via RPC interface."""

import asyncio
from pathlib import Path
from src.environment import ScopedSandbox, SandboxConfig, ModelConfig
from src.workspace import Workspace
from src.execution import create_local_session
import json

async def main():
    example_dir = Path(__file__).parent

    # Create scoped sandbox with GPU
    config = SandboxConfig(
        gpu="A100",
        models=[ModelConfig(name="google/gemma-2-9b")],
        python_packages=["torch", "transformers", "anthropic", "openai"],
        secrets=["ANTHROPIC_API_KEY", "OPENAI_API_KEY"],
    )

    scoped = ScopedSandbox(config)
    scoped.start()

    # Serve interface as MCP server
    interface_lib = scoped.serve(
        str(example_dir / "conversation_interface.py"),
        expose_as="mcp",
        name="conversation_tools"
    )

    # Also upload supporting files
    # (they'll be in /root/ alongside interface.py)
    scoped.upload_file(str(example_dir / "target_agent.py"))
    scoped.upload_file(str(example_dir / "prompts.py"))

    # Create local session with MCP tools
    workspace = Workspace(libraries=[interface_lib])

    session = create_local_session(
        workspace=workspace,
        workspace_dir=str(example_dir / "workspace"),
        name="multi-agent-experiment"
    )

    print(json.dumps({
        "status": "ready",
        "interface": "conversation_tools (MCP)",
        "workspace_dir": str(example_dir / "workspace"),
    }))

    # Now you can use the MCP tools from your local code

if __name__ == "__main__":
    asyncio.run(main())
```

### Example 4: Using External Repos

```python
"""Experiment using code from an external repository."""

import asyncio
from src.environment import Sandbox, SandboxConfig, ExecutionMode, ModelConfig, RepoConfig
from src.execution import create_notebook_session
import json

async def main():
    config = SandboxConfig(
        execution_mode=ExecutionMode.NOTEBOOK,
        gpu="A100",
        models=[ModelConfig(name="google/gemma-2-9b-it")],
        python_packages=["torch", "transformers", "accelerate"],
        system_packages=["git"],
        repos=[
            RepoConfig(
                url="anthropics/transformer-circuits",
                install=True  # Run pip install -e on the repo
            )
        ],
    )

    sandbox = Sandbox(config).start()
    session = create_notebook_session(sandbox, Workspace())

    print(json.dumps({
        "session_id": session.session_id,
        "jupyter_url": session.jupyter_url,
    }))

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Shared Libraries Reference

The `experiments/toolkit/` directory contains reusable interpretability tools:

### steering_hook.py

```python
from steering_hook import create_steering_hook

# Use as context manager
with create_steering_hook(model, layer_idx=20, vector=steering_vec, strength=2.0, start_pos=0):
    outputs = model.generate(**inputs)
```

**Parameters:**
- `model`: The language model
- `layer_idx`: Which layer to inject into (0-indexed)
- `vector`: Steering vector (torch.Tensor)
- `strength`: Multiplier for injection strength
- `start_pos`: Token position to start steering from

### extract_activations.py

```python
from extract_activations import get_layer_activations, get_all_layer_activations

# Get activations from a specific layer
acts = get_layer_activations(model, tokenizer, text="Hello world", layer=15)
# Returns: torch.Tensor of shape (hidden_dim,)

# Get activations from all layers
all_acts = get_all_layer_activations(model, tokenizer, text="Hello world")
# Returns: List[torch.Tensor], one per layer
```

### generate_response.py

```python
from generate_response import generate_response, batch_generate

# Generate single response
response = generate_response(
    model,
    tokenizer,
    prompt="What is AI?",
    max_tokens=256,
    temperature=0.7,
)

# Batch generate multiple responses
prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
responses = batch_generate(model, tokenizer, prompts, max_tokens=128)
```

### research_methodology.md

A skill that provides research guidance. Include it in your prompt:

```python
methodology = Path("experiments/toolkit/research_methodology.md").read_text()
prompt = f"{session.model_info_text}\n\n{methodology}\n\n{task}"
```

Key principles:
- Explore much more than exploit
- Use mandatory exploration checkpoints
- Pivot when signal is weak
- Test falsifiable hypotheses
- Be actively skeptical

---

## Common Patterns

### Pattern 1: Quick Single-Model Experiment

**When:** You want to quickly test something on a model.

```python
config = SandboxConfig(
    execution_mode=ExecutionMode.NOTEBOOK,
    gpu="A100",
    models=[ModelConfig(name="google/gemma-2-9b-it")],
    python_packages=["torch", "transformers", "accelerate"],
)
sandbox = Sandbox(config).start()
session = create_notebook_session(sandbox, Workspace())
# -> attach and experiment
```

### Pattern 2: Steering Investigation

**When:** You want to investigate steering vectors.

```python
config = SandboxConfig(
    gpu="A100",
    execution_mode=ExecutionMode.NOTEBOOK,
    models=[ModelConfig(name="google/gemma-2-9b-it")],
    python_packages=["torch", "transformers", "accelerate", "matplotlib"],
)
sandbox = Sandbox(config).start()
workspace = Workspace(libraries=[
    Library.from_file("experiments/toolkit/steering_hook.py"),
    Library.from_file("experiments/toolkit/extract_activations.py"),
])
session = create_notebook_session(sandbox, workspace)
# -> attach, extract vectors, test steering
```

### Pattern 3: Hidden Adapter Investigation

**When:** You want to investigate a fine-tuned model without revealing which one.

```python
config = SandboxConfig(
    gpu="A100",
    models=[ModelConfig(
        name="user/secret-adapter",
        base_model="google/gemma-2-9b-it",
        is_peft=True,
        hidden=True,
    )],
    python_packages=["torch", "transformers", "peft"],
    secrets=["HF_TOKEN"],
)
```

### Pattern 4: RPC Interface Setup

**When:** You need local code to call GPU functions remotely.

```python
scoped = ScopedSandbox(SandboxConfig(gpu="A100", models=[...]))
scoped.start()
interface_lib = scoped.serve("interface.py", expose_as="library", name="tools")
workspace = Workspace(libraries=[interface_lib])
session = create_local_session(workspace=workspace, workspace_dir="./work", name="exp")
# -> local code can import and call GPU functions via RPC
```

### Pattern 5: Using Secrets

**When:** You need API keys or credentials.

```python
config = SandboxConfig(
    gpu="A100",
    secrets=["HF_TOKEN", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"],
)
```

Secrets are available as environment variables in the sandbox:
```python
execute_code(session_id, """
import os
hf_token = os.environ.get("HF_TOKEN")
openai_key = os.environ.get("OPENAI_API_KEY")
""")
```

---

## Troubleshooting

### Modal Setup Required

Users need Modal configured:
```bash
modal token new
```

And secrets for HuggingFace:
```bash
modal secret create huggingface-secret HF_TOKEN=hf_...
```

### First Run is Slow

First sandbox creation takes ~5 minutes because:
1. GPU provisioning (~2 min)
2. Model download (~3 min)

Subsequent runs are much faster (<1 min) because models are cached.

### Model Loading Errors

Common issues:
- **Invalid model name**: Verify exact HuggingFace ID
- **No HF token**: Private models need `secrets=["HF_TOKEN"]`
- **OOM**: Model too large for GPU (try smaller model or H100)

### Connection Issues

- Ensure jupyter_url is accessible from your network
- Check session_id matches exactly
- Verify Modal app is still running (apps time out after 1 hour by default)

### Package Installation

If a package fails to install:
1. Check if it needs system dependencies (add to `system_packages`)
2. Try pinning version: `"transformers==4.36.0"`
3. Some packages require custom Docker config (advanced)

---

## Best Practices

### 1. Start Small

Begin with minimal config, add complexity incrementally:

```python
# Start here
config = SandboxConfig(
    gpu="A100",
    models=[ModelConfig(name="google/gemma-2-9b-it")],
    python_packages=["torch", "transformers", "accelerate"],
)

# Add as needed
# python_packages += ["matplotlib", "pandas", "datasets"]
# libraries = [Library.from_file("steering_hook.py")]
# secrets = ["HF_TOKEN"]
```

### 2. Reuse Sessions

Don't create new sandboxes unnecessarily. Attach to existing sessions when possible.

### 3. Use Libraries

Don't copy-paste helper code into every experiment. Use the toolkit:
- `experiments/toolkit/steering_hook.py`
- `experiments/toolkit/extract_activations.py`
- `experiments/toolkit/generate_response.py`

### 4. Document as You Go

Use `add_markdown()` to document your experiments in the notebook:

```python
add_markdown(session_id, """
## Hypothesis 1: Model has strong helpfulness steering vector

Testing by:
1. Extracting contrast pair activations
2. Computing difference vector
3. Applying with varying strengths
""")
```

### 5. Include Research Methodology

For investigative work, include the research methodology skill:

```python
methodology = Path("experiments/toolkit/research_methodology.md").read_text()
prompt = f"{methodology}\n\n{task}"
```

This helps you avoid common pitfalls like premature exploitation.

### 6. Use Type Hints in Interfaces

Make interfaces clear with type hints:

```python
@expose
def analyze_text(text: str, layer: int = 20) -> dict:
    """Analyze text at specific layer."""
    ...
```

---

## Quick Reference

### Essential Imports

```python
from src.environment import Sandbox, SandboxConfig, ExecutionMode, ModelConfig, RepoConfig
from src.environment import ScopedSandbox
from src.workspace import Workspace, Library
from src.execution import create_notebook_session, create_local_session
import json
```

### Minimal Notebook Setup

```python
config = SandboxConfig(
    execution_mode=ExecutionMode.NOTEBOOK,
    gpu="A100",
    models=[ModelConfig(name="google/gemma-2-9b-it")],
    python_packages=["torch", "transformers", "accelerate"],
)
sandbox = Sandbox(config).start()
session = create_notebook_session(sandbox, Workspace())
print(json.dumps({
    "session_id": session.session_id,
    "jupyter_url": session.jupyter_url,
}))
```

### Minimal RPC Setup

```python
scoped = ScopedSandbox(SandboxConfig(
    gpu="A100",
    models=[ModelConfig(name="google/gemma-2-9b")],
))
scoped.start()
interface = scoped.serve("interface.py", expose_as="library", name="tools")
workspace = Workspace(libraries=[interface])
```

---

## Tips for Success

1. **Read model_info_text**: Always show the user what models are pre-loaded
2. **Don't reload models**: They're already loaded, just use them
3. **Use workspaces**: Organize libraries for reusability
4. **Test interfaces locally first**: Debug RPC interfaces before using in experiments
5. **Monitor GPU usage**: Check if model fits in memory before starting
6. **Clean up**: Terminate sandboxes when done to avoid costs
7. **Version models**: Pin model revisions for reproducibility
8. **Check outputs**: Always verify execution results before proceeding

---

This skill enables powerful interpretability research workflows. The key is understanding the two modes:
- **Sandbox**: Use for Jupyter notebook experiments (most common)
- **ScopedSandbox**: Use when you need to expose GPU functions via RPC to local code
