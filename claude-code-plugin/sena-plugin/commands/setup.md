---
description: Set up a GPU sandbox for interpretability research
---

# Sena Setup

Create a complete experiment setup with GPU sandbox for interpretability research.

## CRITICAL: DO NOT use start_new_session()

**That's for local-only sessions without GPU.** Follow this workflow instead.

---

## Step 1: Understand the Experiment

The user describes what they want to investigate. They can provide:
- **Detailed**: Full experiment description, specific techniques, models, etc.
- **Vague**: Just an idea like "investigate steering vectors on Gemma"

Ask clarifying questions only if needed:
- What model to investigate?
- What GPU? (default: A100)
- Any specific libraries needed? (steering, activations, etc.)
- Multi-agent setup? (if yes, need scoped sandbox + interface)

Create a short experiment name (e.g., "gemma-steering", "hidden-preference-probe")

---

## Step 2: Create Experiment Directory

Create a complete experiment structure like existing experiments:

```bash
experiments/<experiment-name>/
├── main.py          # Setup script (runs sandbox)
├── task.md          # Research task description
├── libraries/       # Optional: helper code
│   └── custom.py
└── interface.py     # Optional: for scoped sandbox RPC
```

Use Write tool to create these files.

---

## Step 3: Write main.py

Based on user's needs, choose the pattern:

### Pattern A: Regular Notebook (Most Common)

For standard interpretability experiments:

```python
"""<Experiment description from user>"""

import asyncio
from pathlib import Path
from src.environment import Sandbox, SandboxConfig, ExecutionMode, ModelConfig
from src.workspace import Workspace, Library
from src.execution import create_notebook_session
import json

async def main():
    example_dir = Path(__file__).parent

    config = SandboxConfig(
        execution_mode=ExecutionMode.NOTEBOOK,
        gpu="A100",  # or user's choice
        models=[ModelConfig(name="google/gemma-2-9b-it")],
        python_packages=["torch", "transformers", "accelerate"],
    )

    sandbox = Sandbox(config).start()

    # Add libraries if user wants them
    workspace = Workspace(
        libraries=[
            # Library.from_file(example_dir / "libraries" / "custom.py"),
        ]
    )

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

### Pattern B: Scoped Sandbox (Multi-Agent/RPC)

For multi-agent setups (like auditor/target):

```python
"""<Experiment description from user>"""

import asyncio
from pathlib import Path
from src.environment import ScopedSandbox, SandboxConfig, ModelConfig
from src.workspace import Workspace
from src.execution import create_local_session
import json

async def main():
    example_dir = Path(__file__).parent

    config = SandboxConfig(
        gpu="A100",
        models=[ModelConfig(name="google/gemma-2-9b")],
        python_packages=["torch", "transformers"],
    )

    scoped = ScopedSandbox(config)
    scoped.start()

    # Serve interface file
    interface_lib = scoped.serve(
        str(example_dir / "interface.py"),
        expose_as="library",  # or "mcp"
        name="model_tools"
    )

    workspace = Workspace(libraries=[interface_lib])
    session = create_local_session(
        workspace=workspace,
        workspace_dir=str(example_dir / "workspace"),
        name="experiment"
    )

    print(json.dumps({
        "scoped_ready": True,
        "interface": "model_tools",
        "workspace_dir": str(example_dir / "workspace"),
    }))

if __name__ == "__main__":
    asyncio.run(main())
```

**If user needs interface.py**, help create it based on:
- What functions should run on GPU?
- What should be exposed to the agent?
- See `experiments/scoped-simple/interface.py` as template

---

## Step 4: Write task.md

Create a clear research task description:

```markdown
# <Experiment Name>

## Research Question
<What are you investigating?>

## Setup
<What models, tools, environment>

## Task
<Step-by-step what to do>
- Phase 1: ...
- Phase 2: ...

## Expected Outputs
<What should the experiment produce?>
```

Be detailed enough that you can follow it after connecting.

---

## Step 5: Create Libraries (If Needed)

If user wants custom helper functions, create `libraries/`:

```python
# experiments/<name>/libraries/custom.py
"""Helper functions for this experiment."""

def analyze_activations(model, inputs):
    # Custom analysis code
    pass
```

Or copy from existing:
- `experiments/introspection/libraries/steering_hook.py`
- `experiments/introspection/libraries/extract_activations.py`

---

## Step 6: Create Interface (If Scoped Sandbox)

If using scoped sandbox, create `interface.py`:

```python
"""Interface for GPU sandbox - functions exposed via RPC."""

# These functions will be decorated with @expose
# See experiments/scoped-simple/interface.py for full example

@expose
def get_model_info() -> dict:
    """Get model information."""
    # Access pre-loaded model here
    return {"device": str(model.device)}

@expose
def run_analysis(prompt: str) -> dict:
    """Run analysis on prompt."""
    # Use model here
    pass
```

---

## Step 7: Run the Experiment

```bash
cd experiments/<experiment-name>
uv run python main.py
```

Takes ~5 min for GPU provisioning on first run.

---

## Step 8: Parse Output and Connect

### For Regular Notebook:

Look for JSON output:
```json
{
  "session_id": "abc123",
  "jupyter_url": "https://...",
  "model_info": "### Pre-loaded Models\nThe following models are already loaded..."
}
```

Extract values, then connect:
```python
attach_to_session(
    session_id="abc123",
    jupyter_url="https://..."
)
```

### For Scoped Sandbox:

No attach needed - the local session is already running.

---

## Step 9: Begin the Experiment

Now that you're connected, read the `task.md` you created and start the research!

### For Regular Notebook:

Show the `model_info` which looks like:
```
### Pre-loaded Models
The following models are already loaded in the kernel:
- `model`: google/gemma-2-9b-it
- `tokenizer`: Tokenizer

**Do NOT reload these models - they are already available.**
```

Tell user:
- ✓ Connected to session: `<session_id>`
- ✓ Jupyter URL: `<jupyter_url>`
- ✓ Models pre-loaded as: `model`, `tokenizer`
- ✓ Ready to execute code with `execute_code(session_id, code)`

### For Scoped Sandbox:

Tell user:
- ✓ GPU sandbox running
- ✓ Interface exposed as library: `model_tools`
- ✓ Import and use: `import model_tools; model_tools.function()`

---

## How Models Work (CRITICAL!)

**Models are PRE-LOADED into the notebook kernel namespace.**

When you run `create_notebook_session()`, the library:
1. Loads models from environment paths
2. Injects them as variables: `model`, `tokenizer` (or custom names)
3. Returns `session.model_info_text` telling you what exists

### ✅ Correct Usage:

```python
execute_code(session_id, """
import torch

# Models are ALREADY loaded - just use them!
print(f"Model device: {model.device}")
print(f"Model type: {type(model).__name__}")

# Use directly
inputs = tokenizer("Hello world", return_tensors="pt").to(model.device)
outputs = model(**inputs)
print(f"Logits: {outputs.logits.shape}")
""")
```

### ❌ WRONG - Don't Do This:

```python
# ❌ Don't load models manually!
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(...)  # WRONG!
```

The models are already loaded. Just use `model` and `tokenizer` directly.

---

## Common Configurations

### Basic Interpretability
```python
models=[ModelConfig(name="google/gemma-2-9b-it")]
python_packages=["torch", "transformers", "accelerate"]
```

### With Steering Libraries
```python
workspace = Workspace(libraries=[
    Library.from_file("experiments/introspection/libraries/steering_hook.py"),
    Library.from_file("experiments/introspection/libraries/extract_activations.py"),
])
```

### PEFT Adapter Investigation
```python
models=[ModelConfig(
    name="user/adapter-name",
    base_model="google/gemma-2-9b-it",
    is_peft=True,
    hidden=True  # hides name from Claude for blind investigation
)]
```

### With External Repo
```python
repos=[RepoConfig(url="owner/repo")]
system_packages=["git"]
```

### With API Access
```python
secrets=["OPENAI_API_KEY", "HUGGINGFACE_TOKEN"]
```

---

## Troubleshooting

### Modal Setup
- User needs: `modal token new`
- Set secrets: `modal secret create huggingface-secret HF_TOKEN=...`

### Model Loading Fails
- Check HuggingFace token is set as Modal secret
- Verify model name is exact

### Connection Fails
- Verify jupyter_url is accessible
- Check session_id matches

---

## Complete Example Workflow

**User says**: "I want to investigate steering vectors on Gemma-2-9B to make it more helpful"

**You do**:

1. Create directory: `experiments/gemma-helpful-steering/`

2. Write `main.py`:
```python
"""Investigate steering vectors on Gemma-2-9B for helpfulness."""
import asyncio
from pathlib import Path
from src.environment import Sandbox, SandboxConfig, ExecutionMode, ModelConfig
from src.workspace import Workspace, Library
from src.execution import create_notebook_session
import json

async def main():
    example_dir = Path(__file__).parent
    config = SandboxConfig(
        execution_mode=ExecutionMode.NOTEBOOK,
        gpu="A100",
        models=[ModelConfig(name="google/gemma-2-9b-it")],
        python_packages=["torch", "transformers", "accelerate"],
    )
    sandbox = Sandbox(config).start()
    workspace = Workspace(libraries=[
        Library.from_file("experiments/introspection/libraries/steering_hook.py"),
    ])
    session = create_notebook_session(sandbox, workspace)
    print(json.dumps({
        "session_id": session.session_id,
        "jupyter_url": session.jupyter_url,
        "model_info": session.model_info_text,
    }))

if __name__ == "__main__":
    asyncio.run(main())
```

3. Write `task.md`:
```markdown
# Gemma Helpful Steering

## Goal
Find steering vectors that make Gemma more helpful in responses.

## Approach
1. Extract activations from helpful vs unhelpful responses
2. Compute difference vectors
3. Test steering by adding vectors during inference
4. Measure helpfulness improvement

## Tools Available
- steering_hook.py for applying vectors
- Pre-loaded: model, tokenizer
```

4. Run it:
```bash
cd experiments/gemma-helpful-steering
uv run python main.py
```

5. Parse output, connect:
```python
attach_to_session(session_id="...", jupyter_url="...")
```

6. Read `task.md` and execute the experiment!

---

**Summary**: Create full experiment → Run main.py → Connect → Follow task.md
