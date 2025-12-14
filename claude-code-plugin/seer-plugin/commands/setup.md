---
description: Set up a GPU sandbox for interpretability research
---

# Seer Setup

You are helping the user set up an interpretability research experiment.

**IMPORTANT:** Before writing any setup.py code, consult the `seer` skill for the complete API reference. The skill contains critical information about SandboxConfig, ModelConfig, and all available parameters.

---

## Step 1: Understand the Research

**First, understand what they're investigating.** Ask them to describe their research goal.

If they just ran `/seer:setup` without context, ask:
```
What are you trying to investigate?
```

Listen to their description. Understand:
- What's the research question?
- What techniques might they need? (steering, activation analysis, probing, etc.)
- Do they need to interact with external code/repos?
- Is this a notebook experiment or do they need RPC/scoped sandbox?

**Don't jump to model/GPU questions yet.**

---

## Step 2: Determine Requirements

Based on their research, figure out:

**Model:**
- What model makes sense for their investigation?
- Do they need a specific model or is a default fine?
- Do they need multiple models for comparison?
- Is it a private/gated model (needs HF_TOKEN)?

**Setup type:**
- **Notebook sandbox** (default): Interactive Jupyter experiments
- **Scoped sandbox**: If they need to expose GPU functions via RPC to local code

**Repos:**
- Do they need to clone any external repositories?
- Any special dependencies?

**Ask clarifying questions only if needed.** If their description is clear, just confirm your understanding and proceed.

---

## Step 3: Create Experiment Directory

Create a directory structure:
```
experiments/<experiment-name>/
├── setup.py      # Sandbox setup script
└── task.md       # Research description
```

### task.md

Document their research:
```markdown
# <Experiment Name>

## Research Question
<What they told you they're investigating>

## Approach
<Brief description of the technical approach>

## Setup
- Model: <model>
- GPU: <gpu>
- Type: Notebook / Scoped Sandbox

## Notes
<Any specific details they mentioned>
```

---

## Step 4: Write setup.py

**IMPORTANT:** Consult the `seer` skill for exact API details, GPU options, and parameter specifications before writing this code.

**For Notebook Sandbox (most common):**

```python
from src.environment import Sandbox, SandboxConfig, ExecutionMode, ModelConfig
from src.workspace import Workspace
from src.execution import create_notebook_session
import json

config = SandboxConfig(
    execution_mode=ExecutionMode.NOTEBOOK,
    gpu="A100-40GB",  # or A100-80GB for larger models
    models=[ModelConfig(name="google/gemma-2-9b-it")],
    python_packages=["torch", "transformers", "accelerate", "matplotlib", "numpy"],
)

sandbox = Sandbox(config).start()
session = create_notebook_session(sandbox, Workspace())

print(json.dumps({
    "session_id": session.session_id,
    "jupyter_url": session.jupyter_url,
}))
```

**For Scoped Sandbox (RPC interface):**

```python
from src.environment import ScopedSandbox, SandboxConfig, ModelConfig
from src.workspace import Workspace
from src.execution import create_local_session
import json

config = SandboxConfig(
    gpu="A100-40GB",
    models=[ModelConfig(name="google/gemma-2-9b")],
    python_packages=["torch", "transformers"],
)

scoped = ScopedSandbox(config)
scoped.start()

interface_lib = scoped.serve(
    "interface.py",
    expose_as="library",
    name="model_tools"
)

print(json.dumps({"status": "ready", "interface": "model_tools"}))
```

If scoped sandbox, also create `interface.py` with the functions they need.

---

## Step 5: Run Setup

**IMPORTANT: Only now do you run the script.**

```bash
cd experiments/<experiment-name>
uv run python setup.py
```

This takes 3-5 minutes first time. Wait for the JSON output.

---

## Step 6: Connect

**Only after Step 5 completes**, parse the JSON and call:

```python
attach_to_session(session_id="<from output>", jupyter_url="<from output>")
```

---

## Step 7: Confirm Ready

Tell the user:
- Sandbox is ready
- Share the Jupyter URL so they can view the notebook
- Remind them that `model` and `tokenizer` are pre-loaded
- Summarize what's set up based on their research goal

Now you can use `execute_code()` to run experiments.

---

## Reference: GPU Selection

| Model Size | GPU | Notes |
|------------|-----|-------|
| 7B | `"A10G"` or `"L4"` | Cheapest option |
| 9B-13B | `"A100-40GB"` | Good default |
| 30B+ | `"A100-80GB"` | Need 80GB VRAM |
| 70B+ | `"H100"` or `gpu="A100-80GB", gpu_count=2` | Multi-GPU |

## Reference: Common Configurations

**Private/gated models:**
```python
secrets=["HF_TOKEN"]
```

**With external repos:**
```python
repos=[RepoConfig(url="org/repo", install=True)]
```

**Multiple models:**
```python
models=[
    ModelConfig(name="model-a", var_name="model_a"),
    ModelConfig(name="model-b", var_name="model_b"),
]
```
