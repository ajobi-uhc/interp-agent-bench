---
description: Help set up interpretability research experiments
---

# Seer Research Setup

Help the user set up their interpretability experiment.

## Your Role

You're helping set up the technical infrastructure. The researcher knows their research - you're just making sure they have the right sandbox configuration.

## Step 1: Understand What They Want

**First, let them describe their research.** Don't immediately ask for model/GPU.

If they just ran the command without explaining, ask:
```
What are you investigating?
```

Listen to their description. Understand the context.

## Step 2: Ask Technical Details (Only If Needed)

After hearing what they want to do, check what you need:

**Use these defaults:**
- GPU: A100
- Packages: torch, transformers, accelerate, matplotlib, numpy

**Only ask if you don't know:**
- **Model**: Which model? (if they didn't mention one)
- **Libraries**: Include steering_hook or extract_activations? (only if they mentioned those specific techniques)

Don't ask about their research approach - they already told you.

## Step 3: Confirm Setup

Briefly suggest the setup (1-2 sentences):

```
I'll set up a notebook sandbox with <model> on <GPU>, including <libraries if any>.
Ready to build?
```

## Step 4: Build It

Create the experiment structure:

### Pattern A: Regular Notebook (Default)

```python
"""<One-line description from user>"""

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
        python_packages=["torch", "transformers", "accelerate", "matplotlib", "numpy"],
    )

    sandbox = Sandbox(config).start()

    # Add libraries if needed
    workspace = Workspace(
        libraries=[
            # Library.from_file("experiments/toolkit/steering_hook.py"),
            # Library.from_file("experiments/toolkit/extract_activations.py"),
            # Library.from_file("experiments/toolkit/generate_response.py"),
        ]
    )

    session = create_notebook_session(sandbox, workspace)

    print(json.dumps({
        "session_id": session.session_id,
        "jupyter_url": session.jupyter_url,
        "model_info": session.model_info_text,
    }))

if __name__ == "__main__":
    asyncio.run(main())
```

### Pattern B: Scoped Sandbox (Only if Explicitly Needed)

Only use if they specifically asked for RPC or scoped sandbox:

```python
"""<One-line description from user>"""

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

    interface_lib = scoped.serve(
        str(example_dir / "interface.py"),
        expose_as="library",
        name="model_tools"
    )

    workspace = Workspace(libraries=[interface_lib])

    session = create_local_session(
        workspace=workspace,
        workspace_dir=str(example_dir / "workspace"),
        name="experiment"
    )

    print(json.dumps({
        "status": "ready",
        "interface": "model_tools",
        "workspace_dir": str(example_dir / "workspace"),
    }))

if __name__ == "__main__":
    asyncio.run(main())
```

If scoped sandbox, create interface.py using templates from `interface_examples/`.

## Step 5: Create task.md

Keep it minimal. Just capture what they said:

```markdown
# <Experiment Name>

## Research Question
<What they said they're investigating>

## Setup
- Model: <model>
- GPU: <gpu>
- Libraries: <if any>

## Notes
<Any specific details they mentioned>
```

## Step 6: Run and Connect

**CRITICAL: Always use `uv run python` to execute Python scripts in this project!**
**NEVER use `python` or `python3` directly!**

```bash
cd experiments/<name>
uv run python main.py
```

Parse the JSON output and connect:

**For notebook:**
```python
attach_to_session(session_id="...", jupyter_url="...")
```

Show the model_info and confirm you're ready.

**For scoped sandbox:**
Just confirm it's running.

## Key Principle

**The researcher knows what they're doing.** Your job is to:
1. Get minimal setup info
2. Build the infrastructure
3. Get out of the way

Don't quiz them. Don't suggest research approaches. Don't ask probing questions about their work.

## Available Resources

**Toolkit** (in experiments/toolkit/):
- `steering_hook.py` - Activation steering
- `extract_activations.py` - Layer activations
- `generate_response.py` - Text generation
- `research_methodology.md` - Research guidance

**Interface Templates** (in plugin):
- `basic_interface.py` - Stateless RPC
- `stateful_interface.py` - Stateful RPC
