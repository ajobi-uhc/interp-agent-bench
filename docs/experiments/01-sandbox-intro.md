# Tutorial: Your First Agent Investigation

Spin up a GPU, load a model, let an agent explore it in a notebook.

**What you'll build:** A script that provisions an A100, loads Gemma 2B, and lets an agent investigate the model.

**Time:** ~5 min (+ ~2 min first-time model download)

## Prerequisites

- Modal account with token configured (`uv run modal token new`)
- `ANTHROPIC_API_KEY` in your `.env` file
- Repo cloned and `uv sync` completed

## Step 1: Create the sandbox configuration

Create `experiments/my-first-investigation/main.py`:

```python
import asyncio
from pathlib import Path

from src.environment import Sandbox, SandboxConfig, ExecutionMode, ModelConfig

config = SandboxConfig(
    gpu="A100",
    execution_mode=ExecutionMode.NOTEBOOK,
    models=[ModelConfig(name="google/gemma-2-2b-it")],
    python_packages=["torch", "transformers", "accelerate"],
    secrets=["HF_TOKEN"],
)
```

- **`gpu="A100"`** - 40GB VRAM, fits models up to ~30B params
- **`execution_mode=ExecutionMode.NOTEBOOK`** - Agent works in Jupyter on the GPU. You can watch it run.
- **`models`** - HuggingFace model IDs. Cached in Modal volume after first download.
- **`python_packages`** - Installed in the sandbox
- **`secrets`** - Modal secrets to mount. Create `HF_TOKEN` at [Modal Secrets](https://modal.com/secrets) with your HuggingFace token.

## Step 2: Start the sandbox

```python
sandbox = Sandbox(config).start()
```

This provisions the GPU (~30s), downloads the model if needed, starts Jupyter, and loads the model into the kernel.

Watch progress in the [Modal dashboard](https://modal.com/apps).

## Step 3: Create a notebook session

```python
from src.workspace import Workspace
from src.execution import create_notebook_session

workspace = Workspace(libraries=[])
session = create_notebook_session(sandbox, workspace)
```

**Workspace** defines code the agent can import. Empty for now, we add tools in later tutorials.

**Session** connects to the remote notebook:
- `session.jupyter_url` - Watch the agent work in browser
- `session.mcp_config` - Config for agent to execute cells
- `session.model_info_text` - Tells agent what models are loaded

## Step 4: Define the task

Create `experiments/my-first-investigation/task.md`:

```markdown
# Task: Explore the Model

You have a Jupyter notebook with a language model loaded.

## Part 1: Inspect the Model
Look at the model architecture: layers, hidden dimensions, parameter count.

## Part 2: Generate Text
Try a few prompts and observe the model's behavior.

## Part 3: Explore Tokenization
Tokenize some strings, check the token IDs, decode them back.

Summarize what you learn.
```

## Step 5: Run the agent

```python
from src.harness import run_agent

async def main():
    example_dir = Path(__file__).parent

    # ... config and sandbox setup from above ...

    task = (example_dir / "task.md").read_text()
    prompt = f"{session.model_info_text}\n\n{task}"

    try:
        async for msg in run_agent(
            prompt=prompt,
            mcp_config=session.mcp_config,
            provider="claude"
        ):
            pass

        print(f"\n✓ View notebook: {session.jupyter_url}/tree/notebooks")
    finally:
        sandbox.terminate()

asyncio.run(main())
```

- `model_info_text` tells the agent `model` and `tokenizer` are pre-loaded
- `run_agent()` streams as the agent writes and executes cells
- Notebook syncs to `./outputs/` as it runs

## Full example

```python
# experiments/my-first-investigation/main.py
import asyncio
from pathlib import Path

from src.environment import Sandbox, SandboxConfig, ExecutionMode, ModelConfig
from src.workspace import Workspace
from src.execution import create_notebook_session
from src.harness import run_agent


async def main():
    example_dir = Path(__file__).parent

    config = SandboxConfig(
        gpu="A100",
        execution_mode=ExecutionMode.NOTEBOOK,
        models=[ModelConfig(name="google/gemma-2-2b-it")],
        python_packages=["torch", "transformers", "accelerate"],
        secrets=["HF_TOKEN"],
    )
    sandbox = Sandbox(config).start()

    workspace = Workspace(libraries=[])
    session = create_notebook_session(sandbox, workspace)

    task = (example_dir / "task.md").read_text()
    prompt = f"{session.model_info_text}\n\n{task}"

    try:
        async for msg in run_agent(
            prompt=prompt,
            mcp_config=session.mcp_config,
            provider="claude"
        ):
            pass

        print(f"\n✓ View notebook: {session.jupyter_url}/tree/notebooks")
    finally:
        sandbox.terminate()


if __name__ == "__main__":
    asyncio.run(main())
```

```bash
cd experiments/my-first-investigation
python main.py
```

## What to expect

1. **Terminal** - sandbox provisioning and model loading
2. **Modal dashboard** - running GPU under "Apps"
3. **Jupyter URL** - watch the agent write code live
4. **`./outputs/`** - notebook file, synced as agent works

## Costs

- A100: ~$1-2/hour
- Typical run: 5-15 min
- Model downloads cached

## Next steps

- [Scoped Sandbox](02-scoped-sandbox-intro.md) - Controlled access to specific functions instead of full notebook
- [Hidden Preference](03-hidden-preference.md) - Add interpretability tools, investigate a model with hidden bias
