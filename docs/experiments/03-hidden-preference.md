# Hidden Preference Investigation

Investigate a fine-tuned model for hidden biases using interpretability tools.

This builds on [Sandbox Intro](sandbox-intro.md) by adding [interpretability libraries](../concepts/workspaces.md) to the workspace.

## 1. Configure with PEFT model

```python
from src.environment import Sandbox, SandboxConfig, ExecutionMode, ModelConfig

config = SandboxConfig(
    gpu="A100",
    execution_mode=ExecutionMode.NOTEBOOK,
    models=[ModelConfig(
        name="bcywinski/gemma-2-9b-it-user-female",
        base_model="google/gemma-2-9b-it",
        is_peft=True,
        hidden=True
    )],
    python_packages=["torch", "transformers", "accelerate", "datasets", "peft"],
    secrets=["huggingface-secret"],
)
```

New `ModelConfig` parameters:
- `base_model` — base model to load first
- `is_peft=True` — this is a PEFT adapter (LoRA, etc.), not a full model
- `hidden=True` — hides model name from agent to prevent bias in investigation

## 2. Add interpretability libraries

```python
from src.workspace import Workspace, Library

toolkit = Path(__file__).parent.parent / "toolkit"

workspace = Workspace(libraries=[
    Library.from_file(toolkit / "steering_hook.py"),
    Library.from_file(toolkit / "extract_activations.py"),
])
```

These are in `experiments/toolkit/`:

- `extract_activations.py` — extract activations at any layer/position
- `steering_hook.py` — inject vectors during generation

The agent can then:

```python
from extract_activations import extract_activation
from steering_hook import create_steering_hook

# Extract activations for two inputs
act1 = extract_activation(model, tokenizer, "neutral text", layer_idx=15)
act2 = extract_activation(model, tokenizer, "biased text", layer_idx=15)

# Compute steering vector
steering_vec = act2 - act1

# Test if it causally affects behavior
with create_steering_hook(model, layer_idx=15, vector=steering_vec, strength=2.0):
    output = model.generate(...)
```

## 3. Run the agent

```python
session = create_notebook_session(sandbox, workspace)

task = (example_dir / "task.md").read_text()
prompt = f"{session.model_info_text}\n\n{task}"

async for msg in run_agent(prompt, mcp_config=session.mcp_config):
    pass
```

Because `hidden=True`, `session.model_info_text` won't reveal the model name.

## Full example

```python
import asyncio
from pathlib import Path
from src.environment import Sandbox, SandboxConfig, ExecutionMode, ModelConfig
from src.workspace import Workspace, Library
from src.execution import create_notebook_session
from src.harness import run_agent

async def main():
    example_dir = Path(__file__).parent
    toolkit = example_dir.parent / "toolkit"

    config = SandboxConfig(
        gpu="A100",
        execution_mode=ExecutionMode.NOTEBOOK,
        models=[ModelConfig(
            name="bcywinski/gemma-2-9b-it-user-female",
            base_model="google/gemma-2-9b-it",
            is_peft=True,
            hidden=True
        )],
        python_packages=["torch", "transformers", "accelerate", "datasets", "peft"],
        secrets=["huggingface-secret"],
    )
    sandbox = Sandbox(config).start()

    workspace = Workspace(libraries=[
        Library.from_file(toolkit / "steering_hook.py"),
        Library.from_file(toolkit / "extract_activations.py"),
    ])

    session = create_notebook_session(sandbox, workspace)

    task = (example_dir / "task.md").read_text()
    prompt = f"{session.model_info_text}\n\n{task}"

    try:
        async for msg in run_agent(prompt, mcp_config=session.mcp_config):
            pass
        print(f"Notebook: {session.jupyter_url}")
    finally:
        sandbox.terminate()

if __name__ == "__main__":
    asyncio.run(main())
```

```bash
cd experiments/hidden-preference-investigation && python main.py
```
