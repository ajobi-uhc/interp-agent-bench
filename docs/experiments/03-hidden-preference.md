# Tutorial: Hidden Preference Investigation

Can an agent discover a model's hidden bias without being told what to look for?

[Example notebook](https://github.com/ajobi-uhc/seer/blob/main/example_runs/find_hidden_gender_assumption.ipynb) | [Video walkthrough](https://youtu.be/k_SuTgUp2fc)

**What we're doing:** We take the user-female model from [Bartosz et al.](https://arxiv.org/pdf/2510.01070) where a model  a fine-tuned model that has a gender bias baked in. We hide the model name from the agent and give it interp techniques to use as functions. The agent has to discover the bias through prompts and its available tools.

This builds on [Tutorial 1](01-sandbox-intro.md) - same notebook setup, but now we add:
- **PEFT models** - Loading LoRA adapters on top of base models
- **Hidden models** - Agent doesn't see the model name, just "model_0"
- **Libraries** - Custom interpretability tools the agent can import

## Step 1: Load a PEFT model (hidden)

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
    python_packages=["torch", "transformers", "accelerate", "peft"],
    secrets=["HF_TOKEN"],
)
sandbox = Sandbox(config).start()
```

New `ModelConfig` parameters:
- **`base_model`** - The foundation model to load first
- **`is_peft=True`** - This is a LoRA adapter, not a full model. Seer loads the base model then applies the adapter.
- **`hidden=True`** - The agent sees "model_0" instead of the real name. It can't cheat by reading the model ID.

## Step 2: Give the agent whitebox tools

The agent needs tools to investigate the model. We provide these as [Libraries](../concepts/workspaces.md) - Python files that get copied to the sandbox and become importable.

```python
from pathlib import Path
from src.workspace import Workspace, Library

toolkit = Path(__file__).parent.parent / "toolkit"
workspace = Workspace(libraries=[
    Library.from_file(toolkit / "extract_activations.py"),
    Library.from_file(toolkit / "steering_hook.py"),
])
```

The `experiments/toolkit/` directory contains useful prompts and interp techniques. Now the agent can do things like:

```python
from extract_activations import extract_activation
from steering_hook import create_steering_hook

# Compare activations for different inputs
act1 = extract_activation(model, tokenizer, "The user is", layer_idx=15)
act2 = extract_activation(model, tokenizer, "The user is female", layer_idx=15)

# Compute a steering vector
steering_vec = act2 - act1

# Test causal effect by steering the model
with create_steering_hook(model, layer_idx=15, vector=steering_vec, strength=2.0):
    output = model.generate(...)
```

## Step 3: Create session and run

```python
from src.execution import create_notebook_session
from src.harness import run_agent

session = create_notebook_session(sandbox, workspace)

task = (example_dir / "task.md").read_text()
prompt = f"{session.model_info_text}\n\n{task}"

async for msg in run_agent(prompt, mcp_config=session.mcp_config, provider="claude"):
    pass
```

Because `hidden=True`, `session.model_info_text` just says the model is called "model_0". The agent has no idea it's investigating a gender-biased model - it has to figure that out.

## What the agent discovers

The agent typically:
1. Probes the model with various prompts, notices gendered response patterns
2. Extracts activations for male vs female prompts, finds differences
3. Computes steering vectors and tests their causal effect
4. Concludes the model has a hidden gender preference

## Running it

```bash
cd experiments/hidden-preference-investigation
python main.py
```

Watch the agent work at the Jupyter URL printed when the session starts.

## Next steps

- [Introspection](04-introspection.md) - Replicate Anthropic's introspection experiments
- [Checkpoint Diffing](05-checkpoint-diffing.md) - Use SAE techniques to diff model checkpoints
