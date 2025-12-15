# Checkpoint Diffing

What changed between Gemini 2.0 and 2.5 Flash? Use SAE features to find out.

[Example notebook](https://github.com/ajobi-uhc/seer/blob/main/example_runs/checkpoint_diffing.ipynb)

This uses data-centric SAE techniques from [Jiang et al.](https://www.lesswrong.com/posts/a4EDinzAYtRwpNmx9/towards-data-centric-interpretability-with-sparse) to diff model checkpoints. The idea: generate responses from both model versions, encode them with an SAE, diff the feature activations. Features that activate differently reveal behavioral changes between checkpoints.

## New concepts

This experiment introduces repo cloning, external API access, and longer timeouts.

### Clone external repos

```python
from src.environment import RepoConfig

config = SandboxConfig(
    repos=[RepoConfig(url="nickjiang2378/interp_embed")],
    ...
)
```

Cloned to `/workspace/interp_embed`. The agent can import from it directly.

### External API access

```python
config = SandboxConfig(
    secrets=["GEMINI_API_KEY", "OPENROUTER_API_KEY", "HF_TOKEN"],
    ...
)
```

Secrets are [Modal secrets](https://modal.com/secrets). Available as environment variables in the sandbox.

### Longer timeout

```python
config = SandboxConfig(
    timeout=7200,  # 2 hours
    ...
)
```

SAE encoding is slow. Default 1 hour isn't enough for this experiment.

## Setup

```python
from src.environment import Sandbox, SandboxConfig, ExecutionMode, RepoConfig
from src.workspace import Workspace, Library
from src.execution import create_notebook_session

config = SandboxConfig(
    gpu="A100",
    execution_mode=ExecutionMode.NOTEBOOK,
    repos=[RepoConfig(url="nickjiang2378/interp_embed")],
    system_packages=["git"],
    python_packages=[
        "torch", "transformers", "accelerate", "pandas", "numpy", "scipy",
        "google-generativeai", "datasets", "matplotlib", "seaborn",
        "sae-lens", "transformer-lens", "huggingface-hub", "openai",
    ],
    secrets=["GEMINI_API_KEY", "OPENROUTER_API_KEY", "HF_TOKEN"],
    timeout=7200,
)
sandbox = Sandbox(config).start()

workspace = Workspace(libraries=[
    Library.from_file(example_dir / "openrouter_client.py")
])

session = create_notebook_session(sandbox, workspace)
```

The `openrouter_client.py` library provides a simple interface to call both Gemini versions via OpenRouter.

## What the agent does

The task prompt (`experiments/checkpoint-diffing/task.md`) guides the agent through:

1. **Generate prompts** designed to reveal behavioral differences
2. **Collect responses** from both Gemini versions via OpenRouter
3. **Encode with SAE** (Llama 3.1 8B SAE, 65k features)
4. **Diff feature activations** between versions
5. **Analyze top differentiating features** - what changed?

## Running it

```bash
cd experiments/checkpoint-diffing
python main.py
```

Takes 1-2 hours. Requires A100 for SAE encoding.

## Next steps

- [Petri Harness](06-petri-harness.md) - Hackable Petri for auditing model behaviors with blackbox or whitebox access
