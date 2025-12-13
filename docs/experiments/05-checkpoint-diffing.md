# Checkpoint Diffing

Compare two model checkpoints (Gemini 2.0 vs 2.5 Flash) using SAE-based analysis to find behavioral differences.
[Notebook](https://github.com/ajobi-uhc/seer/blob/main/example_runs/checkpoint_diffing.ipynb) produced by this experiment

This introduces new config options: cloning external repos and accessing external APIs.

## New concepts

### Cloning external repos

```python
from src.environment import RepoConfig

config = SandboxConfig(
    repos=[RepoConfig(url="nickjiang2378/interp_embed")],
    # ...
)
```

The repo is cloned to `/workspace/interp_embed` in the sandbox. The agent can import from it.

### External API access

```python
config = SandboxConfig(
    secrets=["GEMINI_API_KEY", "OPENAI_KEY", "OPENROUTER_API_KEY", "HF_TOKEN"],
    # ...
)
```

Secrets are Modal secrets you've configured. They're available as environment variables in the sandbox.

### Longer timeout

```python
config = SandboxConfig(
    timeout=7200,  # 2 hours (default is 1 hour)
    # ...
)
```

SAE encoding is slow — this experiment can take 1-2 hours.

## Setup

```python
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
```

## Full example

```python
import asyncio
from pathlib import Path
from src.environment import Sandbox, SandboxConfig, ExecutionMode, RepoConfig
from src.workspace import Workspace, Library
from src.execution import create_notebook_session
from src.harness import run_agent

async def main():
    example_dir = Path(__file__).parent

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
cd experiments/checkpoint-diffing && python main.py
```

## What the agent does

See `experiments/checkpoint-diffing/task.md` for the full protocol:

1. Generate prompts designed to reveal behavioral differences
2. Collect responses from both Gemini versions via OpenRouter
3. Encode responses using SAE (Llama 3.1 8B SAE with 65k features)
4. Diff feature activations to find what changed between versions
5. Analyze top differentiating features with examples
