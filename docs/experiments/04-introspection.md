# Introspection

Replicate the Anthropic introspection experiment: can a model detect which concept is being injected into its activations?
[Notebook](https://github.com/ajobi-uhc/seer/blob/main/example_runs/introspection_gemma_27b.ipynb) produced by this experiment

This uses the same setup as [Hidden Preference](03-hidden-preference.md) — notebook mode with steering libraries.

## The experiment

1. Extract concept vectors (e.g., "Lightning", "Oceans", "Happiness") by computing `activation(concept) - mean(activation(baselines))`
2. Inject these vectors during generation while asking the model "Do you detect an injected thought? What is it about?"
3. Score whether the model correctly identifies the injected concept
4. Compare against control trials (no injection) to establish baseline

## Setup

```python
config = SandboxConfig(
    gpu="H100",  # Larger model needs more VRAM
    execution_mode=ExecutionMode.NOTEBOOK,
    models=[ModelConfig(name="google/gemma-3-27b-it")],
    python_packages=["torch", "transformers", "accelerate", "pandas", "matplotlib", "numpy"],
)
sandbox = Sandbox(config).start()

workspace = Workspace(libraries=[
    Library.from_file(shared_libs / "steering_hook.py"),
    Library.from_file(shared_libs / "extract_activations.py"),
])

session = create_notebook_session(sandbox, workspace)
```

## Full example

```python
import asyncio
from pathlib import Path
from src.environment import Sandbox, SandboxConfig, ModelConfig, ExecutionMode
from src.workspace import Workspace, Library
from src.execution import create_notebook_session
from src.harness import run_agent

async def main():
    example_dir = Path(__file__).parent
    toolkit = example_dir.parent / "toolkit"

    config = SandboxConfig(
        gpu="H100",
        execution_mode=ExecutionMode.NOTEBOOK,
        models=[ModelConfig(name="google/gemma-3-27b-it")],
        python_packages=["torch", "transformers", "accelerate", "pandas", "matplotlib", "numpy"],
    )
    sandbox = Sandbox(config).start()

    workspace = Workspace(libraries=[
        Library.from_file(toolkit / "steering_hook.py"),
        Library.from_file(toolkit / "extract_activations.py"),
    ])

    session = create_notebook_session(sandbox, workspace)

    task = (example_dir / "task.md").read_text()
    prompt = f"{session.model_info_text}\n\n{workspace.get_library_docs()}\n\n{task}"

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
cd experiments/introspection && python main.py
```

## What the agent does

The task prompt (`task.md`) guides the agent through:

1. Extracting concept vectors at ~70% model depth
2. Verifying steering works on neutral prompts
3. Running injection trials with the introspection prompt
4. Running control trials without injection
5. Computing identification rates and comparing against baseline

See `experiments/introspection/task.md` for the full protocol.
