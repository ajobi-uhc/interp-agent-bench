# Sandbox Intro

**Goal**: Spin up a GPU sandbox with a model and give an agent a Jupyter notebook to explore it

This is the simplest way to run interpretability research - the agent gets full access to a model in a Jupyter notebook environment.

## Sandbox Configuration

First, configure what hardware and models you want:

```python
from src.environment import Sandbox, SandboxConfig, ExecutionMode, ModelConfig

config = SandboxConfig(
    gpu="A100",
    execution_mode=ExecutionMode.NOTEBOOK,
    models=[ModelConfig(name="google/gemma-2-2b-it")],
    python_packages=["torch", "transformers", "accelerate"],
)
```

`SandboxConfig` tells the library what environment to provision:

- `gpu="A100"` - which GPU to use (A100 has 40GB VRAM, good for models under 30B params)
- `execution_mode=ExecutionMode.NOTEBOOK` - agent runs in Jupyter notebook on the GPU
- `models=[...]` - which models to download and load (specified by HuggingFace model ID)
- `python_packages=[...]` - which packages to install (these are the minimum needed to load transformers models)

Start the sandbox:

```python
sandbox = Sandbox(config).start()
```

This provisions the GPU on Modal and downloads the model (first run takes ~2 minutes, then cached).

## Workspace

Next, configure what libraries/tools the agent has access to:

```python
from src.workspace import Workspace

workspace = Workspace(libraries=[])
```

`Workspace` defines custom code you want to provide to the agent:

- `libraries=[]` - empty list means no custom libraries, just standard Python packages
- You could add custom interpretability tools here (we'll see this in later experiments)

## Session

Finally, create a notebook session that connects the agent to the sandbox:

```python
from src.execution import create_notebook_session

session = create_notebook_session(sandbox, workspace)
```

`create_notebook_session` returns:

- `session.mcp_config` - configuration for the agent to connect to Jupyter
- `session.jupyter_url` - URL where you can view the notebook
- `session.model_info_text` - info about loaded models to include in the agent prompt

## Running the Agent

Load your task and run the agent:

```python
from src.harness import run_agent

task = (example_dir / "task.md").read_text()
prompt = f"{session.model_info_text}\n\n{task}"

async for msg in run_agent(
    prompt=prompt,
    mcp_config=session.mcp_config,
    provider="claude"
):
    pass

print(f"\n✓ Jupyter: {session.jupyter_url}")
```

The agent will:

1. Connect to the Jupyter notebook via MCP
2. Load and explore the model
3. Run experiments based on your task
4. Save everything to the notebook

Visit `session.jupyter_url` to see the notebook with all the agent's code and outputs.

## Full Example

experiments/sandbox-intro/main.py

```python
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

        print(f"\n✓ Jupyter: {session.jupyter_url}")

    finally:
        sandbox.terminate()

if __name__ == "__main__":
    asyncio.run(main())
```

## Running

```bash
cd experiments/sandbox-intro
python main.py
```

The agent will explore the model and save results to a Jupyter notebook.
