# Sandbox Intro

Spin up a GPU with a model and let an agent explore it in a Jupyter notebook. See [Core Concepts](../concepts/overview.md) for how the pieces fit together.

## 1. Configure the sandbox

```python
from src.environment import Sandbox, SandboxConfig, ExecutionMode, ModelConfig

config = SandboxConfig(
    gpu="A100",
    execution_mode=ExecutionMode.NOTEBOOK,
    models=[ModelConfig(name="google/gemma-2-2b-it")],
    python_packages=["torch", "transformers", "accelerate"],
)
```

- `gpu` — A100 has 40GB VRAM, fits models up to ~30B params
- `execution_mode` — NOTEBOOK means agent works in Jupyter on the GPU
- `models` — HuggingFace model IDs to download and load
- `python_packages` — installed in the sandbox

## 2. Start the sandbox

```python
sandbox = Sandbox(config).start()
```

Provisions the GPU on Modal. First run downloads the model (~2 min), subsequent runs use cache.

## 3. Create a workspace

```python
from src.workspace import Workspace

workspace = Workspace(libraries=[])
```

[Workspace](../concepts/workspaces.md) defines custom code the agent can import. Empty for now — later examples add interpretability tools here.

## 4. Create a session

```python
from src.execution import create_notebook_session

session = create_notebook_session(sandbox, workspace)
```

Returns:
- `session.mcp_config` — config for agent to connect to the notebook
- `session.jupyter_url` — open this to watch the agent work
- `session.model_info_text` — model details to include in agent prompt

## 5. Run the agent

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

sandbox.terminate()
```

The notebook saves to `./outputs/` as the agent works.

## Full example

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
    session = create_notebook_session(sandbox, Workspace(libraries=[]))

    task = (example_dir / "task.md").read_text()
    prompt = f"{session.model_info_text}\n\n{task}"

    try:
        async for msg in run_agent(
            prompt=prompt,
            mcp_config=session.mcp_config,
            provider="claude"
        ):
            pass
        print(f"Notebook: {session.jupyter_url}")
    finally:
        sandbox.terminate()

if __name__ == "__main__":
    asyncio.run(main())
```

```bash
cd experiments/sandbox-intro && python main.py
```
