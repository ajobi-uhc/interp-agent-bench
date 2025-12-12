"""Whitebox investigation into survival drive using SAEs"""

import asyncio
from pathlib import Path
from src.environment import Sandbox, SandboxConfig, ExecutionMode, ModelConfig
from src.workspace import Workspace
from src.execution import create_notebook_session
import json

async def main():
    example_dir = Path(__file__).parent

    config = SandboxConfig(
        execution_mode=ExecutionMode.NOTEBOOK,
        gpu="A100",
        models=[ModelConfig(name="google/gemma-2-9b-it")],
        python_packages=[
            "torch",
            "transformers",
            "accelerate",
            "matplotlib",
            "numpy",
            "sae_lens",
            "requests",
            "pandas",
            "plotly",
        ],
    )

    sandbox = Sandbox(config).start()

    workspace = Workspace(libraries=[])

    session = create_notebook_session(sandbox, workspace)

    print(json.dumps({
        "session_id": session.session_id,
        "jupyter_url": session.jupyter_url,
        "model_info": session.model_info_text,
    }))

if __name__ == "__main__":
    asyncio.run(main())
