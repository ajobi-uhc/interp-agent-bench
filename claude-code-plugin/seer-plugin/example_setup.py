#!/usr/bin/env python3
"""
Example setup script that Claude would generate and run.

This demonstrates how the Sena plugin works:
1. Claude writes a script like this
2. Runs it with: uv run python example_setup.py
3. Parses the JSON output
4. Calls attach_to_session() with the results
"""

from src.environment import Sandbox, SandboxConfig, ExecutionMode, ModelConfig
from src.workspace import Workspace, Library
from src.execution import create_notebook_session
import json
from pathlib import Path

def main():
    # Configure sandbox
    config = SandboxConfig(
        execution_mode=ExecutionMode.NOTEBOOK,
        gpu="A100",
        models=[
            ModelConfig(name="google/gemma-2-9b-it")
        ],
        python_packages=[
            "torch",
            "transformers",
            "accelerate",
            "matplotlib",
            "numpy",
            "pandas",
        ],
    )

    print("Starting sandbox...", flush=True)
    sandbox = Sandbox(config).start()

    # Optional: Include helper libraries
    example_dir = Path(__file__).parent.parent / "experiments" / "libraries"
    libraries = []
    if example_dir.exists():
        for lib_file in example_dir.glob("*.py"):
            libraries.append(Library.from_file(str(lib_file)))

    workspace = Workspace(libraries=libraries)

    print("Creating notebook session...", flush=True)
    session = create_notebook_session(sandbox, workspace)

    # Output the connection info as JSON
    result = {
        "session_id": session.session_id,
        "jupyter_url": session.jupyter_url,
        "notebook_path": "./outputs/notebook.ipynb",
        "status": "ready",
    }

    print(json.dumps(result))

if __name__ == "__main__":
    main()
