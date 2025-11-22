"""Notebook execution - Jupyter session setup and kernel initialization."""

import requests
from typing import Optional
from dataclasses import dataclass


@dataclass
class NotebookHandle:
    """Handle to a notebook execution environment."""
    session_id: str
    jupyter_url: str
    mcp_config: dict


def setup_notebook_session(env_handle, experiment_config) -> str:
    """
    Pre-warm a Jupyter kernel by executing initialization code.

    This starts a session and runs setup_pipeline to load models & skills
    into the kernel namespace.

    Args:
        env_handle: EnvironmentHandle from Stage 1
        experiment_config: Experiment configuration (passed to avoid re-parsing)

    Returns:
        session_id of the pre-warmed session

    Raises:
        RuntimeError: If initialization fails
        TimeoutError: If warmup takes too long
    """
    jupyter_url = env_handle.jupyter_url
    experiment_name = experiment_config.name

    print(f"Pre-warming kernel...")

    # 1. Start a fresh kernel session
    try:
        response = requests.post(
            f"{jupyter_url}/api/scribe/start",
            json={"experiment_name": experiment_name},
            timeout=30,
        )
        response.raise_for_status()
        session_id = response.json()["session_id"]
        print(f"   Started warmup session: {session_id}")
    except Exception as e:
        raise RuntimeError(f"Failed to start warmup session: {e}")

    # 2. Explicitly execute initialization code in the kernel
    init_code = """
import os, sys, base64, traceback
import time

# Ensure paths are set
sys.path.insert(0, "/root")

_start_time = time.time()

print("Loading experiment config...")
from interp_infra.config.schema import ExperimentConfig
config_json = base64.b64decode(os.environ["EXPERIMENT_CONFIG_B64"]).decode('utf-8')
config = ExperimentConfig.model_validate_json(config_json)
print(f"  Experiment: {config.name}")
if config.environment.models:
    print(f"  Models: {len(config.environment.models)}")
if config.execution.skills:
    print(f"  Skills: {len(config.execution.skills)}")
print()

# Run setup pipeline
from interp_infra.execution.kernel_setup import create_namespace
namespace = create_namespace(config)

# Inject into globals
globals().update(namespace)

_total_time = time.time() - _start_time
print(f"Total initialization time: {_total_time:.1f}s")
"""

    print(f"   Executing initialization code in kernel...")

    try:
        response = requests.post(
            f"{jupyter_url}/api/scribe/exec",
            json={
                "session_id": session_id,
                "code": init_code,
                "hidden": False,
            },
            timeout=600,  # 10 minute timeout for model loading
        )
        response.raise_for_status()
        result = response.json()

        # Print all output from the kernel
        for output in result.get("outputs", []):
            output_type = output.get("output_type", output.get("type"))

            if output_type == "stream":
                text = output.get("text", "")
                if text:
                    for line in text.rstrip().split('\n'):
                        print(f"   {line}")

            elif output_type == "error":
                ename = output.get("ename", "Error")
                evalue = output.get("evalue", "")
                traceback_lines = output.get("traceback", [])
                print(f"   Error: {ename}: {evalue}")
                for line in traceback_lines:
                    # Strip ANSI codes from traceback
                    clean_line = line.replace('\x1b[0m', '').replace('\x1b[1m', '').replace('\x1b[31m', '')
                    print(f"   {clean_line}")
                raise RuntimeError(f"Initialization failed: {ename}: {evalue}")

        print(f"Models constructed and ready")
        return session_id

    except requests.exceptions.Timeout:
        raise TimeoutError(f"Initialization timed out after 10 minutes")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to execute initialization: {e}")


def setup_notebook_execution(env_handle, config) -> NotebookHandle:
    """
    Stage 2: Setup notebook execution interface.

    Creates a Jupyter session and loads models/skills into kernel.

    Args:
        env_handle: EnvironmentHandle from Stage 1
        config: ExperimentConfig

    Returns:
        NotebookHandle with session_id and MCP config
    """
    from pathlib import Path

    # Create pre-warmed session
    session_id = setup_notebook_session(env_handle, config)

    # Build MCP config for agent
    venv_python = Path.cwd() / ".venv" / "bin" / "python"
    mcp_config = {
        "notebooks": {
            "type": "stdio",
            "command": str(venv_python),
            "args": ["-m", "scribe.notebook.notebook_mcp_server"],
            "env": {
                "SCRIBE_URL": env_handle.jupyter_url
            }
        }
    }

    return NotebookHandle(
        session_id=session_id,
        jupyter_url=env_handle.jupyter_url,
        mcp_config=mcp_config
    )
