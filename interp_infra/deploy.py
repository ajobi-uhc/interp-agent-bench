"""Orchestrate full Modal GPU deployment: 3-stage design (Environment -> Execution -> Harness)."""

import os
from pathlib import Path

from .config.parser import load_config
from .environment import setup_environment
from .execution import setup_execution


class Deployment:
    """Represents a deployed experiment (backwards compatibility wrapper)."""

    def __init__(self, env_handle, exec_handle):
        self.sandbox_id = env_handle.sandbox_id
        self.jupyter_url = env_handle.jupyter_url
        self.jupyter_port = env_handle.jupyter_port
        self.jupyter_token = env_handle.jupyter_token
        self.status = env_handle.status

        # Handle both NotebookHandle and dict for exec_handle
        if hasattr(exec_handle, 'session_id'):
            self.session_id = exec_handle.session_id
        elif isinstance(exec_handle, dict):
            self.session_id = exec_handle.get('session_id')
        else:
            self.session_id = None

        self._env_handle = env_handle

    def close(self):
        """Terminate the sandbox and close tunnel."""
        self._env_handle._client.terminate_sandbox(self.sandbox_id)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def deploy_experiment(
    config_path: Path | str,
    **kwargs,  # Ignore RunPod-specific args for backwards compatibility
) -> Deployment:
    """
    Deploy complete GPU experiment using 3-stage design.

    Stages:
    1. Environment: Create sandbox, download models, start Jupyter
    2. Execution: Setup interaction mode (notebook session, MCP, etc.)
    3. Harness: (Handled by caller - run_agent.py)

    Args:
        config_path: Path to YAML config
        **kwargs: Ignored (for backwards compatibility)

    Returns:
        Deployment object with connection info
    """
    config_path = Path(config_path)
    print(f"🚀 Deploying experiment to Modal from: {config_path}")
    print("=" * 70)

    # Load config
    config = load_config(config_path)
    print(f"📋 Experiment: {config.name}")
    if config.environment.models:
        print(f"   Models: {len(config.environment.models)} to load")
    else:
        print(f"   Models: None (API-only)")
    if config.environment.gpu:
        print(f"   GPU: {config.environment.gpu.gpu_type}")
    else:
        print(f"   GPU: None (CPU-only)")
    print()

    # Stage 1: Environment setup
    print("📦 Stage 1: Setting up environment...")
    print("🔨 Building Modal image...")
    env_handle = setup_environment(config)
    print(f"✅ Environment ready\n")

    # Stage 2: Execution setup
    print("⚙️  Stage 2: Setting up execution...")
    exec_handle = setup_execution(env_handle, config)
    print(f"✅ Execution ready\n")

    # Stage 3: Harness (handled by caller)
    print("=" * 70)
    print("🎉 Deployment complete!")
    print(f"   Sandbox ID: {env_handle.sandbox_id}")
    print(f"   Jupyter URL: {env_handle.jupyter_url}")

    if hasattr(exec_handle, 'session_id'):
        print(f"   Session ID: {exec_handle.session_id}")
        print(f"\n   💡 Agent should use: attach_to_session(session_id='{exec_handle.session_id}')")
        print(f"      Models are already loaded and ready!")

    print("=" * 70)

    return Deployment(env_handle, exec_handle)
