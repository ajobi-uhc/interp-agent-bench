"""Orchestrate full Modal GPU deployment: image -> sandbox -> tunnel."""

import os
from pathlib import Path

from .config.parser import load_config
from .gpu import ModalClient, ModalDeploymentInfo


class Deployment:
    """Represents a deployed Modal Sandbox experiment."""

    def __init__(self, deployment_info: ModalDeploymentInfo, client: ModalClient):
        self.sandbox_id = deployment_info.sandbox_id
        self.jupyter_url = deployment_info.jupyter_url
        self.jupyter_port = deployment_info.jupyter_port
        self.jupyter_token = deployment_info.jupyter_token
        self.status = deployment_info.status
        self._sandbox = deployment_info.sandbox
        self._client = client  # Keep reference to the client that created it

    def close(self):
        """Terminate the sandbox and close tunnel."""
        self._client.terminate_sandbox(self.sandbox_id)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def deploy_experiment(
    config_path: Path | str,
    **kwargs,  # Ignore RunPod-specific args for backwards compatibility
) -> Deployment:
    """
    Deploy complete GPU experiment to Modal from config.

    Steps:
    1. Load config
    2. Build Modal Image with dependencies
    3. Create Modal Sandbox with GPU + Jupyter
    4. Return deployment info with local tunnel URL

    Args:
        config_path: Path to YAML config
        **kwargs: Ignored (for backwards compatibility with RunPod code)

    Returns:
        Deployment object with connection info
    """
    config_path = Path(config_path)
    print(f"🚀 Deploying experiment to Modal from: {config_path}")
    print("=" * 70)

    # 1. Load config
    config = load_config(config_path)
    print(f"📋 Experiment: {config.name}")
    print(f"   Models: {[m.name for m in config.models]}")
    print(f"   GPU: {config.gpu.gpu_type}")
    print()

    # 2. Initialize Modal client
    client = ModalClient()

    # 3. Build Modal image
    print("🔨 Building Modal image...")
    image = client.build_image(
        image_config=config.image,
        models=config.models,
        gpu_config=config.gpu,
    )
    print(f"✅ Image ready\n")

    # 4. Deploy Sandbox with Jupyter + GPU
    print("☁️  Creating Modal Sandbox...")
    deployment_info = client.create_jupyter_sandbox(
        name=config.name,
        image=image,
        gpu_config=config.gpu,
        experiment_config=config,
    )
    print(f"✅ Sandbox deployed\n")

    # 5. Return deployment
    print("=" * 70)
    print("🎉 Deployment complete!")
    print(f"   Sandbox ID: {deployment_info.sandbox_id}")
    print(f"   Jupyter (local): {deployment_info.jupyter_url}")
    print(f"   Status: {deployment_info.status}")
    print(f"\n   💡 Your MCP server will automatically connect to:")
    print(f"      {deployment_info.jupyter_url}")
    print("=" * 70)

    return Deployment(deployment_info, client)
