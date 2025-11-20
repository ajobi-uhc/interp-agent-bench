"""
End-to-end orchestrator for experiment setup.

This orchestrator runs in the parent process and coordinates:
1. Image building
2. Prewarm (download models, clone repos)
3. Sandbox creation
4. Kernel warmup (via embedded code)

The kernel warmup runs setup_pipeline.create_namespace() inside the sandbox.
"""

from typing import Optional
from pathlib import Path

from interp_infra.config.schema import ExperimentConfig, GPUConfig, ImageConfig
from interp_infra.gpu.modal_client import ModalClient, ModalDeploymentInfo


class ExperimentOrchestrator:
    """
    Orchestrates the complete experiment setup pipeline.

    This runs in the parent process and coordinates all phases:
    - Phase 1: Build Modal image
    - Phase 2: Create sandbox + prewarm
    - Phase 3: Start Jupyter + kernel warmup
    """

    def __init__(self):
        """Initialize orchestrator with Modal client."""
        self.modal_client = ModalClient()

    def setup_experiment(
        self,
        config: ExperimentConfig,
        name: Optional[str] = None
    ) -> ModalDeploymentInfo:
        """
        Set up complete experiment environment.

        Args:
            config: Experiment configuration
            name: Optional sandbox name (defaults to config.name)

        Returns:
            ModalDeploymentInfo with connection details

        Flow:
            1. Build image (system packages, python packages)
            2. Create sandbox with prewarm (download models, clone repos)
            3. Kernel warmup happens automatically inside sandbox
        """
        name = name or config.name

        print("=" * 70)
        print(f"EXPERIMENT ORCHESTRATOR: {name}")
        print("=" * 70)

        # Phase 1: Build Image
        print("\n[Phase 1/3] Building Modal Image")
        print("-" * 70)
        image = self.modal_client.build_image(config.image, config.gpu)
        print("  ✓ Image built")

        # Phase 2 & 3: Create sandbox (includes prewarm + kernel warmup)
        print("\n[Phase 2/3] Creating Sandbox + Prewarm")
        print("-" * 70)
        deployment_info = self.modal_client.create_jupyter_sandbox(
            name=name,
            image=image,
            gpu_config=config.gpu,
            experiment_config=config
        )

        print("\n[Phase 3/3] Kernel Warmup")
        print("-" * 70)
        print("  (Happens automatically inside sandbox via setup_pipeline)")

        print("\n" + "=" * 70)
        print("EXPERIMENT READY")
        print("=" * 70)
        print(f"Jupyter URL: {deployment_info.jupyter_url}")
        print(f"Tunnel URL:  {deployment_info.tunnel_url}")
        print()

        return deployment_info


def setup_experiment(config: ExperimentConfig) -> ModalDeploymentInfo:
    """
    Convenience function to set up an experiment.

    Args:
        config: Experiment configuration

    Returns:
        ModalDeploymentInfo with connection details

    Example:
        >>> from interp_infra.orchestrator import setup_experiment
        >>> config = ExperimentConfig.model_validate_json(config_json)
        >>> deployment = setup_experiment(config)
        >>> print(f"Connect to: {deployment.jupyter_url}")
    """
    orchestrator = ExperimentOrchestrator()
    return orchestrator.setup_experiment(config)
