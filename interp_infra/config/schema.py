"""Configuration schemas for GPU experiments."""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class GPUConfig(BaseModel):
    """GPU deployment configuration for Modal."""
    gpu_type: str = Field(default="A10G", description="GPU type to request")
    gpu_count: int = Field(default=1, description="Number of GPUs")

    # Modal Volume support for model persistence
    use_model_volumes: bool = Field(
        default=False,
        description="Use Modal Volumes for persistent model storage (one volume per model)"
    )


class ImageConfig(BaseModel):
    """Docker image build configuration."""
    base_image: str = Field(
        default="nvidia/cuda:12.1.0-base-ubuntu22.04",
        description="Base Docker image"
    )
    python_version: str = Field(default="3.11", description="Python version")

    # Packages
    system_packages: List[str] = Field(
        default_factory=lambda: ["openssh-server", "git"],
        description="System packages (apt)"
    )
    python_packages: List[str] = Field(
        default_factory=lambda: ["torch", "transformers", "jupyter", "jupyter_client", "matplotlib", "numpy", "pandas"],
        description="Python packages (pip)"
    )

    # Custom setup
    custom_setup_commands: List[str] = Field(
        default_factory=list,
        description="Custom bash commands to run during image build"
    )


class EnvironmentConfig(BaseModel):
    """Environment configuration for experiment setup.

    Environments define how to build the experiment-specific namespace
    that gets injected into the Jupyter kernel before the agent's first cell.
    """
    name: str = Field(..., description="Environment identifier (e.g., 'model', 'api_access')")
    model_id: Optional[str] = Field(default=None, description="Primary model identifier (optional)")
    extra: Dict[str, Any] = Field(
        default_factory=dict,
        description="Environment-specific parameters (flexible schema per environment)"
    )


class DeploymentConfig(BaseModel):
    """Deployment configuration.

    Docker images must be pushed to a registry (Docker Hub, GHCR, etc.)
    for GPU providers (RunPod, Modal) to access them.
    """
    docker_registry: Optional[str] = Field(
        default=None,
        description="Docker registry username (e.g., 'myusername' for Docker Hub 'myusername/image:tag')"
    )
    ssh_key_path: Optional[str] = Field(
        default="~/.ssh/id_rsa",
        description="Path to SSH private key for RunPod access"
    )
    ssh_password: Optional[str] = Field(
        default=None,
        description="SSH password (alternative to key-based auth)"
    )


class ExperimentConfig(BaseModel):
    """Complete experiment configuration."""
    name: str = Field(..., description="Experiment name")
    task: str = Field(..., description="Task description for the agent")

    # Environment setup
    environment: EnvironmentConfig = Field(..., description="Environment configuration (models, APIs, etc.)")

    # Skills (optional interpretability techniques)
    skills: List[str] = Field(
        default_factory=list,
        description="Skills to load (e.g., ['steering-vectors', 'sae-latents'])"
    )

    gpu: Optional[GPUConfig] = Field(default=None, description="GPU configuration (None for CPU-only)")
    image: ImageConfig = Field(default_factory=ImageConfig, description="Docker image configuration")
    deployment: DeploymentConfig = Field(default_factory=DeploymentConfig, description="Deployment configuration")

    # Infrastructure setup (cloning repos, etc.)
    github_repos: List[str] = Field(
        default_factory=list,
        description="GitHub repos to clone into workspace"
    )
