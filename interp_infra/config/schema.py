"""Configuration schemas for GPU experiments."""

from typing import Optional, List
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Model loading configuration."""
    name: str = Field(..., description="HuggingFace model identifier")
    device: str = Field(default="cuda", description="Device to load model on")
    dtype: str = Field(default="bfloat16", description="Model dtype (bfloat16, float16, float32)")
    quantization: Optional[str] = Field(default=None, description="Quantization method (4bit, 8bit)")
    trust_remote_code: bool = Field(default=True, description="Trust remote code in model")

    # PEFT adapter support
    is_peft: bool = Field(default=False, description="Whether model is a PEFT adapter")
    base_model: Optional[str] = Field(default=None, description="Base model for PEFT adapters")

    # Custom loading code
    custom_load_code: Optional[str] = Field(
        default=None,
        description="Custom Python code to load model (overrides default loading)"
    )

    # Obfuscation
    obfuscate_name: bool = Field(
        default=False,
        description="Hide model name from agent (for blind evaluations)"
    )


class GPUConfig(BaseModel):
    """GPU deployment configuration."""
    provider: str = Field(default="runpod", description="GPU provider (runpod)")
    gpu_type: str = Field(default="NVIDIA RTX A5000", description="GPU type to request")
    gpu_count: int = Field(default=1, description="Number of GPUs")
    container_disk_size: int = Field(default=50, description="Container disk size in GB")
    volume_size: int = Field(default=0, description="Persistent volume size in GB (0 for no volume)")

    # Docker image
    docker_image: Optional[str] = Field(default=None, description="Pre-built Docker image name")
    build_image: bool = Field(default=True, description="Build image from config vs use existing")


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
        default_factory=lambda: ["torch", "transformers", "jupyter", "jupyter_client", "matplotlib", "numpy"],
        description="Python packages (pip)"
    )

    # Custom setup
    custom_setup_commands: List[str] = Field(
        default_factory=list,
        description="Custom bash commands to run during image build"
    )

    # Model preloading
    preload_models: bool = Field(
        default=False,
        description="Preload models into Docker image (faster startup but larger image)"
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

    models: List[ModelConfig] = Field(..., description="Models to load")
    gpu: GPUConfig = Field(default_factory=GPUConfig, description="GPU configuration")
    image: ImageConfig = Field(default_factory=ImageConfig, description="Docker image configuration")
    deployment: DeploymentConfig = Field(default_factory=DeploymentConfig, description="Deployment configuration")

    # Environment setup
    github_repos: List[str] = Field(
        default_factory=list,
        description="GitHub repos to clone into workspace"
    )
    startup_code: Optional[str] = Field(
        default=None,
        description="Python code to run on startup (after models loaded)"
    )
