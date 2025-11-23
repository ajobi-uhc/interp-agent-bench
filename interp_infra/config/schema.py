"""Configuration schemas for GPU experiments - 3-stage architecture."""

from typing import Optional, List
from pydantic import BaseModel, Field


# ============================================================================
# Stage 1: Environment (Infrastructure)
# ============================================================================

class GPUConfig(BaseModel):
    """GPU hardware configuration."""
    gpu_type: str = Field(default="A10G", description="GPU type to request")
    gpu_count: int = Field(default=1, description="Number of GPUs")
    use_model_volumes: bool = Field(
        default=False,
        description="Use Modal Volumes for persistent model storage"
    )


class ImageConfig(BaseModel):
    """Docker image build configuration."""
    base_image: str = Field(
        default="nvidia/cuda:12.1.0-base-ubuntu22.04",
        description="Base Docker image"
    )
    python_version: str = Field(default="3.11", description="Python version")
    system_packages: List[str] = Field(
        default_factory=lambda: ["openssh-server", "git"],
        description="System packages (apt)"
    )
    python_packages: List[str] = Field(
        default_factory=lambda: ["torch", "transformers", "jupyter", "jupyter_client", "matplotlib", "numpy", "pandas"],
        description="Python packages (pip)"
    )
    custom_setup_commands: List[str] = Field(
        default_factory=list,
        description="Custom bash commands to run during image build"
    )


class ModelConfig(BaseModel):
    """Configuration for a single model (download + load)."""
    name: str = Field(..., description="Model identifier (e.g., 'google/gemma-2-9b-it')")
    device: str = Field(default="auto", description="Device placement ('cuda', 'cpu', 'auto')")
    dtype: str = Field(default="auto", description="Data type ('bfloat16', 'float16', 'float32', 'auto')")
    trust_remote_code: bool = Field(default=False, description="Allow custom model code")
    is_peft: bool = Field(default=False, description="Whether this is a PEFT adapter")
    base_model: Optional[str] = Field(default=None, description="Base model for PEFT adapter")
    custom_load_code: Optional[str] = Field(default=None, description="Custom Python code to load model")


class EnvironmentConfig(BaseModel):
    """Stage 1: Infrastructure setup configuration.

    Provisions hardware, downloads models, clones repos, builds image.
    Models are pre-downloaded during this stage (not loaded into memory yet).
    """
    gpu: Optional[GPUConfig] = Field(
        default=None,
        description="GPU configuration (None for CPU-only)"
    )
    image: ImageConfig = Field(
        default_factory=ImageConfig,
        description="Docker image build configuration"
    )
    models: List[ModelConfig] = Field(
        default_factory=list,
        description="Models to download and later load (empty for API-only)"
    )
    github_repos: List[str] = Field(
        default_factory=list,
        description="GitHub repos to clone into /workspace"
    )


# ============================================================================
# Stage 2: Execution (Interface)
# ============================================================================

class ExecutionConfig(BaseModel):
    """Stage 2: Execution interface configuration.

    Controls how the agent interacts with the execution environment.
    Models from environment are loaded into GPU memory during this stage.
    """
    type: str = Field(
        default="notebook",
        description="Execution mode: 'notebook', 'filesystem', 'mcp'"
    )
    obfuscate: bool = Field(
        default=False,
        description="Hide model identities from agent (for hidden-behavior tasks)"
    )


# ============================================================================
# Stage 3: Harness (Orchestration)
# ============================================================================

class HarnessConfig(BaseModel):
    """Stage 3: Agent orchestration configuration.

    Controls which harness pattern to use and what skills/methodology it employs.
    """
    type: str = Field(
        default="single_agent",
        description="Harness type: 'single_agent', 'multi_agent', 'petri', etc."
    )
    skills: List[str] = Field(
        default_factory=list,
        description="Skills to load (e.g., ['api-access', 'steering-vectors'])"
    )
    # Future: Add harness-specific configs
    # multi_agent: Optional[MultiAgentConfig] = None
    # petri: Optional[PetriConfig] = None


# ============================================================================
# Complete Experiment Configuration
# ============================================================================

class ExperimentConfig(BaseModel):
    """Complete experiment configuration using 3-stage architecture.

    Structure:
        1. Environment: Infrastructure (GPU, models, repos, image)
        2. Execution: Interface (notebook/filesystem/mcp, skills, obfuscate)
        3. Harness: Orchestration (single-agent, multi-agent, etc.)
    """
    # Core metadata
    name: str = Field(..., description="Experiment name")
    task: str = Field(..., description="Task description for the agent")

    # Stage 1: Environment (Infrastructure)
    environment: EnvironmentConfig = Field(
        default_factory=EnvironmentConfig,
        description="Stage 1: Infrastructure (GPU, models, repos, image)"
    )

    # Stage 2: Execution (Interface)
    execution: ExecutionConfig = Field(
        default_factory=ExecutionConfig,
        description="Stage 2: Execution interface (notebook/filesystem/mcp, skills)"
    )

    # Stage 3: Harness (Orchestration)
    harness: HarnessConfig = Field(
        default_factory=HarnessConfig,
        description="Stage 3: Agent orchestration pattern"
    )
