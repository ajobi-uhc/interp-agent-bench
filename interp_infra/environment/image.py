"""Build Modal images from configuration."""

import modal

from ..config.schema import ImageConfig


class ModalImageBuilder:
    """Builds Modal Images with dependencies."""

    def __init__(
        self,
        image_config: ImageConfig,
        image_name: str = "interp-gpu-image",
    ):
        """
        Initialize Modal image builder.

        Args:
            image_config: Image configuration
            image_name: Name for the built image (used for caching)
        """
        self.config = image_config
        self.image_name = image_name

    def build(self) -> modal.Image:
        """
        Build a Modal Image from the configuration.

        Returns:
            modal.Image object ready to use with Modal functions
        """
        # Start with base image matching the CUDA version from Docker config
        # Extract CUDA version from base_image like "nvidia/cuda:12.1.0-base-ubuntu22.04"
        base_image = self.config.base_image

        # Determine Python version
        python_version = self.config.python_version

        # Start with debian_slim and Python version
        image = modal.Image.debian_slim(python_version=python_version)

        # Install system packages
        if self.config.system_packages:
            image = image.apt_install(*self.config.system_packages)

        # Install Python packages
        if self.config.python_packages:
            image = image.uv_pip_install(*self.config.python_packages)

        # Install Scribe notebook server dependencies
        image = image.uv_pip_install(
            "jupyter_server",
            "nbformat",
            "tornado",
            "fastmcp",
            "Pillow",  # Required by scribe._image_processing_utils
            "requests",  # Required by scribe._notebook_server_utils
        )

        # Copy Scribe notebook server code into the image
        from pathlib import Path
        scribe_dir = Path(__file__).parent.parent.parent / "scribe"
        image = image.add_local_dir(str(scribe_dir), remote_path="/root/scribe")

        # Copy interp_infra code (needed for setup_pipeline)
        interp_infra_dir = Path(__file__).parent.parent
        image = image.add_local_dir(str(interp_infra_dir), remote_path="/root/interp_infra")

        # Copy skills directory (needed for kernel_setup)
        skills_dir = Path(__file__).parent.parent.parent / "skills"
        if skills_dir.exists():
            image = image.add_local_dir(str(skills_dir), remote_path="/root/skills")

        # Run custom setup commands
        if self.config.custom_setup_commands:
            image = image.run_commands(*self.config.custom_setup_commands)

        return image
