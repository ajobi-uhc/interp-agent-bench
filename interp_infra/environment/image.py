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

        # ===== ALL BUILD STEPS (must come before add_local_*) =====

        # Install system packages
        if self.config.system_packages:
            image = image.apt_install(*self.config.system_packages)

        # Add Docker support if enabled (must be early to install packages and create script)
        if self.config.enable_docker:
            image = self._add_docker_packages(image)
            image = self._add_docker_script(image)

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

        # Run custom setup commands
        if self.config.custom_setup_commands:
            image = image.run_commands(*self.config.custom_setup_commands)

        # ===== ALL ADD_LOCAL_* CALLS (must come after build steps) =====

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

        return image

    def _add_docker_packages(self, image: modal.Image) -> modal.Image:
        """
        Install Docker packages and dependencies (build steps only).

        This must be called before any add_local_* commands.
        Based on Modal's Docker-in-Sandbox documentation.
        """
        # Set builder version for Docker support
        import os
        os.environ["MODAL_IMAGE_BUILDER_VERSION"] = "2025.06"

        # Install Docker and dependencies
        image = (
            image
            .env({"DEBIAN_FRONTEND": "noninteractive"})
            .apt_install(["wget", "ca-certificates", "curl", "net-tools", "iproute2"])
            .run_commands([
                "install -m 0755 -d /etc/apt/keyrings",
                "curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc",
                "chmod a+r /etc/apt/keyrings/docker.asc",
                'echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian $(. /etc/os-release && echo \\"$VERSION_CODENAME\\") stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null',
            ])
            .apt_install([
                "docker-ce",
                "docker-ce-cli",
                "containerd.io",
                "docker-buildx-plugin",
                "docker-compose-plugin"
            ])
            # Install modern runc for reliable networking
            .run_commands([
                "rm $(which runc)",
                "wget https://github.com/opencontainers/runc/releases/download/v1.3.0/runc.amd64",
                "chmod +x runc.amd64",
                "mv runc.amd64 /usr/local/bin/runc",
            ])
            # Use iptables-legacy for gVisor compatibility
            .run_commands([
                "update-alternatives --set iptables /usr/sbin/iptables-legacy",
                "update-alternatives --set ip6tables /usr/sbin/ip6tables-legacy",
            ])
        )

        return image

    def _add_docker_script(self, image: modal.Image) -> modal.Image:
        """
        Add Docker daemon startup script using copy=True.
        Must be called BEFORE add_local_dir calls (those create mount layers).
        """
        script = """#!/bin/bash
set -xe -o pipefail

dev=$(ip route show default | awk '/default/ {print $5}')
if [ -z "$dev" ]; then
    echo "Error: No default device found."
    exit 1
fi
addr=$(ip addr show dev "$dev" | grep -w inet | awk '{print $2}' | cut -d/ -f1)
if [ -z "$addr" ]; then
    echo "Error: No IP address found for device $dev."
    exit 1
fi

echo 1 > /proc/sys/net/ipv4/ip_forward
iptables-legacy -t nat -A POSTROUTING -o "$dev" -j SNAT --to-source "$addr" -p tcp
iptables-legacy -t nat -A POSTROUTING -o "$dev" -j SNAT --to-source "$addr" -p udp

update-alternatives --set iptables /usr/sbin/iptables-legacy
update-alternatives --set ip6tables /usr/sbin/ip6tables-legacy

exec /usr/bin/dockerd --iptables=false --ip6tables=false --dns 8.8.8.8 --dns 8.8.4.4 -D
"""
        # Write to temp file and add with copy=True
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.sh') as f:
            f.write(script)
            f.flush()
            os.chmod(f.name, 0o755)
            # copy=True allows build steps after this
            image = image.add_local_file(f.name, "/start-dockerd.sh", copy=True)

        return image
