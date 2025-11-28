"""Build Modal images from configuration."""

import os
import tempfile
from pathlib import Path
from typing import Optional

import modal

# Use the 2025.06 Modal Image Builder which avoids the need to install Modal client
# dependencies into the container image.
os.environ["MODAL_IMAGE_BUILDER_VERSION"] = "2025.06"


class ModalImageBuilder:
    """Builds Modal Images with dependencies."""

    def __init__(
        self,
        python_packages: list[str] = None,
        system_packages: list[str] = None,
        python_version: str = "3.11",
        docker_in_docker: bool = False,
        notebook: bool = False,
        custom_setup_commands: list[str] = None,
    ):
        """
        Initialize Modal image builder.

        Args:
            python_packages: Python packages to install
            system_packages: System packages to apt-install
            python_version: Python version
            docker_in_docker: Whether to enable docker-in-docker
            notebook: Whether to include notebook/jupyter server
            custom_setup_commands: Additional setup commands
        """
        self.python_packages = python_packages or []
        self.system_packages = system_packages or []
        self.python_version = python_version
        self.docker_in_docker = docker_in_docker
        self.notebook = notebook
        self.custom_setup_commands = custom_setup_commands or []
        
        # Store dockerd script path if needed
        self._dockerd_script_file = None

    def build(self) -> modal.Image:
        """
        Build a Modal Image from the configuration.

        Returns:
            modal.Image object ready to use with Modal functions
        """
        if self.docker_in_docker:
            image = self._build_docker_base()
        else:
            image = self._build_standard_base()

        # Install user-specified system packages
        if self.system_packages:
            image = image.apt_install(*self.system_packages)

        # Install user-specified Python packages
        if self.python_packages:
            image = image.pip_install(*self.python_packages)

        # Add notebook support if needed
        if self.notebook:
            image = self._add_notebook_support(image)

        # Run custom setup commands
        if self.custom_setup_commands:
            image = image.run_commands(*self.custom_setup_commands)

        return image

    def _build_standard_base(self) -> modal.Image:
        """Build standard debian slim base."""
        return modal.Image.debian_slim(python_version=self.python_version)

    def _build_docker_base(self) -> modal.Image:
        """Build base image with docker-in-docker support."""
        # Create the dockerd startup script
        dockerd_script = self._get_dockerd_script()
        
        # Write to temp file for Modal to pick up
        self._dockerd_script_file = tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".sh"
        )
        self._dockerd_script_file.write(dockerd_script)
        self._dockerd_script_file.flush()
        os.chmod(self._dockerd_script_file.name, 0o755)

        image = (
            modal.Image.from_registry("ubuntu:22.04")
            .env({"DEBIAN_FRONTEND": "noninteractive"})
            .apt_install(["wget", "ca-certificates", "curl", "net-tools", "iproute2"])
            .run_commands([
                "install -m 0755 -d /etc/apt/keyrings",
                "curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc",
                "chmod a+r /etc/apt/keyrings/docker.asc",
                'echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo \\"${UBUNTU_CODENAME:-$VERSION_CODENAME}\\") stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null',
                "mkdir /build",
            ])
            .apt_install([
                "docker-ce=5:27.5.0-1~ubuntu.22.04~jammy",
                "docker-ce-cli=5:27.5.0-1~ubuntu.22.04~jammy",
                "containerd.io",
                "docker-buildx-plugin",
                "docker-compose-plugin",
            ])
            # Modern runc for reliable networking
            .run_commands([
                "rm $(which runc)",
                "wget https://github.com/opencontainers/runc/releases/download/v1.3.0/runc.amd64",
                "chmod +x runc.amd64",
                "mv runc.amd64 /usr/local/bin/runc",
            ])
            # Use iptables-legacy (gVisor doesn't support nftables)
            .run_commands([
                "update-alternatives --set iptables /usr/sbin/iptables-legacy",
                "update-alternatives --set ip6tables /usr/sbin/ip6tables-legacy",
            ])
            # Add dockerd startup script
            .add_local_file(self._dockerd_script_file.name, "/start-dockerd.sh", copy=True)
            .run_commands(["chmod +x /start-dockerd.sh"])
            # Install Python since we started from ubuntu, not debian_slim
            .apt_install(["python3", "python3-pip", "python3-venv", "python-is-python3"])
        )

        return image

    def _add_notebook_support(self, image: modal.Image) -> modal.Image:
        """Add Jupyter notebook server and MCP support."""
        # Install notebook dependencies and common ML packages
        image = image.pip_install(
            # Jupyter core
            "jupyter_server",
            "ipykernel",
            "jupyter",
            "jupyter_client",
            "nbformat",
            "tornado",
            # MCP and utilities
            "fastmcp",
            "Pillow",
            "requests",
            # ML and data science packages
            "torch",
            "transformers",
            "accelerate",
            "pandas",
            "matplotlib",
            "numpy",
            "seaborn",
        )

        # Copy scribe notebook server code
        scribe_dir = Path(__file__).parent.parent.parent / "scribe"
        if scribe_dir.exists():
            image = image.add_local_dir(str(scribe_dir), remote_path="/root/scribe")

        # Copy infra code (needed for setup pipeline)
        infra_dir = Path(__file__).parent.parent
        image = image.add_local_dir(str(infra_dir), remote_path="/root/interp_infra")

        # Copy skills directory
        skills_dir = Path(__file__).parent.parent.parent / "skills"
        if skills_dir.exists():
            image = image.add_local_dir(str(skills_dir), remote_path="/root/skills")

        return image

    def _get_dockerd_script(self) -> str:
        """Get the docker daemon startup script."""
        return """#!/bin/bash
set -xe -o pipefail

dev=$(ip route show default | awk '/default/ {print $5}')
if [ -z "$dev" ]; then
    echo "Error: No default device found."
    ip route show
    exit 1
else
    echo "Default device: $dev"
fi
addr=$(ip addr show dev "$dev" | grep -w inet | awk '{print $2}' | cut -d/ -f1)
if [ -z "$addr" ]; then
    echo "Error: No IP address found for device $dev."
    ip addr show dev "$dev"
    exit 1
else
    echo "IP address for $dev: $addr"
fi

echo 1 > /proc/sys/net/ipv4/ip_forward
iptables-legacy -t nat -A POSTROUTING -o "$dev" -j SNAT --to-source "$addr" -p tcp
iptables-legacy -t nat -A POSTROUTING -o "$dev" -j SNAT --to-source "$addr" -p udp

update-alternatives --set iptables /usr/sbin/iptables-legacy
update-alternatives --set ip6tables /usr/sbin/ip6tables-legacy

exec /usr/bin/dockerd --iptables=false --ip6tables=false -D"""

    def get_sandbox_options(self) -> dict:
        """Get options needed when creating the sandbox."""
        options = {}
        if self.docker_in_docker:
            options["experimental_options"] = {"enable_docker": True}
        return options

    def get_sandbox_entrypoint(self) -> Optional[str]:
        """Get entrypoint command for sandbox, if any."""
        if self.docker_in_docker:
            return "/start-dockerd.sh"
        return None

    def cleanup(self):
        """Clean up temporary files."""
        if self._dockerd_script_file:
            try:
                os.unlink(self._dockerd_script_file.name)
            except:
                pass