"""Build Modal images - standard and docker-in-docker variants."""

import os
import tempfile
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import modal

if TYPE_CHECKING:
    from .sandbox import ExecutionMode

# Use the 2025.06 Modal Image Builder
os.environ["MODAL_IMAGE_BUILDER_VERSION"] = "2025.06"


def add_notebook_support(image: modal.Image, notebook_packages: Optional[list[str]]) -> modal.Image:
    """
    Add Jupyter notebook server and MCP support to an image.

    Args:
        image: Base Modal image
        notebook_packages: Packages to install (if None, uses defaults)

    Returns:
        Image with notebook support added
    """
    if notebook_packages is None:
        packages = [
            "jupyter_server", "ipykernel", "jupyter", "jupyter_client",
            "nbformat", "tornado", "fastmcp", "Pillow", "requests",
            "torch", "transformers", "accelerate", "pandas",
            "matplotlib", "numpy", "seaborn", "datasets",
        ]
    else:
        packages = notebook_packages

    if packages:
        image = image.pip_install(*packages)

    # Copy scribe notebook server code
    scribe_dir = Path(__file__).parent.parent.parent / "scribe"
    if scribe_dir.exists():
        image = image.add_local_dir(str(scribe_dir), remote_path="/root/scribe")

    # Copy infra code
    infra_dir = Path(__file__).parent.parent
    image = image.add_local_dir(str(infra_dir), remote_path="/root/interp_infra")

    # Copy skills directory
    skills_dir = Path(__file__).parent.parent.parent / "skills"
    if skills_dir.exists():
        image = image.add_local_dir(str(skills_dir), remote_path="/root/skills")

    return image


class StandardImageBuilder:
    """Builds standard Modal images with Python dependencies."""

    def __init__(
        self,
        python_packages: list[str] = None,
        system_packages: list[str] = None,
        python_version: str = "3.11",
        execution_mode: Optional["ExecutionMode"] = None,
        notebook_packages: list[str] = None,
        custom_setup_commands: list[str] = None,
    ):
        """
        Initialize standard image builder.

        Args:
            python_packages: Python packages to install
            system_packages: System packages to apt-install
            python_version: Python version
            execution_mode: Execution mode (CLI, NOTEBOOK, or None for bare sandbox)
            notebook_packages: Packages for notebook mode (if None, uses defaults)
            custom_setup_commands: Additional setup commands
        """
        self.python_packages = python_packages or []
        self.system_packages = system_packages or []
        self.python_version = python_version
        self.execution_mode = execution_mode
        self.notebook_packages = notebook_packages
        self.custom_setup_commands = custom_setup_commands or []

    def build(self) -> modal.Image:
        """Build a Modal Image from the configuration."""
        image = modal.Image.debian_slim(python_version=self.python_version)

        if self.system_packages:
            image = image.apt_install(*self.system_packages)

        if self.python_packages:
            image = image.pip_install(*self.python_packages)

        if self.execution_mode:
            from .sandbox import ExecutionMode
            if self.execution_mode == ExecutionMode.NOTEBOOK:
                image = add_notebook_support(image, self.notebook_packages)

        if self.custom_setup_commands:
            image = image.run_commands(*self.custom_setup_commands)

        return image

    def get_sandbox_options(self) -> dict:
        """Get options for creating the sandbox."""
        return {}

    def get_sandbox_entrypoint(self) -> Optional[str]:
        """Get entrypoint command for sandbox."""
        return None

    def cleanup(self):
        """Clean up resources."""
        pass


class DockerImageBuilder:
    """Builds Modal images with docker-in-docker support."""

    def __init__(
        self,
        python_packages: list[str] = None,
        system_packages: list[str] = None,
        execution_mode: Optional["ExecutionMode"] = None,
        notebook_packages: list[str] = None,
        custom_setup_commands: list[str] = None,
    ):
        """
        Initialize docker-in-docker image builder.

        Args:
            python_packages: Python packages to install
            system_packages: System packages to apt-install
            execution_mode: Execution mode (CLI, NOTEBOOK, or None)
            notebook_packages: Packages for notebook mode
            custom_setup_commands: Additional setup commands
        """
        self.python_packages = python_packages or []
        self.system_packages = system_packages or []
        self.execution_mode = execution_mode
        self.notebook_packages = notebook_packages
        self.custom_setup_commands = custom_setup_commands or []
        self._dockerd_script_file = None

    def build(self) -> modal.Image:
        """Build docker-in-docker Modal Image."""
        dockerd_script = self._get_dockerd_script()

        # Write to temp file for Modal
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
            .run_commands([
                "rm $(which runc)",
                "wget https://github.com/opencontainers/runc/releases/download/v1.3.0/runc.amd64",
                "chmod +x runc.amd64",
                "mv runc.amd64 /usr/local/bin/runc",
            ])
            .run_commands([
                "update-alternatives --set iptables /usr/sbin/iptables-legacy",
                "update-alternatives --set ip6tables /usr/sbin/ip6tables-legacy",
            ])
            .add_local_file(self._dockerd_script_file.name, "/start-dockerd.sh", copy=True)
            .run_commands(["chmod +x /start-dockerd.sh"])
            .apt_install(["python3", "python3-pip", "python3-venv", "python-is-python3"])
        )

        if self.system_packages:
            image = image.apt_install(*self.system_packages)

        if self.python_packages:
            image = image.pip_install(*self.python_packages)

        if self.execution_mode:
            from .sandbox import ExecutionMode
            if self.execution_mode == ExecutionMode.NOTEBOOK:
                image = add_notebook_support(image, self.notebook_packages)

        if self.custom_setup_commands:
            image = image.run_commands(*self.custom_setup_commands)

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
        """Get options for creating the sandbox."""
        return {"experimental_options": {"enable_docker": True}}

    def get_sandbox_entrypoint(self) -> Optional[str]:
        """Get entrypoint command for sandbox."""
        return "/start-dockerd.sh"

    def cleanup(self):
        """Clean up temporary files."""
        if self._dockerd_script_file:
            os.unlink(self._dockerd_script_file.name)
