"""Sandbox management for Modal environments."""

import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import modal
import requests

from .image import ModalImageBuilder
from .volumes import (
    get_or_create_volume,
    check_model_in_volume,
    download_model_to_volume,
    commit_volumes,
)
from .handles import ModelHandle, RepoHandle


class ExecutionMode(Enum):
    """Execution mode for the sandbox."""
    CLI = "cli"  # Agent executes shell commands on sandbox
    NOTEBOOK = "notebook"  # Agent executes code in Jupyter kernel


@dataclass
class ModelConfig:
    """Configuration for a model to prepare."""
    name: str
    var_name: str = "model"
    hidden: bool = False
    is_peft: bool = False
    base_model: Optional[str] = None


@dataclass
class RepoConfig:
    """Configuration for a repository to clone."""
    url: str
    dockerfile: Optional[str] = None
    install: str = False


@dataclass
class SandboxConfig:
    """Configuration for a sandbox."""
    # Packages
    python_packages: list[str] = field(default_factory=list)
    system_packages: list[str] = field(default_factory=list)
    python_version: str = "3.11"

    # Hardware
    gpu: Optional[str] = None  # e.g. "H100", "A100"
    gpu_count: int = 1

    # Features
    docker_in_docker: bool = False
    execution_mode: Optional[ExecutionMode] = ExecutionMode.CLI
    debug: bool = False  # Start code-server for debugging
    timeout: int = 3600 * 24  # 24 hours

    # Secrets and environment
    secrets: list[str] = field(default_factory=list)  # named secrets
    env: dict[str, str] = field(default_factory=dict)
    encrypted_ports: list[int] = field(default_factory=list)  # Additional ports to expose

    # Ports
    jupyter_port: int = 8888
    rpc_port: int = 8080
    debug_port: int = 8080

    # Timeouts
    rpc_timeout: int = 600
    wait_timeout: int = 300

    # API Keys to pass through
    api_key_names: list[str] = field(default_factory=lambda: [
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"
    ])

    # HuggingFace secret name
    hf_secret_name: str = "huggingface-secret"

    # Notebook packages (only installed if execution_mode == NOTEBOOK)
    notebook_packages: list[str] = field(default_factory=lambda: [
        "jupyter_server", "ipykernel", "jupyter", "jupyter_client",
        "nbformat", "tornado", "fastmcp", "Pillow", "requests",
        "torch", "transformers", "accelerate", "pandas",
        "matplotlib", "numpy", "seaborn", "datasets"
    ])

    # Models to prepare (download to volumes)
    models: list[ModelConfig] = field(default_factory=list)

    # Repositories to clone
    repos: list[RepoConfig] = field(default_factory=list)

    # Local files to add to image [(local_path, remote_path)]
    local_files: list[tuple[str, str]] = field(default_factory=list)

    # Local directories to add to image [(local_path, remote_path)]
    local_dirs: list[tuple[str, str]] = field(default_factory=list)


class Sandbox:
    """
    A Modal sandbox environment.

    Handles image building, sandbox creation, model/repo preparation.

    Usage:
        config = SandboxConfig(
            python_packages=["torch", "transformers"],
            gpu="H100",
            execution_mode=ExecutionMode.NOTEBOOK,
        )

        sandbox = Sandbox(config)
        model = sandbox.prepare_model("google/gemma-9b")
        # Or prepare multiple models at once:
        # models = sandbox.prepare_models({
        #     "google/gemma-2-2b-it": {"hidden": False},
        #     "google/gemma-2-2b": {},
        # })
        sandbox.start(name="my-experiment")

        # models are now downloaded to volume, sandbox is running
    """
    
    def __init__(self, config: SandboxConfig):
        self.config = config
        self._sandbox: Optional[modal.Sandbox] = None
        self._app: Optional[modal.App] = None
        self._volumes: dict[str, modal.Volume] = {}
        self._model_handles: list[ModelHandle] = []
        self._repo_handles: list[RepoHandle] = []
        self._jupyter_url: Optional[str] = None
        self._code_server_url: Optional[str] = None
        self._image_builder: Optional[ModalImageBuilder] = None
        
    def start(self, name: str = "sandbox") -> "Sandbox":
        """
        Build image and start the sandbox.

        Args:
            name: Name for the Modal app

        Returns:
            self for chaining
        """
        print(f"Starting sandbox: {name}")

        # Prepare models from config
        self._prepare_models()

        # Prepare repos from config
        self._prepare_repos()

        image = self._build_image()
        self._app = modal.App.lookup(name, create_if_missing=True)
        self._create_sandbox(image)

        if self.config.docker_in_docker:
            print("  Starting docker daemon...")
            self._start_docker_daemon()

        self._download_prepared_models()
        self._clone_prepared_repos()
        commit_volumes(self._volumes)

        self._start_services()

        print("  Sandbox ready")
        return self

    def _build_image(self) -> modal.Image:
        """Build Modal image with dependencies."""
        print("  Building image...")
        self._image_builder = ModalImageBuilder(
            python_packages=self.config.python_packages,
            system_packages=self.config.system_packages,
            python_version=self.config.python_version,
            docker_in_docker=self.config.docker_in_docker,
            execution_mode=self.config.execution_mode,
            notebook_packages=self.config.notebook_packages,
        )
        image = self._image_builder.build()

        # Add local files to image
        for local_path, remote_path in self.config.local_files:
            print(f"  Adding local file: {local_path} -> {remote_path}")
            image = image.add_local_file(local_path=local_path, remote_path=remote_path)

        # Add local directories to image
        for local_path, remote_path in self.config.local_dirs:
            print(f"  Adding local dir: {local_path} -> {remote_path}")
            image = image.add_local_dir(local_path=local_path, remote_path=remote_path)

        return image

    def _create_sandbox(self, image: modal.Image):
        """Create and configure Modal sandbox."""
        gpu = f"{self.config.gpu}:{self.config.gpu_count}" if self.config.gpu else None
        if gpu:
            print(f"  GPU: {self.config.gpu} x{self.config.gpu_count}")

        secrets = self._collect_secrets()

        kwargs = {"image": image, "timeout": self.config.timeout, "app": self._app}
        if gpu:
            kwargs["gpu"] = gpu
        if secrets:
            kwargs["secrets"] = secrets
        if self.config.env:
            kwargs["env"] = self.config.env
        if self._volumes:
            kwargs["volumes"] = self._volumes
            print(f"  Volumes: {len(self._volumes)}")
        if self.config.docker_in_docker:
            kwargs["experimental_options"] = {"enable_docker": True}

        # Encrypted ports
        ports = list(self.config.encrypted_ports)
        if self.config.execution_mode == ExecutionMode.NOTEBOOK:
            ports.append(self.config.jupyter_port)
        if self.config.debug:
            ports.append(self.config.debug_port)
        if ports:
            kwargs["encrypted_ports"] = ports

        print("  Creating sandbox...")
        self._sandbox = modal.Sandbox.create(**kwargs)
        print(f"  Sandbox ID: {self._sandbox.object_id}")

    def _start_services(self):
        """Start execution mode services."""
        if self.config.execution_mode == ExecutionMode.NOTEBOOK:
            print("  Starting jupyter server...")
            self._start_jupyter()
            tunnels = self._sandbox.tunnels()
            self._jupyter_url = tunnels[self.config.jupyter_port].url
            self._wait_for_jupyter()
            print(f"  Jupyter URL: {self._jupyter_url}")
        elif self.config.execution_mode == ExecutionMode.CLI:
            print("  CLI mode - no services started")
        elif self.config.execution_mode is None:
            print("  Bare mode - no services started")

        if self.config.debug:
            print("  Debug mode: Starting code-server...")
            self._start_code_server()
            tunnels = self._sandbox.tunnels()
            self._code_server_url = tunnels[self.config.debug_port].url
            self._wait_for_code_server()
            print(f"  Code Server URL: {self._code_server_url}")

    def _run(self, *args) -> str:
        """Run command and return stdout."""
        if not self._sandbox:
            raise RuntimeError("Sandbox not started. Call start() first.")

        p = self._sandbox.exec(*args)
        stdout = p.stdout.read()
        p.wait()

        if p.returncode != 0:
            stderr = p.stderr.read()
            raise RuntimeError(f"Command failed (exit {p.returncode}): {stderr}")

        return stdout

    def exec(self, cmd: str) -> str:
        """Execute a shell command in the sandbox."""
        return self._run("bash", "-c", cmd)

    def exec_python(self, code: str) -> str:
        """Execute Python code in the sandbox."""
        return self._run("python", "-c", code)

    def _prepare_models(self):
        """Prepare models from config by setting up volumes."""
        for model_cfg in self.config.models:
            base_model_path = None

            # Handle PEFT base model first
            if model_cfg.is_peft:
                if not model_cfg.base_model:
                    raise ValueError("PEFT models require base_model to be specified")

                base_volume, base_mount = get_or_create_volume(model_cfg.base_model)
                self._volumes[base_mount] = base_volume
                base_model_path = base_mount

            # Get or create volume for main model/adapter
            volume, mount_path = get_or_create_volume(model_cfg.name)
            self._volumes[mount_path] = volume

            handle = ModelHandle(
                name=model_cfg.name,
                volume_path=mount_path,
                var_name=model_cfg.var_name,
                hidden=model_cfg.hidden,
                is_peft=model_cfg.is_peft,
                base_model=model_cfg.base_model,
                base_model_path=base_model_path,
            )
            self._model_handles.append(handle)

    def _prepare_repos(self):
        """Prepare repos from config."""
        for repo_cfg in self.config.repos:
            url = repo_cfg.url
            if not url.startswith("http"):
                url = f"https://github.com/{url}"

            repo_name = url.split("/")[-1].replace(".git", "")
            local_path = f"/workspace/{repo_name}"

            handle = RepoHandle(
                url=url,
                local_path=local_path,
                dockerfile=repo_cfg.dockerfile,
                container_name=repo_name if repo_cfg.dockerfile else None,
                install=repo_cfg.install,
            )
            self._repo_handles.append(handle)

    def start_container(self, repo_handle: RepoHandle) -> None:
        """
        Start a container for a repo that has a dockerfile.

        Args:
            repo_handle: RepoHandle with dockerfile specified
        """
        if not repo_handle.dockerfile:
            raise ValueError("RepoHandle has no dockerfile")

        if not self.config.docker_in_docker:
            raise RuntimeError("docker_in_docker must be enabled")

        if not self._sandbox:
            raise RuntimeError("Sandbox not started")

        # Build container
        dockerfile_path = f"{repo_handle.local_path}/{repo_handle.dockerfile}"
        print(f"  Building container {repo_handle.container_name}...")
        self.exec(f"docker build -t {repo_handle.container_name} -f {dockerfile_path} {repo_handle.local_path}")

        # Run container
        print(f"  Starting container {repo_handle.container_name}...")
        self.exec(f"docker run -d --name {repo_handle.container_name} {repo_handle.container_name}")

        repo_handle.container_running = True

    def _collect_secrets(self) -> list[modal.Secret]:
        """Collect Modal secrets."""
        secrets = []

        # Named secrets from config
        for secret_name in self.config.secrets:
            try:
                secrets.append(modal.Secret.from_name(secret_name))
            except modal.exception.NotFoundError:
                print(f"  Warning: Secret '{secret_name}' not found")

        # HF secret (optional)
        if self.config.hf_secret_name:
            try:
                secrets.append(modal.Secret.from_name(self.config.hf_secret_name))
            except modal.exception.NotFoundError:
                pass

        # Modal credentials for nested sandboxes
        if os.getenv("MODAL_TOKEN_ID") and os.getenv("MODAL_TOKEN_SECRET"):
            secrets.append(modal.Secret.from_dict({
                "MODAL_TOKEN_ID": os.environ["MODAL_TOKEN_ID"],
                "MODAL_TOKEN_SECRET": os.environ["MODAL_TOKEN_SECRET"],
            }))

        # API keys from environment
        api_keys = {key: os.environ[key] for key in self.config.api_key_names if os.getenv(key)}
        if api_keys:
            secrets.append(modal.Secret.from_dict(api_keys))

        return secrets
    
    def _start_docker_daemon(self):
        """Start docker daemon in background."""
        dockerd_script = self._get_dockerd_script()

        # Write and execute script
        self._sandbox.open("/start-dockerd.sh", "w").write(dockerd_script)
        self.exec("chmod +x /start-dockerd.sh")
        self.exec("nohup /start-dockerd.sh > /var/log/dockerd.log 2>&1 &")

        # Wait for docker to be ready
        for _ in range(30):
            try:
                self.exec("docker info > /dev/null 2>&1")
                return
            except RuntimeError:
                time.sleep(1)

        raise RuntimeError("Docker daemon failed to start")
    
    def _get_dockerd_script(self) -> str:
        """Get the docker daemon startup script."""
        return '''#!/bin/bash
set -xe -o pipefail

dev=$(ip route show default | awk '/default/ {print $5}')
if [ -z "$dev" ]; then
    echo "Error: No default device found."
    ip route show
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

exec /usr/bin/dockerd --iptables=false --ip6tables=false -D
'''
    
    def _start_jupyter(self):
        """Start jupyter server in background."""
        jupyter_script = f'''
import sys
sys.path.insert(0, "/root")
from scribe.notebook.notebook_server import ScribeServerApp
app = ScribeServerApp()
app.initialize([
    "--ip=0.0.0.0",
    "--port={self.config.jupyter_port}",
    "--ServerApp.token=",
    "--ServerApp.password=",
    "--ServerApp.allow_root=True",
])
app.start()
'''
        # Escape for shell
        escaped_script = jupyter_script.replace('"', '\\"')
        self.exec(f'nohup python -c "{escaped_script}" > /var/log/jupyter.log 2>&1 &')

    def _wait_for_service(self, url: str, service_name: str, max_retries: int = 100, retry_delay: float = 2.0):
        """Wait for HTTP service to be ready."""
        for _ in range(max_retries):
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    return
            except (requests.RequestException, ConnectionError):
                pass
            time.sleep(retry_delay)
        print(f"  Warning: {service_name} may not be fully ready")

    def _wait_for_jupyter(self):
        """Wait for jupyter server to be ready."""
        self._wait_for_service(f"{self._jupyter_url}/api/scribe/health", "Jupyter")

    def _start_code_server(self):
        """Start code-server (VS Code in browser)."""
        install_script = "curl -fsSL https://code-server.dev/install.sh | sh > /var/log/code-server-install.log 2>&1"
        self.exec(install_script)
        self.exec(f'nohup code-server --bind-addr 0.0.0.0:{self.config.debug_port} --auth none /workspace > /var/log/code-server.log 2>&1 &')

    def _wait_for_code_server(self):
        """Wait for code-server to be ready."""
        self._wait_for_service(f"{self._code_server_url}/healthz", "Code-server")
    
    def _download_prepared_models(self):
        """Download all prepared models."""
        for handle in self._model_handles:
            self._download_model(handle)
    
    def _download_model(self, handle: ModelHandle):
        """Download a model to its volume."""
        # Download base model for PEFT
        if handle.is_peft and handle.base_model:
            if not check_model_in_volume(self, handle.base_model_path):
                print(f"  Downloading base model: {handle.base_model}")
                download_model_to_volume(self, handle.base_model, handle.base_model_path)

        # Download main model/adapter
        if not check_model_in_volume(self, handle.volume_path):
            if handle.hidden:
                print(f"  Downloading model...")
            else:
                print(f"  Downloading model: {handle.name}")
            download_model_to_volume(self, handle.name, handle.volume_path)

    def _clone_prepared_repos(self):
        """Clone all prepared repos."""
        for handle in self._repo_handles:
            self._clone_repo(handle)
    
    def _clone_repo(self, handle: RepoHandle):
        """Clone a repo to local path."""
        print(f"  Cloning repo: {handle.url}")
        script = f'''
import subprocess
from pathlib import Path

repo_path = Path("{handle.local_path}")
repo_path.parent.mkdir(parents=True, exist_ok=True)

if not repo_path.exists():
    subprocess.run(
        ["git", "clone", "{handle.url}", str(repo_path)],
        check=True
    )
'''
        self.exec_python(script)

        # Install if requested
        if handle.install:
            print(f"  Installing repo: {handle.local_path}")
            try:
                self.exec(f"cd {handle.local_path} && {handle.install}")
                print(f"  Installation successful")
            except RuntimeError as e:
                print(f"  Warning: Installation failed: {e}")
                print(f"  You can manually install in the sandbox later")

    @property
    def jupyter_url(self) -> Optional[str]:
        """Get jupyter URL if notebook mode."""
        return self._jupyter_url

    @property
    def code_server_url(self) -> Optional[str]:
        """Get code-server URL if code_server mode."""
        return self._code_server_url

    @property
    def sandbox_id(self) -> Optional[str]:
        """Get Modal sandbox ID."""
        if self._sandbox:
            return self._sandbox.object_id
        return None
    
    @property
    def modal_sandbox(self) -> Optional[modal.Sandbox]:
        """Get underlying Modal sandbox object."""
        return self._sandbox

    def terminate(self):
        """Terminate the sandbox."""
        if self._sandbox:
            self._sandbox.terminate()
            self._sandbox = None
        if self._image_builder:
            self._image_builder.cleanup()