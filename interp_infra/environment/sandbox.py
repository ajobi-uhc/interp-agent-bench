"""Sandbox management for Modal environments."""

import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import modal
import requests

from .image import ModalImageBuilder
from .volumes import get_or_create_volume, check_model_in_volume, download_model_to_volume, commit_volumes
from .handles import ModelHandle, RepoHandle
from ._scripts import DOCKERD_SCRIPT, jupyter_startup_script, code_server_install_script


class ExecutionMode(Enum):
    CLI = "cli"
    NOTEBOOK = "notebook"


@dataclass
class ModelConfig:
    name: str
    var_name: str = "model"
    hidden: bool = False
    is_peft: bool = False
    base_model: Optional[str] = None


@dataclass
class RepoConfig:
    url: str
    dockerfile: Optional[str] = None
    install: str = False


@dataclass
class SandboxConfig:
    python_packages: list[str] = field(default_factory=list)
    system_packages: list[str] = field(default_factory=list)
    python_version: str = "3.11"
    gpu: Optional[str] = None
    gpu_count: int = 1
    docker_in_docker: bool = False
    execution_mode: Optional[ExecutionMode] = ExecutionMode.CLI
    debug: bool = False
    timeout: int = 3600 * 24
    secrets: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    encrypted_ports: list[int] = field(default_factory=list)
    jupyter_port: int = 8888
    rpc_port: int = 8080
    debug_port: int = 8080
    rpc_timeout: int = 600
    wait_timeout: int = 300
    api_key_names: list[str] = field(default_factory=lambda: ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"])
    hf_secret_name: str = "huggingface-secret"
    notebook_packages: list[str] = field(default_factory=lambda: [
        "jupyter_server", "ipykernel", "jupyter", "jupyter_client", "nbformat", "tornado",
        "fastmcp", "Pillow", "requests", "torch", "transformers", "accelerate",
        "pandas", "matplotlib", "numpy", "seaborn", "datasets"
    ])
    models: list[ModelConfig] = field(default_factory=list)
    repos: list[RepoConfig] = field(default_factory=list)
    local_files: list[tuple[str, str]] = field(default_factory=list)
    local_dirs: list[tuple[str, str]] = field(default_factory=list)


class Sandbox:
    """Modal sandbox environment that handles setup, execution, and teardown."""

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
        """Build image and start the sandbox."""
        print(f"Starting sandbox: {name}")

        self._prepare_models()
        self._prepare_repos()
        image = self._build_image()

        self._app = modal.App.lookup(name, create_if_missing=True)
        self._create_sandbox(image)

        if self.config.docker_in_docker:
            self._start_docker_daemon()

        self._download_models()
        self._clone_repos()
        commit_volumes(self._volumes)
        self._start_services()

        print("Sandbox ready")
        return self

    def exec(self, cmd: str) -> str:
        """Execute shell command in sandbox."""
        return self._exec("bash", "-c", cmd)

    def exec_python(self, code: str) -> str:
        """Execute Python code in sandbox."""
        return self._exec("python", "-c", code)

    def start_container(self, repo_handle: RepoHandle) -> None:
        """Start a container for a repo with a dockerfile."""
        if not repo_handle.dockerfile or not self.config.docker_in_docker or not self._sandbox:
            raise RuntimeError("Invalid container setup")

        dockerfile_path = f"{repo_handle.local_path}/{repo_handle.dockerfile}"
        self.exec(f"docker build -t {repo_handle.container_name} -f {dockerfile_path} {repo_handle.local_path}")
        self.exec(f"docker run -d --name {repo_handle.container_name} {repo_handle.container_name}")
        repo_handle.container_running = True

    def terminate(self):
        """Terminate the sandbox."""
        if self._sandbox:
            self._sandbox.terminate()
            self._sandbox = None
        if self._image_builder:
            self._image_builder.cleanup()

    @property
    def jupyter_url(self) -> Optional[str]:
        return self._jupyter_url

    @property
    def code_server_url(self) -> Optional[str]:
        return self._code_server_url

    @property
    def sandbox_id(self) -> Optional[str]:
        return self._sandbox.object_id if self._sandbox else None

    @property
    def modal_sandbox(self) -> Optional[modal.Sandbox]:
        return self._sandbox

    # Internal methods

    def _build_image(self) -> modal.Image:
        """Build Modal image with dependencies."""
        self._image_builder = ModalImageBuilder(
            python_packages=self.config.python_packages,
            system_packages=self.config.system_packages,
            python_version=self.config.python_version,
            docker_in_docker=self.config.docker_in_docker,
            execution_mode=self.config.execution_mode,
            notebook_packages=self.config.notebook_packages,
        )
        image = self._image_builder.build()

        for local_path, remote_path in self.config.local_files:
            image = image.add_local_file(local_path=local_path, remote_path=remote_path)

        for local_path, remote_path in self.config.local_dirs:
            image = image.add_local_dir(local_path=local_path, remote_path=remote_path)

        return image

    def _create_sandbox(self, image: modal.Image):
        """Create Modal sandbox with configuration."""
        gpu = f"{self.config.gpu}:{self.config.gpu_count}" if self.config.gpu else None
        secrets = self._collect_secrets()

        ports = list(self.config.encrypted_ports)
        if self.config.execution_mode == ExecutionMode.NOTEBOOK:
            ports.append(self.config.jupyter_port)
        if self.config.debug:
            ports.append(self.config.debug_port)

        kwargs = {
            "image": image,
            "timeout": self.config.timeout,
            "app": self._app,
            **({"gpu": gpu} if gpu else {}),
            **({"secrets": secrets} if secrets else {}),
            **({"env": self.config.env} if self.config.env else {}),
            **({"volumes": self._volumes} if self._volumes else {}),
            **({"encrypted_ports": ports} if ports else {}),
            **({"experimental_options": {"enable_docker": True}} if self.config.docker_in_docker else {}),
        }

        self._sandbox = modal.Sandbox.create(**kwargs)

    def _exec(self, *args) -> str:
        """Execute command and return stdout."""
        if not self._sandbox:
            raise RuntimeError("Sandbox not started")

        p = self._sandbox.exec(*args)
        stdout = p.stdout.read()
        p.wait()

        if p.returncode != 0:
            stderr = p.stderr.read()
            raise RuntimeError(f"Command failed: {stderr}")

        return stdout

    def _prepare_models(self):
        """Setup model volumes and handles."""
        for model_cfg in self.config.models:
            base_model_path = None

            if model_cfg.is_peft:
                if not model_cfg.base_model:
                    raise ValueError("PEFT models require base_model")

                base_volume, base_mount = get_or_create_volume(model_cfg.base_model)
                self._volumes[base_mount] = base_volume
                base_model_path = base_mount

            volume, mount_path = get_or_create_volume(model_cfg.name)
            self._volumes[mount_path] = volume

            self._model_handles.append(ModelHandle(
                name=model_cfg.name,
                volume_path=mount_path,
                var_name=model_cfg.var_name,
                hidden=model_cfg.hidden,
                is_peft=model_cfg.is_peft,
                base_model=model_cfg.base_model,
                base_model_path=base_model_path,
            ))

    def _prepare_repos(self):
        """Setup repo handles."""
        for repo_cfg in self.config.repos:
            url = repo_cfg.url if repo_cfg.url.startswith("http") else f"https://github.com/{repo_cfg.url}"
            repo_name = url.split("/")[-1].replace(".git", "")

            self._repo_handles.append(RepoHandle(
                url=url,
                local_path=f"/workspace/{repo_name}",
                dockerfile=repo_cfg.dockerfile,
                container_name=repo_name if repo_cfg.dockerfile else None,
                install=repo_cfg.install,
            ))

    def _collect_secrets(self) -> list[modal.Secret]:
        """Collect all Modal secrets."""
        secrets = []

        for secret_name in self.config.secrets:
            try:
                secrets.append(modal.Secret.from_name(secret_name))
            except modal.exception.NotFoundError:
                pass

        if self.config.hf_secret_name:
            try:
                secrets.append(modal.Secret.from_name(self.config.hf_secret_name))
            except modal.exception.NotFoundError:
                pass

        if os.getenv("MODAL_TOKEN_ID") and os.getenv("MODAL_TOKEN_SECRET"):
            secrets.append(modal.Secret.from_dict({
                "MODAL_TOKEN_ID": os.environ["MODAL_TOKEN_ID"],
                "MODAL_TOKEN_SECRET": os.environ["MODAL_TOKEN_SECRET"],
            }))

        api_keys = {k: os.environ[k] for k in self.config.api_key_names if k in os.environ}
        if api_keys:
            secrets.append(modal.Secret.from_dict(api_keys))

        return secrets

    def _download_models(self):
        """Download all models to volumes."""
        for handle in self._model_handles:
            if handle.is_peft and handle.base_model and not check_model_in_volume(self, handle.base_model_path):
                print(f"Downloading base: {handle.base_model}")
                download_model_to_volume(self, handle.base_model, handle.base_model_path)

            if not check_model_in_volume(self, handle.volume_path):
                model_name = "model" if handle.hidden else handle.name
                print(f"Downloading: {model_name}")
                download_model_to_volume(self, handle.name, handle.volume_path)

    def _clone_repos(self):
        """Clone all repos."""
        for handle in self._repo_handles:
            print(f"Cloning: {handle.url}")
            self.exec_python(f'''
import subprocess
from pathlib import Path

repo_path = Path("{handle.local_path}")
repo_path.parent.mkdir(parents=True, exist_ok=True)

if not repo_path.exists():
    subprocess.run(["git", "clone", "{handle.url}", str(repo_path)], check=True)
''')
            if handle.install:
                try:
                    self.exec(f"cd {handle.local_path} && {handle.install}")
                except RuntimeError:
                    pass

    def _start_services(self):
        """Start execution mode services."""
        if self.config.execution_mode == ExecutionMode.NOTEBOOK:
            self._start_jupyter()
            self._jupyter_url = self._sandbox.tunnels()[self.config.jupyter_port].url
            self._wait_for_service(f"{self._jupyter_url}/api/scribe/health")
            print(f"Jupyter: {self._jupyter_url}")

        if self.config.debug:
            self._start_code_server()
            self._code_server_url = self._sandbox.tunnels()[self.config.debug_port].url
            self._wait_for_service(f"{self._code_server_url}/healthz")
            print(f"Code-server: {self._code_server_url}")

    def _start_docker_daemon(self):
        """Start docker daemon."""
        self._sandbox.open("/start-dockerd.sh", "w").write(DOCKERD_SCRIPT)
        self.exec("chmod +x /start-dockerd.sh && nohup /start-dockerd.sh > /var/log/dockerd.log 2>&1 &")

        for _ in range(30):
            try:
                self.exec("docker info > /dev/null 2>&1")
                return
            except RuntimeError:
                time.sleep(1)

        raise RuntimeError("Docker daemon failed to start")

    def _start_jupyter(self):
        """Start Jupyter server."""
        script = jupyter_startup_script(self.config.jupyter_port).replace('"', '\\"')
        self.exec(f'nohup python -c "{script}" > /var/log/jupyter.log 2>&1 &')

    def _start_code_server(self):
        """Start code-server."""
        self.exec(f"{code_server_install_script()} > /var/log/code-server-install.log 2>&1")
        self.exec(f'nohup code-server --bind-addr 0.0.0.0:{self.config.debug_port} --auth none /workspace > /var/log/code-server.log 2>&1 &')

    def _wait_for_service(self, url: str, max_retries: int = 100):
        """Wait for HTTP service to be ready."""
        for _ in range(max_retries):
            try:
                if requests.get(url, timeout=5).status_code == 200:
                    return
            except (requests.RequestException, ConnectionError):
                pass
            time.sleep(2)
