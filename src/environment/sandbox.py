"""Sandbox management for Modal environments."""

import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import modal

try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file if it exists
except ImportError:
    pass  # dotenv not installed, skip

from .utils import (
    ModalImageBuilder,
    get_or_create_volume,
    check_model_in_volume,
    download_model_to_volume,
    commit_volumes,
    start_jupyter,
    start_docker_daemon,
    start_code_server,
    wait_for_service,
)
from ..harness.logging import get_logger

logger = get_logger("sandbox")


# Resource handles

@dataclass
class ModelHandle:
    """Handle to a prepared model."""
    name: str
    volume_path: str
    var_name: str = "model"
    hidden: bool = False
    is_peft: bool = False
    base_model: Optional[str] = None
    base_model_path: Optional[str] = None


@dataclass
class RepoHandle:
    """Handle to a prepared repo."""
    url: str
    local_path: str
    dockerfile: Optional[str] = None
    container_name: Optional[str] = None
    container_running: bool = False
    install: str = False


class ExecutionMode(Enum):
    CLI = "cli"
    NOTEBOOK = "notebook"


@dataclass
class ModelConfig:
    """Configuration for a model to load in the sandbox.

    Args:
        name: HuggingFace model ID (e.g. "google/gemma-2-9b")
        var_name: Variable name for model info dict (default: "model")
        hidden: Hide model details from agent (default: False)
        is_peft: Model is a PEFT adapter (default: False)
        base_model: Base model ID if this is a PEFT adapter
    """
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
    """Configuration for Modal sandbox environment.

    Args:
        gpu: GPU type (e.g. "A100", "H100", None for CPU)
        models: List of ModelConfig for models to download
        python_packages: List of pip packages to install
        system_packages: List of apt packages to install
        secrets: List of Modal secret names to mount
        execution_mode: ExecutionMode.NOTEBOOK or ExecutionMode.CLI
        timeout: Sandbox timeout in seconds (default: 1 hour)
        local_files: List of (local_path, sandbox_path) tuples for files
        local_dirs: List of (local_path, sandbox_path) tuples for directories
    """
    python_packages: list[str] = field(default_factory=list)
    system_packages: list[str] = field(default_factory=list)
    python_version: str = "3.11"
    gpu: Optional[str] = None
    gpu_count: int = 1
    docker_in_docker: bool = False
    execution_mode: Optional[ExecutionMode] = ExecutionMode.CLI
    debug: bool = False
    timeout: int = 3600
    secrets: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    encrypted_ports: list[int] = field(default_factory=list)
    jupyter_port: int = 8888
    rpc_port: int = 8080
    debug_port: int = 8080
    rpc_timeout: int = 600
    wait_timeout: int = 300
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
    """Modal sandbox environment.

    Provisions GPU compute, downloads models, installs packages, starts services.

    Example:
        config = SandboxConfig(
            gpu="A100",
            models=[ModelConfig(name="google/gemma-2-9b")],
            python_packages=["torch", "transformers"]
        )
        sandbox = Sandbox(config).start()
        # ... use sandbox ...
        sandbox.terminate()
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

    def _check_modal_auth(self):
        """Check if Modal is properly authenticated."""
        try:
            user_config = modal.config._user_config
            if not user_config or not isinstance(user_config, dict):
                raise ValueError("No Modal configuration found")

            # Check if any profile has a valid token
            has_valid_token = any(
                profile.get('token_id') and profile.get('token_secret')
                for profile in user_config.values()
                if isinstance(profile, dict)
            )

            if not has_valid_token:
                raise ValueError("No valid Modal token found")

        except Exception as e:
            raise RuntimeError(
                f"Modal is not configured properly: {e}\n\n"
                "Please authenticate with Modal:\n"
                "  1. Run: modal token new\n"
                "  2. Follow the prompts to log in\n"
                "  3. Try running your script again\n\n"
                "For more info: https://modal.com/docs/guide/getting-started"
            )

    def start(self, name: str = "sandbox", snapshot_image: Optional[modal.Image] = None) -> "Sandbox":
        """
        Build image and start the sandbox.

        Args:
            name: Name for the sandbox app
            snapshot_image: Optional snapshot image to restore from. If provided,
                          this will be used as the base image instead of building fresh.

        Returns:
            Self for chaining
        """
        if snapshot_image:
            logger.info(f"Starting sandbox from snapshot: {name}")
        else:
            logger.info(f"Starting sandbox: {name}")

        # Check Modal authentication early
        self._check_modal_auth()

        # Setup models and repos (creates handles + volumes)
        self._setup_models()
        self._setup_repos()

        logger.info("Models and repos are configured")

        # Build image (use snapshot if provided, otherwise build fresh)
        if snapshot_image:
            logger.info("Using snapshot image as base...")
            image = snapshot_image
        else:
            image = self._build_image()

        self._app = modal.App.lookup(name, create_if_missing=True)
        self._create_sandbox(image)

        # Start docker if needed
        if self.config.docker_in_docker:
            start_docker_daemon(self)

        # Download models and clone repos (only if not using snapshot)
        if not snapshot_image:
            logger.info("Setting up models and repos...")
            self._download_models()
            self._clone_repos()
            logger.info("Models and repos are set up")
            commit_volumes(self._volumes)

        # Start services (jupyter, code-server)
        logger.info("Starting services...")
        self._start_services()

        logger.info("Sandbox ready")
        return self

    @classmethod
    def from_snapshot(cls, snapshot_image: modal.Image, config: SandboxConfig, name: str = "sandbox") -> "Sandbox":
        """
        Create and start a new sandbox from a snapshot.

        This is a convenience method that creates a Sandbox instance and starts it
        with the provided snapshot image.

        Args:
            snapshot_image: The snapshot image to restore from
            config: Configuration for the new sandbox
            name: Name for the sandbox app

        Returns:
            Started Sandbox instance

        Example:
            # Create snapshot from existing sandbox
            snapshot = sandbox1.snapshot("checkpoint 1")

            # Later, restore to new sandbox
            sandbox2 = Sandbox.from_snapshot(snapshot, config)
            # sandbox2 now has all files from when snapshot was taken
        """
        sandbox = cls(config)
        return sandbox.start(name=name, snapshot_image=snapshot_image)

    @classmethod
    def from_id(cls, sandbox_id: str, config: Optional[SandboxConfig] = None) -> "Sandbox":
        """
        Reconnect to an existing running sandbox by its ID.

        This allows you to reconnect to a sandbox that was created earlier,
        enabling management operations like exec, terminate, snapshot, etc.

        Args:
            sandbox_id: The Modal sandbox object ID (e.g., "sb-xxx...")
            config: Optional config (used for timeout settings, etc.)

        Returns:
            Sandbox instance connected to the existing sandbox

        Example:
            # Original script outputs sandbox_id
            sandbox = Sandbox(config).start()
            print(sandbox.sandbox_id)  # "sb-abc123..."

            # Later, reconnect to manage it
            sandbox = Sandbox.from_id("sb-abc123...")
            sandbox.exec("nvidia-smi")
            sandbox.terminate()
        """
        instance = cls(config or SandboxConfig())
        instance._sandbox = modal.Sandbox.from_id(sandbox_id)
        return instance

    def exec(self, cmd: str, timeout: Optional[int] = None) -> str:
        """Execute shell command in sandbox."""
        return self._exec("bash", "-c", cmd, timeout=timeout)

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

    def write_file(self, path: str, content: str) -> None:
        """Write a file to the sandbox, creating parent directories as needed."""
        if not self._sandbox:
            raise RuntimeError("Sandbox not started")

        # Ensure parent directories exist
        parent_dir = "/".join(path.split("/")[:-1])
        if parent_dir:
            self.ensure_dir(parent_dir)

        # Write the file
        with self._sandbox.open(path, "w") as f:
            f.write(content)

    def ensure_dir(self, path: str) -> None:
        """Ensure a directory exists, creating parent directories as needed."""
        if not self._sandbox:
            raise RuntimeError("Sandbox not started")

        # Build list of directories to create from root to leaf
        dirs_to_create = []
        current = path
        while current and current != "/":
            dirs_to_create.append(current)
            current = "/".join(current.split("/")[:-1])

        # Create directories from root to leaf
        for d in reversed(dirs_to_create):
            try:
                self._sandbox.mkdir(d)
            except Exception:
                pass  # Directory already exists

    def snapshot(self, description: str = "") -> modal.Image:
        """
        Create a filesystem snapshot of the sandbox's current state.

        The snapshot captures all filesystem changes since the sandbox started.
        You can later create a new sandbox from this snapshot to restore the state.

        Args:
            description: Optional description for the snapshot

        Returns:
            modal.Image that can be passed to Sandbox.from_snapshot()

        Example:
            # Work in sandbox
            sandbox.exec("pip install some-package")
            sandbox.exec("echo 'test' > /data.txt")

            # Save state
            snapshot = sandbox.snapshot("After installing packages")

            # Later, restore from snapshot
            new_sandbox = Sandbox.from_snapshot(snapshot, config)
        """
        if not self._sandbox:
            raise RuntimeError("Sandbox not started - cannot create snapshot")

        logger.info(f"Creating filesystem snapshot{f': {description}' if description else ''}...")
        image = self._sandbox.snapshot_filesystem()
        logger.info(f"✓ Snapshot created")

        return image

    def terminate(self, save_snapshot: bool = False, snapshot_description: str = "") -> Optional[modal.Image]:
        """
        Terminate the sandbox, optionally saving a snapshot first.

        Args:
            save_snapshot: If True, create a snapshot before terminating
            snapshot_description: Description for the snapshot

        Returns:
            modal.Image if save_snapshot=True, otherwise None
        """
        snapshot_image = None

        if save_snapshot and self._sandbox:
            snapshot_image = self.snapshot(snapshot_description)

        if self._sandbox:
            try:
                self._sandbox.terminate()
            except (KeyboardInterrupt, Exception) as e:
                # If termination fails (e.g., due to Ctrl+C), log and continue
                logger.debug(f"Sandbox termination interrupted: {e}")
            finally:
                self._sandbox = None

        if self._image_builder:
            try:
                self._image_builder.cleanup()
            except (KeyboardInterrupt, Exception) as e:
                logger.debug(f"Image builder cleanup interrupted: {e}")

        return snapshot_image

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

    @property
    def model_handles(self) -> list[ModelHandle]:
        """List of prepared model handles."""
        return list(self._model_handles)

    @property
    def repo_handles(self) -> list[RepoHandle]:
        """List of prepared repo handles."""
        return list(self._repo_handles)

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

        # Collect environment variables from .env file and config
        secret_env_vars = self._collect_env_vars()
        env_vars = {**self.config.env, **secret_env_vars}

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
            **({"env": env_vars} if env_vars else {}),
            **({"volumes": self._volumes} if self._volumes else {}),
            **({"encrypted_ports": ports} if ports else {}),
            **({"experimental_options": {"enable_docker": True}} if self.config.docker_in_docker else {}),
        }

        self._sandbox = modal.Sandbox.create(**kwargs)

    def _exec(self, *args, timeout: Optional[int] = None) -> str:
        """Execute command and return stdout."""
        if not self._sandbox:
            raise RuntimeError("Sandbox not started")

        kwargs = {}
        if timeout is not None:
            kwargs['timeout'] = timeout

        p = self._sandbox.exec(*args, **kwargs)
        stdout = p.stdout.read()
        p.wait()

        if p.returncode != 0:
            stderr = p.stderr.read()
            raise RuntimeError(f"Command failed: {stderr}")

        return stdout

    def _setup_models(self):
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

    def _setup_repos(self):
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

    def _collect_env_vars(self) -> dict[str, str]:
        """Collect environment variables from config.secrets names.

        This loads values from os.environ (including .env file) and passes
        them as plain environment variables to the sandbox.
        Modal secrets are NOT used - everything comes from local environment.

        Returns:
            dict: Environment variables to pass to sandbox
        """
        env_vars = {}

        for name in self.config.secrets:
            if name in os.environ:
                env_vars[name] = os.environ[name]
            else:
                logger.warning(f"'{name}' not found in environment, skipping")

        return env_vars

    def _download_models(self):
        """Download all models to volumes."""
        for handle in self._model_handles:
            if handle.is_peft and handle.base_model and not check_model_in_volume(self, handle.base_model_path):
                logger.info(f"Downloading base: {handle.base_model}")
                download_model_to_volume(self, handle.base_model, handle.base_model_path)

            if not check_model_in_volume(self, handle.volume_path):
                model_name = "model" if handle.hidden else handle.name
                logger.info(f"Downloading: {model_name}")
                download_model_to_volume(self, handle.name, handle.volume_path)

    def _clone_repos(self):
        """Clone all repos."""
        for handle in self._repo_handles:
            logger.info(f"Cloning: {handle.url}")
            self.exec(f"git clone {handle.url} {handle.local_path}")
            logger.info(f"  ✓ Cloned to {handle.local_path}")
            if handle.install:
                logger.info(f"  Installing dependencies for repo {handle.url}...")
                try:
                    self.exec(f"cd {handle.local_path} && {handle.install}", timeout=120)
                    logger.info(f"  ✓ Dependencies installed")
                except Exception as e:
                    error_msg = str(e)
                    if "timeout" in error_msg.lower():
                        logger.warning(f"Installation timed out for repo {handle.url}, continuing...")
                    else:
                        logger.warning(f"Installation failed for repo {handle.url}, continuing...")
                    logger.debug(f"Error: {error_msg}")
    def _start_services(self):
        """Start execution mode services."""
        if self.config.execution_mode == ExecutionMode.NOTEBOOK:
            start_jupyter(self, self.config.jupyter_port)
            self._jupyter_url = self._sandbox.tunnels()[self.config.jupyter_port].url
            wait_for_service(f"{self._jupyter_url}/api/scribe/health")
            logger.info(f"Jupyter: {self._jupyter_url}")

        if self.config.debug:
            start_code_server(self, self.config.debug_port)
            self._code_server_url = self._sandbox.tunnels()[self.config.debug_port].url
            wait_for_service(f"{self._code_server_url}/healthz")
            logger.info(f"Code-server: {self._code_server_url}")
