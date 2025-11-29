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
    CLI = "cli"  # No UI, just programmatic access via exec()
    NOTEBOOK = "notebook"  # Jupyter notebook interface
    MCP = "mcp"  # MCP server mode (not yet implemented)


@dataclass
class SandboxConfig:
    """Configuration for a sandbox."""
    python_packages: list[str] = field(default_factory=list)
    system_packages: list[str] = field(default_factory=list)
    python_version: str = "3.11"
    gpu: Optional[str] = None  # e.g. "H100", "A100"
    gpu_count: int = 1
    docker_in_docker: bool = False
    execution_mode: ExecutionMode = ExecutionMode.CLI
    debug: bool = False  # Start code-server for debugging
    timeout: int = 3600 * 24  # 24 hours
    secrets: list[str] = field(default_factory=list)  # named secrets
    env: dict[str, str] = field(default_factory=dict)
    encrypted_ports: list[int] = field(default_factory=list)  # Additional ports to expose


class Sandbox:
    """
    A Modal sandbox environment.
    
    Handles image building, sandbox creation, model/repo preparation.
    
    Usage:
        config = SandboxConfig(
            python_packages=["torch", "transformers"],
            gpu="H100",
            notebook=True,
        )
        
        sandbox = Sandbox(config)
        model = sandbox.prepare_model("google/gemma-9b")
        sandbox.start(name="my-experiment")
        
        # model is now downloaded to volume, sandbox is running
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
        
        # Build image
        print("  Building image...")
        self._image_builder = ModalImageBuilder(
            python_packages=self.config.python_packages,
            system_packages=self.config.system_packages,
            python_version=self.config.python_version,
            docker_in_docker=self.config.docker_in_docker,
            execution_mode=self.config.execution_mode,
        )
        image = self._image_builder.build()
        
        # Get GPU config
        gpu = None
        if self.config.gpu:
            gpu = f"{self.config.gpu}:{self.config.gpu_count}"
            print(f"  GPU: {self.config.gpu} x{self.config.gpu_count}")
        
        # Create app
        self._app = modal.App.lookup(name, create_if_missing=True)
        
        # Collect secrets
        secrets = self._collect_secrets()
        
        # Prepare sandbox kwargs
        sandbox_kwargs = {
            "image": image,
            "timeout": self.config.timeout,
            "app": self._app,
        }
        
        if gpu:
            sandbox_kwargs["gpu"] = gpu
            
        if secrets:
            sandbox_kwargs["secrets"] = secrets
        
        # Add environment variables
        if self.config.env:
            sandbox_kwargs["env"] = self.config.env
        
        # Add volumes
        if self._volumes:
            sandbox_kwargs["volumes"] = self._volumes
            print(f"  Volumes: {len(self._volumes)}")
            
        # Add docker options if needed
        if self.config.docker_in_docker:
            sandbox_kwargs["experimental_options"] = {"enable_docker": True}

        # Add encrypted ports based on execution mode and debug flag
        encrypted_ports = list(self.config.encrypted_ports)  # Start with config
        if self.config.execution_mode == ExecutionMode.NOTEBOOK:
            encrypted_ports.append(8888)
        if self.config.debug:
            encrypted_ports.append(8080)
        if encrypted_ports:
            sandbox_kwargs["encrypted_ports"] = encrypted_ports

        # Create sandbox
        print("  Creating sandbox...")
        self._sandbox = modal.Sandbox.create(**sandbox_kwargs)
        print(f"  Sandbox ID: {self._sandbox.object_id}")
        
        # Start docker daemon if needed
        if self.config.docker_in_docker:
            print("  Starting docker daemon...")
            self._start_docker_daemon()
        
        # Download any models that were prepared before start
        self._download_prepared_models()

        # Clone any repos that were prepared before start
        self._clone_prepared_repos()

        # Commit volume changes after all downloads
        commit_volumes(self._volumes)
        
        # Start services based on execution mode
        if self.config.execution_mode == ExecutionMode.NOTEBOOK:
            print("  Starting jupyter server...")
            self._start_jupyter()
            tunnels = self._sandbox.tunnels()
            self._jupyter_url = tunnels[8888].url
            self._wait_for_jupyter()
            print(f"  Jupyter URL: {self._jupyter_url}")
        elif self.config.execution_mode == ExecutionMode.MCP:
            print("  MCP mode not yet implemented")
        elif self.config.execution_mode == ExecutionMode.CLI:
            print("  CLI mode - no services started")

        # Start code-server if debug flag is set
        if self.config.debug:
            print("  Debug mode: Starting code-server...")
            self._start_code_server()
            tunnels = self._sandbox.tunnels()
            self._code_server_url = tunnels[8080].url
            self._wait_for_code_server()
            print(f"  Code Server URL: {self._code_server_url}")

        print("  Sandbox ready")
        return self
    
    def exec(self, cmd: str) -> str:
        """
        Execute a shell command in the sandbox.
        
        Args:
            cmd: Shell command to run
            
        Returns:
            stdout from command
        """
        if not self._sandbox:
            raise RuntimeError("Sandbox not started. Call start() first.")
            
        p = self._sandbox.exec("bash", "-c", cmd)
        stdout = p.stdout.read()
        p.wait()
        
        if p.returncode != 0:
            stderr = p.stderr.read()
            raise RuntimeError(f"Command failed (exit {p.returncode}): {stderr}")
            
        return stdout
    
    def exec_python(self, code: str) -> str:
        """
        Execute Python code in the sandbox.
        
        Args:
            code: Python code to run
            
        Returns:
            stdout from execution
        """
        if not self._sandbox:
            raise RuntimeError("Sandbox not started. Call start() first.")
            
        p = self._sandbox.exec("python", "-c", code)
        stdout = p.stdout.read()
        p.wait()
        
        if p.returncode != 0:
            stderr = p.stderr.read()
            raise RuntimeError(f"Python execution failed (exit {p.returncode}): {stderr}")
            
        return stdout
    
    def prepare_model(
        self, 
        name: str, 
        hidden: bool = False,
        is_peft: bool = False,
        base_model: Optional[str] = None,
    ) -> ModelHandle:
        """
        Prepare a model by setting up volume.
        
        Downloads happen when sandbox starts (or immediately if already started).
        
        Args:
            name: HuggingFace model identifier
            hidden: Whether to hide model name from agent
            is_peft: Whether this is a PEFT adapter
            base_model: Base model for PEFT adapters
            
        Returns:
            ModelHandle with volume path
        """
        base_model_path = None
        
        # Handle PEFT base model first
        if is_peft:
            if not base_model:
                raise ValueError("PEFT models require base_model to be specified")
                
            base_volume, base_mount = get_or_create_volume(base_model)
            self._volumes[base_mount] = base_volume
            base_model_path = base_mount
        
        # Get or create volume for main model/adapter
        volume, mount_path = get_or_create_volume(name)
        self._volumes[mount_path] = volume
        
        handle = ModelHandle(
            name=name, 
            volume_path=mount_path, 
            hidden=hidden,
            is_peft=is_peft,
            base_model=base_model,
            base_model_path=base_model_path,
        )
        self._model_handles.append(handle)
        
        # If sandbox already running, download now
        if self._sandbox:
            self._download_model(handle)
        
        return handle
    
    def prepare_repo(
        self,
        url: str,
        dockerfile: Optional[str] = None,
        install: str = False,
    ) -> RepoHandle:
        """
        Prepare a repo for cloning.

        Cloning happens when sandbox starts (or immediately if already started).

        Args:
            url: GitHub repo URL or "org/repo"
            dockerfile: Optional path to Dockerfile to build
            install: Whether to pip install the repo after cloning

        Returns:
            RepoHandle with local path
        """
        if not url.startswith("http"):
            url = f"https://github.com/{url}"

        repo_name = url.split("/")[-1].replace(".git", "")
        local_path = f"/workspace/{repo_name}"

        handle = RepoHandle(
            url=url,
            local_path=local_path,
            dockerfile=dockerfile,
            container_name=repo_name if dockerfile else None,
            install=install,
        )
        self._repo_handles.append(handle)

        # If sandbox already running, clone now
        if self._sandbox:
            self._clone_repo(handle)

        return handle
    
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
            except Exception:
                print(f"  Warning: Secret '{secret_name}' not found")
        
        # Always try HF token
        try:
            secrets.append(modal.Secret.from_name("huggingface-secret"))
        except Exception:
            pass
        
        # Modal credentials for nested sandboxes (needed for IsolatedSandbox)
        if os.getenv("MODAL_TOKEN_ID") and os.getenv("MODAL_TOKEN_SECRET"):
            secrets.append(modal.Secret.from_dict({
                "MODAL_TOKEN_ID": os.environ["MODAL_TOKEN_ID"],
                "MODAL_TOKEN_SECRET": os.environ["MODAL_TOKEN_SECRET"],
            }))

        # Pass common API keys from local environment
        api_keys = {}
        for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"]:
            if os.getenv(key):
                api_keys[key] = os.environ[key]

        if api_keys:
            secrets.append(modal.Secret.from_dict(api_keys))

        return secrets
    
    def _start_docker_daemon(self):
        """Start docker daemon in background."""
        dockerd_script = self._get_dockerd_script()
        
        # Write script to sandbox
        self._sandbox.open("/start-dockerd.sh", "w").write(dockerd_script)
        self.exec("chmod +x /start-dockerd.sh")
        
        # Run in background
        self.exec("nohup /start-dockerd.sh > /var/log/dockerd.log 2>&1 &")
        
        # Wait for docker to be ready
        for i in range(30):
            try:
                self.exec("docker info > /dev/null 2>&1")
                return
            except Exception:
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
        jupyter_script = '''
import sys
sys.path.insert(0, "/root")
from scribe.notebook.notebook_server import ScribeServerApp
app = ScribeServerApp()
app.initialize([
    "--ip=0.0.0.0",
    "--port=8888",
    "--ServerApp.token=",
    "--ServerApp.password=",
    "--ServerApp.allow_root=True",
])
app.start()
'''
        # Escape for shell
        escaped_script = jupyter_script.replace('"', '\\"')
        self.exec(f'nohup python -c "{escaped_script}" > /var/log/jupyter.log 2>&1 &')
    
    def _wait_for_jupyter(self, max_retries: int = 100, retry_delay: float = 2.0):
        """Wait for jupyter server to be ready."""
        for i in range(max_retries):
            try:
                response = requests.get(f"{self._jupyter_url}/api/scribe/health", timeout=5)
                if response.status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(retry_delay)

        print("  Warning: Jupyter may not be fully ready")

    def _start_code_server(self):
        """Start code-server (VS Code in browser)."""
        # Install code-server
        install_script = """
curl -fsSL https://code-server.dev/install.sh | sh > /var/log/code-server-install.log 2>&1
"""
        self.exec(install_script)

        # Start code-server
        self.exec('nohup code-server --bind-addr 0.0.0.0:8080 --auth none /workspace > /var/log/code-server.log 2>&1 &')

    def _wait_for_code_server(self, max_retries: int = 100, retry_delay: float = 2.0):
        """Wait for code-server to be ready."""
        for i in range(max_retries):
            try:
                response = requests.get(f"{self._code_server_url}/healthz", timeout=5)
                if response.status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(retry_delay)

        print("  Warning: Code-server may not be fully ready")
    
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