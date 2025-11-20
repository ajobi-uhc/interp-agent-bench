"""Modal client for GPU deployment and management."""

import modal
import requests
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from ..config.schema import GPUConfig, ImageConfig, ExperimentConfig
from .modal_image_builder import ModalImageBuilder


@dataclass
class ModalDeploymentInfo:
    """Information about a deployed Modal Sandbox."""
    sandbox_id: str
    jupyter_url: str  # Connect URL
    jupyter_port: int  # Port number
    jupyter_token: str  # Authentication token
    status: str
    sandbox: modal.Sandbox  # Keep reference to sandbox
    session_id: Optional[str] = None  # Pre-warmed session ID (if available)


class ModalClient:
    """Client for deploying and managing Modal GPU Sandboxes with Jupyter."""

    def __init__(self):
        """Initialize Modal client."""
        # Modal will use the token from ~/.modal.toml or MODAL_TOKEN_ID/MODAL_TOKEN_SECRET
        self._active_sandboxes: Dict[str, tuple] = {}  # sandbox_id -> (sandbox, tunnel, session_id)

    def _extract_text_result(self, exec_result: Dict[str, Any]) -> Optional[str]:
        """Extract text/plain from raw /api/scribe/exec response."""
        outputs = exec_result.get("outputs", [])
        if not outputs:
            return None

        for output in outputs:
            output_type = output.get("output_type", output.get("type"))

            if output_type == "execute_result" and "text/plain" in output.get("data", {}):
                return output["data"]["text/plain"].strip().strip("'\"")

            if output_type == "stream":
                return output.get("text", "").strip()

        return None

    def _infra_prewarm(
        self,
        sandbox: modal.Sandbox,
        experiment_config: ExperimentConfig,
        model_paths: Dict[str, str] = None
    ) -> None:
        """
        Pre-warm infrastructure by executing recipe-defined tasks.

        Clean separation:
        - Recipe decides WHAT to prepare (via get_prewarm_plan)
        - ModalClient decides HOW to prepare (handlers below)

        No fallbacks. All task kinds must have explicit handlers.

        Args:
            sandbox: Modal sandbox to run commands in
            experiment_config: Experiment configuration
            model_paths: Optional mapping of model_id -> volume mount path
        """
        from ..environment.base import get_environment

        print("Preparing environment...")

        if model_paths is None:
            model_paths = {}

        # 1. Get prewarm plan from environment
        environment = get_environment(experiment_config.environment.name)
        prewarm_plan = environment.get_prewarm_plan(experiment_config.environment)

        # 2. Execute prewarm tasks via handlers
        obfuscate = experiment_config.environment.extra.get("obfuscate", False)

        if prewarm_plan:
            print(f"  Executing {len(prewarm_plan)} prewarm task(s)...")

            for i, task in enumerate(prewarm_plan, 1):
                try:
                    if task.kind == "model":
                        if obfuscate:
                            purpose = task.extra.get("purpose", "model")
                            print(f"    [{i}/{len(prewarm_plan)}] Downloading {purpose}...")
                        else:
                            print(f"    [{i}/{len(prewarm_plan)}] Downloading {task.id}...")

                        volume_path = model_paths.get(task.id)
                        self._download_model_weights(sandbox, task.id, volume_path)

                    elif task.kind == "repo":
                        print(f"    [{i}/{len(prewarm_plan)}] Cloning {task.id}...")
                        self._clone_repo(sandbox, task.id)

                    else:
                        print(f"    Warning: Unknown task kind '{task.kind}', skipping")

                except Exception as e:
                    error_msg = f"Failed to execute prewarm task {i}/{len(prewarm_plan)} ({task.kind}: {task.id})"
                    print(f"    ERROR: {error_msg}")
                    print(f"    Details: {str(e)}")
                    raise RuntimeError(error_msg) from e

            print(f"  Prewarm complete")

        # 3. Clone infrastructure-level repos (from config.github_repos)
        github_repos = experiment_config.github_repos
        if github_repos:
            print(f"  Cloning {len(github_repos)} infrastructure repo(s)...")
            for i, repo in enumerate(github_repos, 1):
                try:
                    print(f"    [{i}/{len(github_repos)}] {repo}")
                    self._clone_repo(sandbox, repo)
                except Exception as e:
                    error_msg = f"Failed to clone infrastructure repo {i}/{len(github_repos)}: {repo}"
                    print(f"    ERROR: {error_msg}")
                    print(f"    Details: {str(e)}")
                    raise RuntimeError(error_msg) from e
            print(f"  Infrastructure repos cloned")

        print("Environment ready\n")

    def _download_model_weights(
        self,
        sandbox: modal.Sandbox,
        model_id: str,
        volume_path: Optional[str] = None
    ) -> None:
        """
        Load model weights from volume or download to HF cache.

        If volume_path is provided, checks that model exists in volume (read-only).
        Otherwise, downloads to HF cache (ephemeral).

        Args:
            sandbox: Modal sandbox to run download in
            model_id: HuggingFace model identifier
            volume_path: Optional volume mount path to check (read-only)
        """
        if volume_path:
            # Volume mode: verify model exists (read-only, no downloads)
            check_script = f"""
from pathlib import Path
model_path = Path("{volume_path}")
if (model_path / "config.json").exists():
    print("Model found in volume")
else:
    raise FileNotFoundError(
        f"Model not found in volume at {volume_path}. "
        f"Pre-populate volume using: python scripts/download_model_to_volume.py --model-id {model_id}"
    )
"""
            p = sandbox.exec("python", "-c", check_script)
            for line in p.stdout:
                line = line.rstrip()
                if line:
                    print(f"      {line}")
        else:
            # No volume: download to ephemeral HF cache
            script = f"""
import os
from huggingface_hub import snapshot_download

token = os.environ.get("HF_TOKEN")
print("Downloading to HF cache")
snapshot_download(
    "{model_id}",
    token=token,
    resume_download=True,
)
print("Download complete")
"""
            p = sandbox.exec("python", "-c", script)
            for line in p.stdout:
                line = line.rstrip()
                if line and not line.startswith("Fetching"):  # Filter verbose HF logs
                    print(f"      {line}")

    def _clone_repo(self, sandbox: modal.Sandbox, repo: str) -> None:
        """
        Clone a GitHub repository to /workspace.

        Args:
            sandbox: Modal sandbox to run clone in
            repo: Repo identifier ("org/repo") or full URL
        """
        if not repo.startswith("http"):
            repo_url = f"https://github.com/{repo}"
        else:
            repo_url = repo

        repo_name = repo_url.split("/")[-1].replace(".git", "")

        script = f"""
import subprocess
from pathlib import Path

workspace = Path("/workspace")
workspace.mkdir(exist_ok=True)
repo_path = workspace / "{repo_name}"

if not repo_path.exists():
    subprocess.run(
        ["git", "clone", "{repo_url}", str(repo_path)],
        check=True,
        capture_output=False
    )
    print("Cloned")
else:
    print("Already exists")
"""

        p = sandbox.exec("python", "-c", script)

        # Print stdout
        for line in p.stdout:
            line = line.rstrip()
            if line:
                print(f"      {line}")

        # Print stderr (git outputs progress here, not just errors)
        stderr_lines = []
        for line in p.stderr:
            line = line.rstrip()
            if line:
                stderr_lines.append(line)

        # Wait for process to complete
        p.wait()

        # Check if the command failed (check=True in subprocess.run will raise if git clone fails)
        if p.returncode != 0:
            # Print stderr on failure
            for line in stderr_lines:
                print(f"      ERROR: {line}")
            raise RuntimeError(f"Failed to clone repo {repo}: exit code {p.returncode}")

    def _warmup_session(self, tunnel_url: str, sandbox: modal.Sandbox, experiment_name: str = "_warmup") -> str:
        """
        Pre-warm a Jupyter kernel by explicitly executing initialization code.

        This starts a session and explicitly runs recipe initialization via the Scribe API,
        ensuring models are constructed from cache before the agent uses the kernel.

        Args:
            tunnel_url: The Jupyter tunnel URL
            sandbox: The Modal sandbox (unused, for future extensibility)
            experiment_name: Name for the warmup notebook

        Returns:
            session_id of the pre-warmed session

        Raises:
            RuntimeError: If initialization fails
            TimeoutError: If warmup takes too long
        """
        print(f"Pre-warming kernel...")

        # 1. Start a fresh kernel session
        try:
            response = requests.post(
                f"{tunnel_url}/api/scribe/start",
                json={"experiment_name": experiment_name},
                timeout=30,
            )
            response.raise_for_status()
            session_id = response.json()["session_id"]
            print(f"   Started warmup session: {session_id}")
        except Exception as e:
            raise RuntimeError(f"Failed to start warmup session: {e}")

        # 2. Explicitly execute initialization code in the kernel
        init_code = """
import os, sys, base64, traceback
import time

# Ensure paths are set
sys.path.insert(0, "/root")

_start_time = time.time()

print("Loading experiment config...")
from interp_infra.config.schema import ExperimentConfig
config_json = base64.b64decode(os.environ["EXPERIMENT_CONFIG_B64"]).decode('utf-8')
config = ExperimentConfig.model_validate_json(config_json)
print(f"  Experiment: {config.name}")
print(f"  Environment: {config.environment.name}")
print()

# Run setup pipeline
from interp_infra.gpu.setup_pipeline import create_namespace
namespace = create_namespace(config)

# Inject into globals
globals().update(namespace)

_total_time = time.time() - _start_time
print(f"Total initialization time: {_total_time:.1f}s")
"""

        print(f"   Executing initialization code in kernel...")

        try:
            response = requests.post(
                f"{tunnel_url}/api/scribe/exec",
                json={
                    "session_id": session_id,
                    "code": init_code,
                    "hidden": False,
                },
                timeout=600,  # 10 minute timeout for model loading
            )
            response.raise_for_status()
            result = response.json()

            # Print all output from the kernel
            for output in result.get("outputs", []):
                output_type = output.get("output_type", output.get("type"))

                if output_type == "stream":
                    text = output.get("text", "")
                    if text:
                        for line in text.rstrip().split('\n'):
                            print(f"   {line}")

                elif output_type == "error":
                    ename = output.get("ename", "Error")
                    evalue = output.get("evalue", "")
                    traceback_lines = output.get("traceback", [])
                    print(f"   Error: {ename}: {evalue}")
                    for line in traceback_lines:
                        # Strip ANSI codes from traceback
                        clean_line = line.replace('\x1b[0m', '').replace('\x1b[1m', '').replace('\x1b[31m', '')
                        print(f"   {clean_line}")
                    raise RuntimeError(f"Initialization failed: {ename}: {evalue}")

            print(f"Models constructed and ready")
            return session_id

        except requests.exceptions.Timeout:
            raise TimeoutError(f"Initialization timed out after 10 minutes")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to execute initialization: {e}")

    def build_image(
        self,
        image_config: ImageConfig,
        gpu_config: Optional[GPUConfig],
    ) -> modal.Image:
        """
        Build a Modal Image from configuration.

        Args:
            image_config: Image configuration
            gpu_config: GPU configuration (used for determining CUDA version), None for CPU-only

        Returns:
            modal.Image ready to deploy
        """
        builder = ModalImageBuilder(image_config)
        return builder.build()

    def create_jupyter_sandbox(
        self,
        name: str,
        image: modal.Image,
        gpu_config: Optional[GPUConfig],
        experiment_config: ExperimentConfig,
        jupyter_port: int = 8888,  # Standard Jupyter port
    ) -> ModalDeploymentInfo:
        """
        Create a Modal Sandbox with Jupyter server and optional GPU.

        Uses the recipe pattern for composable environment setup:
        - Parent process: Stores config in env var, starts Jupyter
        - Kernel process: Runs initialize_session() + recipe.warm_init()

        Args:
            name: Sandbox name (for tracking)
            image: Modal Image to use
            gpu_config: GPU configuration (None for CPU-only)
            experiment_config: Complete experiment config (serialized to env var)
            jupyter_port: Port for Jupyter server (default 8888)

        Returns:
            ModalDeploymentInfo with connection details
        """
        # Map GPU type string to Modal GPU config (or None for CPU-only)
        gpu = self._get_modal_gpu(gpu_config) if gpu_config else None

        # Create the sandbox
        print(f"Creating Modal Sandbox: {name}")
        if gpu_config:
            print(f"  GPU: {gpu_config.gpu_type} x{gpu_config.gpu_count}")
        else:
            print(f"  Mode: CPU-only")
        print(f"  Environment: {experiment_config.environment.name}")

        # Create or lookup an App for the sandbox
        app = modal.App.lookup(name, create_if_missing=True)

        # Setup volumes if requested
        volumes_dict = {}
        model_paths = {}  # model_id -> mount_path mapping

        if gpu_config and gpu_config.use_model_volumes:
            print("Setting up model volumes...")
            from ..environment.base import get_environment
            environment = get_environment(experiment_config.environment.name)
            prewarm_plan = environment.get_prewarm_plan(experiment_config.environment)

            if prewarm_plan:
                for task in prewarm_plan:
                    if task.kind == "model":
                        model_id = task.id
                        volume_name = f"model--{model_id.replace('/', '--')}"
                        mount_path = f"/models/{model_id.replace('/', '--')}"

                        print(f"  Volume: {volume_name}")
                        volume = modal.Volume.from_name(volume_name, create_if_missing=True)
                        volumes_dict[mount_path] = volume
                        model_paths[model_id] = mount_path

        # Serialize config for the container (using JSON for stability)
        import base64
        import json
        config_json = experiment_config.model_dump_json()
        config_b64 = base64.b64encode(config_json.encode('utf-8')).decode('ascii')

        # Serialize model paths (JSON)
        model_paths_json = json.dumps(model_paths)
        model_paths_b64 = base64.b64encode(model_paths_json.encode('utf-8')).decode('ascii')

        # Parent process: Store config in env var, start Jupyter
        startup_script = f"""
import os
import sys

# Add scribe and interp_infra to Python path
sys.path.insert(0, '/root')

# Store config in environment for kernel to access
os.environ['EXPERIMENT_CONFIG_B64'] = '{config_b64}'
os.environ['MODEL_PATHS_B64'] = '{model_paths_b64}'

# Start Scribe server
print("Starting Scribe notebook server...")
from scribe.notebook.notebook_server import ScribeServerApp

app = ScribeServerApp()
app.initialize([
    '--ip=0.0.0.0',
    '--port={jupyter_port}',
    '--ServerApp.token=',
    '--ServerApp.password=',
    '--ServerApp.allow_root=True',
])

print(f"Scribe server starting on port {jupyter_port}")
app.start()
"""

        # Create sandbox with encrypted port forwarding for Jupyter
        # Build kwargs conditionally to avoid passing None

        # Collect secrets based on recipe type
        secrets = []

        # Always add HuggingFace token (if available)
        try:
            secrets.append(modal.Secret.from_name("huggingface-secret"))
        except Exception:
            # HF secret not configured, skip it
            pass

        # For API access environments, pass local API keys
        if experiment_config.environment.name == "api_access":
            import os
            api_keys = {}

            # Collect API keys from local environment
            api_key_names = [
                "OPENROUTER_API_KEY",
                "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY",
                "GOOGLE_API_KEY",
            ]

            for key_name in api_key_names:
                value = os.getenv(key_name)
                if value:
                    api_keys[key_name] = value
                    print(f"  Passing {key_name} to sandbox")

            if api_keys:
                secrets.append(modal.Secret.from_dict(api_keys))
            else:
                print("  Warning: No API keys found in environment")
                print("  Set OPENROUTER_API_KEY or other API keys in your .env file")

        sandbox_kwargs = {
            "image": image,
            "gpu": gpu,
            "timeout": 3600 * 24,  # 24 hour timeout
            "app": app,
            "encrypted_ports": [jupyter_port],  # Public HTTPS tunnel to Jupyter
            "secrets": secrets,
        }

        if volumes_dict:
            sandbox_kwargs["volumes"] = volumes_dict

        sandbox = modal.Sandbox.create(
            "python",
            "-c",
            startup_script,
            **sandbox_kwargs,
        )

        print(f"Sandbox created: {sandbox.object_id}")

        # Pre-download model weights (to volumes or HF cache)
        # Note: volumes auto-persist on sandbox termination, manual commit not needed
        self._infra_prewarm(sandbox, experiment_config, model_paths)

        print("Starting Jupyter...")

        # Get the public tunnel URL (no auth needed with encrypted_ports)
        tunnels = sandbox.tunnels()
        tunnel = tunnels[jupyter_port]

        # Wait for Jupyter to start and health check it
        max_retries = 100
        retry_delay = 4
        for i in range(max_retries):
            try:
                response = requests.get(f"{tunnel.url}/api/scribe/health", timeout=5)
                if response.status_code == 200:
                    print(f"Jupyter server ready")
                    print(f"  Tunnel URL: {tunnel.url}")
                    print(f"  Sandbox ID: {sandbox.object_id}")
                    break
            except Exception:
                if i == max_retries - 1:
                    print(f"Warning: Jupyter server may not be fully ready yet (timed out after {max_retries * retry_delay}s)")
                    print(f"  Tunnel URL: {tunnel.url}")
                    print(f"  Sandbox ID: {sandbox.object_id}")
                    break
                time.sleep(retry_delay)

        # Pre-warm kernel: start a session and wait for recipe initialization
        # This happens BEFORE the agent starts, so the agent gets a warm kernel
        session_id = self._warmup_session(tunnel.url, sandbox, experiment_name=name)

        # Store sandbox with session_id for cleanup
        self._active_sandboxes[sandbox.object_id] = (sandbox, tunnel, session_id)

        return ModalDeploymentInfo(
            sandbox_id=sandbox.object_id,
            jupyter_url=tunnel.url,
            jupyter_port=jupyter_port,
            jupyter_token="",  # No token needed with encrypted_ports
            status="running",
            sandbox=sandbox,
            session_id=session_id,  # Pre-warmed session ready for agent
        )

    def _get_modal_gpu(self, gpu_config: GPUConfig) -> str:
        """
        Get Modal GPU string from config.

        Args:
            gpu_config: GPU configuration

        Returns:
            Modal GPU string like "T4", "A100", etc.
        """
        gpu_type_str = gpu_config.gpu_type
        count = gpu_config.gpu_count

        # Map common GPU types to Modal GPU strings
        if "H100" in gpu_type_str:
            gpu_name = "H100"
        elif "A100" in gpu_type_str:
            gpu_name = "A100"
        elif "A10G" in gpu_type_str:
            gpu_name = "A10G"
        elif "L4" in gpu_type_str:
            gpu_name = "L4"
        elif "T4" in gpu_type_str:
            gpu_name = "T4"
        elif "H200" in gpu_type_str:
            gpu_name = "H200"
        else:
            # Default to A100
            print(f"Warning: Unknown GPU type '{gpu_type_str}', defaulting to A100")
            gpu_name = "A100"

        # Return GPU string with count syntax
        if count > 1:
            return f"{gpu_name}:{count}"
        return gpu_name

    def terminate_sandbox(self, sandbox_id: str):
        """
        Terminate a Modal Sandbox.

        Args:
            sandbox_id: Sandbox ID to terminate
        """
        if sandbox_id in self._active_sandboxes:
            sandbox_info = self._active_sandboxes[sandbox_id]
            # Handle both old (sandbox, tunnel) and new (sandbox, tunnel, session_id) formats
            sandbox = sandbox_info[0]
            # Explicitly terminate the sandbox
            try:
                sandbox.terminate()
                print(f"Sandbox {sandbox_id} terminated")
            except Exception as e:
                print(f"Warning: Error terminating sandbox: {e}")
            finally:
                del self._active_sandboxes[sandbox_id]
        else:
            # Sandbox not in our dict, try to terminate by ID anyway
            print(f"Warning: Sandbox {sandbox_id} not in active list, attempting direct termination...")
            try:
                # Get the sandbox object from Modal and terminate it
                sb = modal.Sandbox.from_id(sandbox_id)
                sb.terminate()
                print(f"Sandbox {sandbox_id} terminated")
            except Exception as e:
                print(f"Error: Failed to terminate sandbox {sandbox_id}: {e}")

    def list_sandboxes(self) -> List[str]:
        """
        List active sandbox IDs.

        Returns:
            List of sandbox IDs
        """
        return list(self._active_sandboxes.keys())
