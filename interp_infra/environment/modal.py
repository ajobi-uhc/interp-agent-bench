"""Modal environment setup - sandbox creation, model downloads, repo cloning."""

import modal
import requests
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from ..config.schema import GPUConfig, ImageConfig, ExperimentConfig
from .image import ModalImageBuilder


@dataclass
class EnvironmentHandle:
    """Handle to a deployed environment (GPU sandbox + Jupyter server)."""
    sandbox_id: str
    jupyter_url: str
    jupyter_port: int
    jupyter_token: str
    status: str
    sandbox: modal.Sandbox
    experiment_config: ExperimentConfig
    _client: 'ModalEnvironment'


class ModalEnvironment:
    """Manages Modal GPU environments (sandboxes, models, repos)."""

    def __init__(self):
        """Initialize Modal client."""
        self._active_sandboxes: Dict[str, tuple] = {}  # sandbox_id -> (sandbox, tunnel)

    def _infra_prewarm(
        self,
        sandbox: modal.Sandbox,
        experiment_config: ExperimentConfig,
        model_paths: Dict[str, str] = None
    ) -> None:
        """
        Pre-warm infrastructure by downloading models and cloning repos.

        Args:
            sandbox: Modal sandbox to run commands in
            experiment_config: Experiment configuration
            model_paths: Optional mapping of model_id -> volume mount path
        """
        print("Preparing environment...")

        if model_paths is None:
            model_paths = {}

        # 1. Download models (if specified)
        models = experiment_config.environment.models
        if models:
            print(f"  Downloading {len(models)} model(s)...")

            for i, model_config in enumerate(models, 1):
                # Skip custom load code (can't predict dependencies)
                if model_config.custom_load_code:
                    print(f"    [{i}/{len(models)}] Skipping (custom load code)")
                    continue

                model_id = model_config.name

                # Log message (respect obfuscation)
                if experiment_config.execution.obfuscate:
                    if model_config.is_peft:
                        print(f"    [{i}/{len(models)}] Downloading adapter...")
                    else:
                        print(f"    [{i}/{len(models)}] Downloading model...")
                else:
                    print(f"    [{i}/{len(models)}] Downloading {model_id}...")

                try:
                    # Download PEFT base model + adapter
                    if model_config.is_peft:
                        if not model_config.base_model:
                            raise ValueError(f"Model {i}: PEFT adapter requires base_model")

                        # Download base model
                        base_volume_path = model_paths.get(model_config.base_model)
                        self._download_model_weights(sandbox, model_config.base_model, base_volume_path)

                        # Download adapter
                        adapter_volume_path = model_paths.get(model_id)
                        self._download_model_weights(sandbox, model_id, adapter_volume_path)
                    else:
                        # Download regular model
                        volume_path = model_paths.get(model_id)
                        self._download_model_weights(sandbox, model_id, volume_path)

                except Exception as e:
                    error_msg = f"Failed to download model {i}/{len(models)}: {model_id}"
                    print(f"    ERROR: {error_msg}")
                    print(f"    Details: {str(e)}")
                    raise RuntimeError(error_msg) from e

            print(f"  Model downloads complete")

        # 2. Clone infrastructure-level repos (from environment.github_repos)
        github_repos = experiment_config.environment.github_repos
        if github_repos:
            print(f"  Cloning {len(github_repos)} repo(s)...")
            for i, repo in enumerate(github_repos, 1):
                try:
                    print(f"    [{i}/{len(github_repos)}] {repo}")
                    self._clone_repo(sandbox, repo)
                except Exception as e:
                    error_msg = f"Failed to clone repo {i}/{len(github_repos)}: {repo}"
                    print(f"    ERROR: {error_msg}")
                    print(f"    Details: {str(e)}")
                    raise RuntimeError(error_msg) from e
            print(f"  Repos cloned")

        print("Infrastructure ready\n")

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

# Inject compose.yaml for Inspect AI Docker builds (if it doesn't exist)
compose_path = repo_path / "compose.yaml"
if not compose_path.exists():
    compose_content = '''services:
  default:
    build:
      context: .
      network: host
    network_mode: host
    init: true
    security_opt:
      - no-new-privileges:false
    command: tail -f /dev/null
'''
    compose_path.write_text(compose_content)
    print("Added compose.yaml for Docker-in-Modal compatibility")
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

    def create_sandbox(
        self,
        name: str,
        image: modal.Image,
        gpu_config: Optional[GPUConfig],
        experiment_config: ExperimentConfig,
        jupyter_port: int = 8888,
    ) -> EnvironmentHandle:
        """
        Create a Modal Sandbox with Jupyter server and optional GPU.

        This is Stage 1: Infrastructure setup only.
        - Creates sandbox
        - Downloads models to disk
        - Clones repos
        - Starts Jupyter server

        Does NOT create sessions or load models into memory (that's Stage 2).

        Args:
            name: Sandbox name (for tracking)
            image: Modal Image to use
            gpu_config: GPU configuration (None for CPU-only)
            experiment_config: Complete experiment config (serialized to env var)
            jupyter_port: Port for Jupyter server (default 8888)

        Returns:
            EnvironmentHandle with connection details
        """
        # Set builder version if Docker is enabled
        if experiment_config.environment.image.enable_docker:
            import os
            os.environ["MODAL_IMAGE_BUILDER_VERSION"] = "2025.06"

        # Map GPU type string to Modal GPU config (or None for CPU-only)
        gpu = self._get_modal_gpu(gpu_config) if gpu_config else None

        # Create the sandbox
        print(f"Creating Modal Sandbox: {name}")
        if gpu_config:
            print(f"  GPU: {gpu_config.gpu_type} x{gpu_config.gpu_count}")
        else:
            print(f"  Mode: CPU-only")
        if experiment_config.environment.models:
            print(f"  Models: {len(experiment_config.environment.models)} to load")
        if experiment_config.harness.skills:
            print(f"  Skills: {', '.join(experiment_config.harness.skills)}")

        # Create or lookup an App for the sandbox
        app = modal.App.lookup(name, create_if_missing=True)

        # Setup volumes if requested
        volumes_dict = {}
        model_paths = {}  # model_id -> mount_path mapping

        if gpu_config and gpu_config.use_model_volumes:
            print("Setting up model volumes...")

            for model_config in experiment_config.environment.models:
                if model_config.custom_load_code:
                    continue  # Skip custom code

                # Create volume for base model if PEFT
                if model_config.is_peft and model_config.base_model:
                    model_id = model_config.base_model
                    volume_name = f"model--{model_id.replace('/', '--')}"
                    mount_path = f"/models/{model_id.replace('/', '--')}"

                    print(f"  Volume: {volume_name}")
                    volume = modal.Volume.from_name(volume_name, create_if_missing=True)
                    volumes_dict[mount_path] = volume
                    model_paths[model_id] = mount_path

                # Create volume for main model/adapter
                model_id = model_config.name
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

        # Parent process: Store config in env var, start Docker (if enabled), start Jupyter
        docker_startup = ""
        if experiment_config.environment.image.enable_docker:
            docker_startup = """
# Start Docker daemon in background
print("Starting Docker daemon...")
import subprocess
dockerd_proc = subprocess.Popen(['/start-dockerd.sh'])

# Wait for Docker to be ready
import time
for i in range(30):
    try:
        result = subprocess.run(['docker', 'info'], capture_output=True, timeout=5)
        if result.returncode == 0:
            print("Docker daemon ready")
            break
    except:
        pass
    time.sleep(1)
else:
    print("Warning: Docker daemon may not be fully ready")
"""

        startup_script = f"""
import os
import sys

{docker_startup}

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

        # Collect secrets based on recipe type
        secrets = []

        # Always add HuggingFace token (if available)
        try:
            secrets.append(modal.Secret.from_name("huggingface-secret"))
        except Exception:
            # HF secret not configured, skip it
            pass

        # Add Modal auth for nested sandboxes (needed for Inspect Modal sandbox provider)
        import os
        from dotenv import load_dotenv
        load_dotenv()  # Load .env file to get Modal credentials

        if os.getenv("MODAL_TOKEN_ID") and os.getenv("MODAL_TOKEN_SECRET"):
            secrets.append(modal.Secret.from_dict({
                "MODAL_TOKEN_ID": os.environ["MODAL_TOKEN_ID"],
                "MODAL_TOKEN_SECRET": os.environ["MODAL_TOKEN_SECRET"],
            }))
            print("  Passing Modal credentials for nested sandbox support")

        # Pass API keys if using api-access skill (typically for API-only experiments)
        if "api-access" in experiment_config.harness.skills:
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

        # Pass config for nested Modal sandboxes (Inspect integration)
        import json
        sandbox_env = {}
        if experiment_config.environment.image.system_packages:
            sandbox_env["MODAL_SANDBOX_SYSTEM_PACKAGES"] = json.dumps(
                experiment_config.environment.image.system_packages
            )
        if experiment_config.environment.github_repos:
            sandbox_env["MODAL_SANDBOX_GITHUB_REPOS"] = json.dumps(
                experiment_config.environment.github_repos
            )

        sandbox_kwargs = {
            "image": image,
            "gpu": gpu,
            "timeout": 3600 * 24,  # 24 hour timeout
            "app": app,
            "encrypted_ports": [jupyter_port],  # Public HTTPS tunnel to Jupyter
            "secrets": secrets,
            "env": sandbox_env,
        }

        if volumes_dict:
            sandbox_kwargs["volumes"] = volumes_dict

        # Enable Docker if configured
        if experiment_config.environment.image.enable_docker:
            sandbox_kwargs["experimental_options"] = {"enable_docker": True}
            print("  Docker-in-Sandbox: enabled")

        sandbox = modal.Sandbox.create(
            "python",
            "-c",
            startup_script,
            **sandbox_kwargs,
        )

        print(f"Sandbox created: {sandbox.object_id}")

        # Pre-download model weights (to volumes or HF cache)
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

        # Store sandbox for cleanup
        self._active_sandboxes[sandbox.object_id] = (sandbox, tunnel)

        return EnvironmentHandle(
            sandbox_id=sandbox.object_id,
            jupyter_url=tunnel.url,
            jupyter_port=jupyter_port,
            jupyter_token="",  # No token needed with encrypted_ports
            status="running",
            sandbox=sandbox,
            experiment_config=experiment_config,
            _client=self,
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


def setup_environment(experiment_config: ExperimentConfig) -> EnvironmentHandle:
    """
    Stage 1: Setup environment (infrastructure only).

    Creates Modal sandbox, downloads models to disk, clones repos, starts Jupyter.
    Does NOT create sessions or load models into memory.

    Args:
        experiment_config: Complete experiment configuration

    Returns:
        EnvironmentHandle with sandbox_id and jupyter_url
    """
    client = ModalEnvironment()

    # Build image
    image = client.build_image(
        image_config=experiment_config.environment.image,
        gpu_config=experiment_config.environment.gpu,
    )

    # Create sandbox
    env_handle = client.create_sandbox(
        name=experiment_config.name,
        image=image,
        gpu_config=experiment_config.environment.gpu,
        experiment_config=experiment_config,
    )

    return env_handle
