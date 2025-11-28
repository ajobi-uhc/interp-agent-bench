"""Environment orchestration - coordinates infrastructure setup using client functions."""

import modal
import requests
import time
import base64
import json
import os
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

from ..config.schema import GPUConfig, ImageConfig, ExperimentConfig
from .image import ModalImageBuilder
from . import clients


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
    print(f"Building image for {experiment_config.name}...")

    # 

    builder = ModalImageBuilder(experiment_config.environment.image)
    image = builder.build()

    # 2. Prepare volumes and model paths
    volumes_dict, model_paths = _prepare_volumes(experiment_config)

    # 3. Collect secrets
    secrets = _collect_secrets(experiment_config)

    # 4. Create startup script
    jupyter_port = 8888
    startup_script = _prepare_startup_script(
        experiment_config=experiment_config,
        model_paths=model_paths,
        jupyter_port=jupyter_port,
    )

    # 5. Create Modal app
    app = clients.lookup_or_create_app(experiment_config.name)

    # 6. Get GPU config
    gpu = None
    if experiment_config.environment.gpu:
        gpu = clients.get_modal_gpu_string(
            experiment_config.environment.gpu.gpu_type,
            experiment_config.environment.gpu.gpu_count,
        )

    # 7. Prepare environment variables for nested sandboxes
    sandbox_env = _prepare_sandbox_env(experiment_config)

    # 8. Create sandbox
    print(f"Creating Modal Sandbox: {experiment_config.name}")
    if experiment_config.environment.gpu:
        print(f"  GPU: {experiment_config.environment.gpu.gpu_type} x{experiment_config.environment.gpu.gpu_count}")
    else:
        print(f"  Mode: CPU-only")
    if experiment_config.environment.models:
        print(f"  Models: {len(experiment_config.environment.models)} to load")
    if experiment_config.harness.skills:
        print(f"  Skills: {', '.join(experiment_config.harness.skills)}")

    sandbox = clients.create_sandbox(
        startup_script=startup_script,
        image=image,
        gpu=gpu,
        timeout=3600 * 24,  # 24 hour timeout
        app=app,
        encrypted_ports=[jupyter_port],
        secrets=secrets,
        env=sandbox_env,
        volumes=volumes_dict if volumes_dict else None,
    )

    print(f"Sandbox created: {sandbox.object_id}")

    # 9. Pre-warm infrastructure (download models, clone repos)
    _prewarm_infrastructure(sandbox, experiment_config, model_paths, volumes_dict)

    # 10. Wait for Jupyter to be ready
    print("Starting Jupyter...")
    tunnels = clients.get_sandbox_tunnels(sandbox)
    tunnel = tunnels[jupyter_port]
    jupyter_url = tunnel.url

    _wait_for_jupyter(jupyter_url)

    print(f"Jupyter server ready")
    print(f"  Tunnel URL: {jupyter_url}")
    print(f"  Sandbox ID: {sandbox.object_id}")

    return EnvironmentHandle(
        sandbox_id=sandbox.object_id,
        jupyter_url=jupyter_url,
        jupyter_port=jupyter_port,
        jupyter_token="",  # No token needed with encrypted_ports
        status="running",
        sandbox=sandbox,
        experiment_config=experiment_config,
    )


def _prepare_volumes(
    experiment_config: ExperimentConfig
) -> Tuple[Dict[str, modal.Volume], Dict[str, str]]:
    """
    Prepare Modal volumes for model storage.

    Args:
        experiment_config: Experiment configuration

    Returns:
        Tuple of (volumes_dict, model_paths):
        - volumes_dict: mount_path -> Volume
        - model_paths: model_id -> mount_path
    """
    volumes_dict = {}
    model_paths = {}

    gpu_config = experiment_config.environment.gpu
    if not gpu_config or not gpu_config.use_model_volumes:
        return volumes_dict, model_paths

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
            volume = clients.create_volume(volume_name)
            volumes_dict[mount_path] = volume
            model_paths[model_id] = mount_path

        # Create volume for main model/adapter
        model_id = model_config.name
        volume_name = f"model--{model_id.replace('/', '--')}"
        mount_path = f"/models/{model_id.replace('/', '--')}"

        print(f"  Volume: {volume_name}")
        volume = clients.create_volume(volume_name)
        volumes_dict[mount_path] = volume
        model_paths[model_id] = mount_path

    return volumes_dict, model_paths


def _collect_secrets(experiment_config: ExperimentConfig) -> List[modal.Secret]:
    """
    Collect Modal secrets based on experiment config.

    Args:
        experiment_config: Experiment configuration

    Returns:
        List of modal.Secret objects
    """
    secrets = []

    # Always add HuggingFace token (if available)
    try:
        secrets.append(modal.Secret.from_name("huggingface-secret"))
    except Exception:
        pass  # HF secret not configured

    # Add Modal auth for nested sandboxes
    from dotenv import load_dotenv
    load_dotenv()

    if os.getenv("MODAL_TOKEN_ID") and os.getenv("MODAL_TOKEN_SECRET"):
        secrets.append(modal.Secret.from_dict({
            "MODAL_TOKEN_ID": os.environ["MODAL_TOKEN_ID"],
            "MODAL_TOKEN_SECRET": os.environ["MODAL_TOKEN_SECRET"],
        }))
        print("  Passing Modal credentials for nested sandbox support")

    # Pass API keys if using api-access skill
    if "api-access" in experiment_config.harness.skills:
        api_keys = {}
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

    return secrets


def _prepare_startup_script(
    experiment_config: ExperimentConfig,
    model_paths: Dict[str, str],
    jupyter_port: int,
) -> str:
    """
    Prepare the startup script for the sandbox.

    Args:
        experiment_config: Experiment configuration
        model_paths: Mapping of model_id -> volume mount path
        jupyter_port: Port for Jupyter server

    Returns:
        Python script as string
    """
    # Serialize config (base64 encoded JSON)
    config_json = experiment_config.model_dump_json()
    config_b64 = base64.b64encode(config_json.encode('utf-8')).decode('ascii')

    # Serialize model paths
    model_paths_json = json.dumps(model_paths)
    model_paths_b64 = base64.b64encode(model_paths_json.encode('utf-8')).decode('ascii')

    return f"""
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


def _prepare_sandbox_env(experiment_config: ExperimentConfig) -> Dict[str, str]:
    """
    Prepare environment variables for nested Modal sandboxes.

    Args:
        experiment_config: Experiment configuration

    Returns:
        Dict of environment variables
    """
    sandbox_env = {}

    if experiment_config.environment.image.system_packages:
        sandbox_env["MODAL_SANDBOX_SYSTEM_PACKAGES"] = json.dumps(
            experiment_config.environment.image.system_packages
        )

    if experiment_config.environment.github_repos:
        sandbox_env["MODAL_SANDBOX_GITHUB_REPOS"] = json.dumps(
            experiment_config.environment.github_repos
        )

    return sandbox_env


def _prewarm_infrastructure(
    sandbox: modal.Sandbox,
    experiment_config: ExperimentConfig,
    model_paths: Dict[str, str],
    volumes_dict: Dict[str, modal.Volume],
) -> None:
    """
    Pre-warm infrastructure by downloading models and cloning repos.

    Args:
        sandbox: Modal sandbox
        experiment_config: Experiment configuration
        model_paths: Mapping of model_id -> volume mount path
        volumes_dict: Mapping of mount_path -> Volume
    """
    print("Preparing environment...")

    # 1. Download models
    models = experiment_config.environment.models
    if models:
        print(f"  Downloading {len(models)} model(s)...")

        for i, model_config in enumerate(models, 1):
            # Skip custom load code
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
                    _download_model_weights(sandbox, model_config.base_model, base_volume_path)

                    # Download adapter
                    adapter_volume_path = model_paths.get(model_id)
                    _download_model_weights(sandbox, model_id, adapter_volume_path)
                else:
                    # Download regular model
                    volume_path = model_paths.get(model_id)
                    _download_model_weights(sandbox, model_id, volume_path)

            except Exception as e:
                error_msg = f"Failed to download model {i}/{len(models)}: {model_id}"
                print(f"    ERROR: {error_msg}")
                print(f"    Details: {str(e)}")
                raise RuntimeError(error_msg) from e

        print(f"  Model downloads complete")

    # 2. Clone repos
    github_repos = experiment_config.environment.github_repos
    if github_repos:
        print(f"  Cloning {len(github_repos)} repo(s)...")
        for i, repo in enumerate(github_repos, 1):
            try:
                print(f"    [{i}/{len(github_repos)}] {repo}")
                _clone_repo(sandbox, repo)
            except Exception as e:
                error_msg = f"Failed to clone repo {i}/{len(github_repos)}: {repo}"
                print(f"    ERROR: {error_msg}")
                print(f"    Details: {str(e)}")
                raise RuntimeError(error_msg) from e
        print(f"  Repos cloned")

    # 3. Commit volumes if models were downloaded to them
    if volumes_dict and model_paths:
        print(f"  Committing volume changes...")
        for mount_path, volume in volumes_dict.items():
            try:
                volume.commit()
                print(f"    ✓ Committed {mount_path}")
            except Exception as e:
                print(f"    Warning: Failed to commit volume at {mount_path}: {e}")

    print("Infrastructure ready\n")


def _download_model_weights(
    sandbox: modal.Sandbox,
    model_id: str,
    volume_path: Optional[str] = None
) -> None:
    """
    Download model weights to volume or HF cache.

    Args:
        sandbox: Modal sandbox
        model_id: HuggingFace model identifier
        volume_path: Optional volume mount path for persistent storage
    """
    if volume_path:
        # Volume mode: check if model exists, download if missing
        download_script = f"""
import os
from pathlib import Path
from huggingface_hub import snapshot_download

model_path = Path("{volume_path}")
model_id = "{model_id}"
token = os.environ.get("HF_TOKEN")

# Check if model already exists
if (model_path / "config.json").exists():
    print("Model found in volume")
else:
    print(f"Model not found in volume, downloading {{model_id}} to {{model_path}}...")
    model_path.mkdir(parents=True, exist_ok=True)

    # Download to volume
    snapshot_download(
        model_id,
        local_dir=str(model_path),
        token=token,
        resume_download=True,
    )
    print("Download complete, model saved to volume")
"""
    else:
        # No volume: download to ephemeral HF cache
        download_script = f"""
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

    p = clients.exec_in_sandbox(sandbox, "python", "-c", download_script)

    for line in p.stdout:
        line = line.rstrip()
        if line and not line.startswith("Fetching"):
            print(f"      {line}")

    # Check for errors
    p.wait()
    if p.returncode != 0:
        stderr_output = "".join(p.stderr)
        raise RuntimeError(f"Failed to download model: {stderr_output}")


def _clone_repo(sandbox: modal.Sandbox, repo: str) -> None:
    """
    Clone a GitHub repository to /workspace.

    Args:
        sandbox: Modal sandbox
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

    p = clients.exec_in_sandbox(sandbox, "python", "-c", script)

    # Print stdout
    for line in p.stdout:
        line = line.rstrip()
        if line:
            print(f"      {line}")

    # Collect stderr
    stderr_lines = []
    for line in p.stderr:
        line = line.rstrip()
        if line:
            stderr_lines.append(line)

    # Wait for process to complete
    p.wait()

    if p.returncode != 0:
        for line in stderr_lines:
            print(f"      ERROR: {line}")
        raise RuntimeError(f"Failed to clone repo {repo}: exit code {p.returncode}")


def _wait_for_jupyter(jupyter_url: str, max_retries: int = 100, retry_delay: int = 4) -> None:
    """
    Wait for Jupyter server to be ready.

    Args:
        jupyter_url: Jupyter server URL
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
    """
    for i in range(max_retries):
        try:
            response = requests.get(f"{jupyter_url}/api/scribe/health", timeout=5)
            if response.status_code == 200:
                return
        except Exception:
            if i == max_retries - 1:
                print(f"Warning: Jupyter server may not be fully ready yet (timed out after {max_retries * retry_delay}s)")
                return
            time.sleep(retry_delay)


def terminate_environment(sandbox_id: str) -> None:
    """
    Terminate a Modal sandbox by ID.

    Args:
        sandbox_id: Sandbox ID to terminate
    """
    try:
        sandbox = clients.get_sandbox_from_id(sandbox_id)
        clients.terminate_sandbox(sandbox)
        print(f"Sandbox {sandbox_id} terminated")
    except Exception as e:
        print(f"Error: Failed to terminate sandbox {sandbox_id}: {e}")
