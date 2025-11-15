"""Modal client for GPU deployment and management."""

import modal
import requests
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from ..config.schema import GPUConfig, ModelConfig, ImageConfig, ExperimentConfig
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

    def _infra_prewarm(self, sandbox: modal.Sandbox, experiment_config: ExperimentConfig) -> None:
        """
        Pre-warm infrastructure: download model weights and clone repos.

        This runs BEFORE the kernel starts, using sandbox.exec to prepare the filesystem:
        - Downloads model weights to HF cache (disk only, no GPU memory)
        - Clones GitHub repos to /workspace
        - Shows progress logs in real-time

        The kernel's recipe.warm_init() will then construct models from the local cache.

        Args:
            sandbox: Modal sandbox to run commands in
            experiment_config: Experiment configuration with models and repos
        """
        print("🧊 Infra prewarm: preparing filesystem...")

        # 1. Download model weights to HF cache
        models_cfg = experiment_config.recipe.extra.get("models", [])
        if models_cfg:
            print(f"   📥 Pre-downloading {len(models_cfg)} model(s) to cache...")

            for i, model_spec in enumerate(models_cfg):
                # Skip custom load code (can't predict what it needs)
                if model_spec.get("custom_load_code"):
                    print(f"   ⏭️  Model {i}: custom load code (skipping prewarm)")
                    continue

                model_id = model_spec["name"]
                is_peft = model_spec.get("is_peft", False)
                obfuscate = experiment_config.recipe.extra.get("obfuscate", False)

                # Download base model for PEFT adapters
                if is_peft:
                    base_model_id = model_spec.get("base_model")
                    if not base_model_id:
                        raise ValueError(f"Model {i}: PEFT adapter requires base_model")

                    # Download base model
                    if obfuscate:
                        print(f"   📦 Downloading base model for adapter {i}...")
                    else:
                        print(f"   📦 Downloading {base_model_id}...")

                    self._download_model_weights(sandbox, base_model_id)

                    # PEFT adapters are lightweight, download in prewarm too
                    if not obfuscate:
                        print(f"   📦 Downloading PEFT adapter {model_id}...")
                    self._download_model_weights(sandbox, model_id)

                else:
                    # Regular model
                    if obfuscate:
                        print(f"   📦 Downloading model {i}...")
                    else:
                        print(f"   📦 Downloading {model_id}...")

                    self._download_model_weights(sandbox, model_id)

            print(f"   ✅ Model weights cached")

        # 2. Clone GitHub repos
        github_repos = experiment_config.github_repos
        if github_repos:
            print(f"   📂 Cloning {len(github_repos)} GitHub repo(s)...")

            for repo in github_repos:
                # Handle both "org/repo" and full URLs
                if not repo.startswith("http"):
                    repo_url = f"https://github.com/{repo}"
                else:
                    repo_url = repo

                repo_name = repo_url.split("/")[-1].replace(".git", "")
                print(f"   🔗 Cloning {repo_name}...")

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
    print("✅ Cloned")
else:
    print("✅ Already exists")
"""

                p = sandbox.exec("python", "-c", script)
                for line in p.stdout:
                    line = line.rstrip()
                    if line:
                        print(f"      {line}")

            print(f"   ✅ Repos cloned to /workspace")

        print("✅ Infra prewarm complete\n")

    def _download_model_weights(self, sandbox: modal.Sandbox, model_id: str) -> None:
        """
        Download model weights to HF cache using snapshot_download.

        This only downloads to disk, no GPU memory is used.
        The kernel will construct model objects from this cache.

        Args:
            sandbox: Modal sandbox to run download in
            model_id: HuggingFace model identifier
        """
        script = f"""
import os
from huggingface_hub import snapshot_download

cache_dir = os.environ.get("HF_HOME", "/root/.cache/huggingface")
token = os.environ.get("HF_TOKEN")

print(f"Downloading to {{cache_dir}}")
snapshot_download(
    "{model_id}",
    cache_dir=cache_dir,
    token=token,
    resume_download=True,
)
print("✅ Download complete")
"""

        p = sandbox.exec("python", "-c", script)
        for line in p.stdout:
            line = line.rstrip()
            if line and not line.startswith("Fetching"):  # Filter verbose HF logs
                print(f"      {line}")

    def _warmup_session(self, tunnel_url: str, sandbox: modal.Sandbox, experiment_name: str = "_warmup") -> str:
        """
        Pre-warm a Jupyter kernel session by creating it and waiting for recipe init to complete.

        This starts a real session that the agent will reuse, ensuring models are constructed
        from the pre-downloaded cache before the agent ever touches the notebook.

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
        print(f"🔥 Pre-warming kernel (constructing models from cache)...")

        # 1) Start a real session
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

        # 2) Poll until kernel finishes initialization
        max_wait = 600  # 10 minutes
        delay = 10

        print(f"   Waiting for kernel to construct models from cache...")

        for i in range(max_wait // delay):
            time.sleep(delay)

            try:
                resp = requests.post(
                    f"{tunnel_url}/api/scribe/exec",
                    json={
                        "session_id": session_id,
                        "code": "_init_success if '_init_success' in globals() else None",
                        "hidden": True,
                    },
                    timeout=10,
                )

                if resp.status_code != 200:
                    continue

                value = self._extract_text_result(resp.json())

                if value == "True":
                    print(f"✅ Models constructed and ready! ({(i+1)*delay}s)")
                    return session_id

                if value == "False":
                    err_resp = requests.post(
                        f"{tunnel_url}/api/scribe/exec",
                        json={"session_id": session_id, "code": "_init_error", "hidden": True},
                        timeout=10,
                    )
                    error_msg = self._extract_text_result(err_resp.json()) if err_resp.status_code == 200 else "Unknown"
                    raise RuntimeError(f"Init failed: {error_msg}")

                # Still initializing
                print(f"   Still constructing models... ({(i+1)*delay}s)")

            except requests.exceptions.RequestException:
                # Kernel busy, keep waiting
                if (i + 1) % 3 == 0:
                    print(f"   Still constructing... ({(i+1)*delay}s)")

        raise TimeoutError(f"Timed out after {max_wait}s")

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
        print(f"🚀 Creating Modal Sandbox: {name}")
        if gpu_config:
            print(f"   GPU: {gpu_config.gpu_type} x{gpu_config.gpu_count}")
        else:
            print(f"   Mode: CPU-only (no GPU)")
        print(f"   Recipe: {experiment_config.recipe.name}")
        print(f"   This may take a few minutes...")

        # Create or lookup an App for the sandbox
        app = modal.App.lookup(name, create_if_missing=True)

        # Serialize config for the container
        import pickle
        import base64
        config_bytes = pickle.dumps(experiment_config)
        config_b64 = base64.b64encode(config_bytes).decode('ascii')

        # Parent process: Store config in env var, create IPython startup file, start Jupyter
        startup_script = f"""
import os
import sys
from pathlib import Path

# Add scribe and interp_infra to Python path
sys.path.insert(0, '/root')

# Store config in environment for kernel to access
os.environ['EXPERIMENT_CONFIG_B64'] = '{config_b64}'

# Create IPython startup directory and file
ipython_dir = Path.home() / '.ipython' / 'profile_default' / 'startup'
ipython_dir.mkdir(parents=True, exist_ok=True)

# Write startup file that runs when kernel starts
startup_code = '''import os, sys, pickle, base64, traceback
sys.path.insert(0, "/root")

_kernel_init_log = []
_kernel_init_log.append("Kernel startup file executing")

try:
    _kernel_init_log.append("Loading config from env...")
    config = pickle.loads(base64.b64decode(os.environ["EXPERIMENT_CONFIG_B64"]))
    _kernel_init_log.append(f"Config loaded: {{config.name}}")

    from interp_infra.gpu.session_init import initialize_session
    from interp_infra.recipes.base import get_recipe

    _kernel_init_log.append("Running initialize_session...")
    ns = initialize_session(config)

    _kernel_init_log.append(f"Getting recipe: {{config.recipe.name}}")
    recipe = get_recipe(config.recipe.name)

    _kernel_init_log.append("Running warm_init (constructing models from cache)...")
    ns.update(recipe.warm_init(config.recipe))

    globals().update(ns)
    _kernel_init_log.append(f"SUCCESS! Loaded: {{list(ns.keys())}}")
    _init_success = True
    print("✅ Recipe initialization complete!")
    print(f"Available objects: {{list(ns.keys())}}")
except Exception as _e:
    _kernel_init_log.append(f"ERROR: {{_e}}")
    _kernel_init_log.append(traceback.format_exc())
    _init_success = False
    _init_error = str(_e)
    print("❌ Recipe initialization FAILED!")
    print(f"Error: {{_e}}")
    print("Run: print('\\\\n'.join(_kernel_init_log)) to see full log")
'''

startup_file = ipython_dir / '00-recipe-init.py'
startup_file.write_text(startup_code)
print(f"✅ Created IPython startup file: {{startup_file}}")

# Start Scribe server
print("🚀 Starting Scribe notebook server...")
from scribe.notebook.notebook_server import ScribeServerApp

app = ScribeServerApp()
app.initialize([
    '--ip=0.0.0.0',
    '--port={jupyter_port}',
    '--ServerApp.token=',
    '--ServerApp.password=',
    '--ServerApp.allow_root=True',
])

print(f"✅ Scribe server starting on port {jupyter_port}")
app.start()
"""

        # Create sandbox with encrypted port forwarding for Jupyter
        sandbox = modal.Sandbox.create(
            "python",
            "-c",
            startup_script,
            image=image,
            gpu=gpu,
            timeout=3600 * 24,  # 24 hour timeout
            app=app,
            encrypted_ports=[jupyter_port],  # Public HTTPS tunnel to Jupyter
            secrets=[modal.Secret.from_name("huggingface-secret")],  # HF_TOKEN for gated models
        )

        print(f"✅ Sandbox created: {sandbox.object_id}")

        # Pre-download model weights to HF cache (before kernel starts)
        self._infra_prewarm(sandbox, experiment_config)

        print("   Starting Jupyter...")

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
                    print(f"✅ Jupyter server ready!")
                    print(f"   Tunnel URL: {tunnel.url}")
                    print(f"   Sandbox ID: {sandbox.object_id}")
                    break
            except Exception:
                if i == max_retries - 1:
                    print(f"⚠️  Jupyter server may not be fully ready yet (timed out after {max_retries * retry_delay}s)")
                    print(f"   Tunnel URL: {tunnel.url}")
                    print(f"   Sandbox ID: {sandbox.object_id}")
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
            print(f"⚠️  Unknown GPU type '{gpu_type_str}', defaulting to A100")
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
                print(f"✅ Sandbox {sandbox_id} terminated")
            except Exception as e:
                print(f"⚠️  Error terminating sandbox: {e}")
            finally:
                del self._active_sandboxes[sandbox_id]
        else:
            # Sandbox not in our dict, try to terminate by ID anyway
            print(f"⚠️  Sandbox {sandbox_id} not in active list, attempting direct termination...")
            try:
                # Get the sandbox object from Modal and terminate it
                sb = modal.Sandbox.from_id(sandbox_id)
                sb.terminate()
                print(f"✅ Sandbox {sandbox_id} terminated")
            except Exception as e:
                print(f"❌ Failed to terminate sandbox {sandbox_id}: {e}")

    def list_sandboxes(self) -> List[str]:
        """
        List active sandbox IDs.

        Returns:
            List of sandbox IDs
        """
        return list(self._active_sandboxes.keys())
