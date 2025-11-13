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


class ModalClient:
    """Client for deploying and managing Modal GPU Sandboxes with Jupyter."""

    def __init__(self):
        """Initialize Modal client."""
        # Modal will use the token from ~/.modal.toml or MODAL_TOKEN_ID/MODAL_TOKEN_SECRET
        self._active_sandboxes: Dict[str, modal.Sandbox] = {}

    def build_image(
        self,
        image_config: ImageConfig,
        models: List[ModelConfig],
        gpu_config: GPUConfig,
    ) -> modal.Image:
        """
        Build a Modal Image from configuration.

        Args:
            image_config: Image configuration
            models: List of models to load
            gpu_config: GPU configuration (used for determining CUDA version)

        Returns:
            modal.Image ready to deploy
        """
        builder = ModalImageBuilder(image_config, models)
        return builder.build()

    def create_jupyter_sandbox(
        self,
        name: str,
        image: modal.Image,
        gpu_config: GPUConfig,
        experiment_config: ExperimentConfig,
        jupyter_port: int = 8888,  # Standard Jupyter port
    ) -> ModalDeploymentInfo:
        """
        Create a Modal Sandbox with Jupyter server and GPU.

        Setup hooks run synchronously before the session is available to ensure
        models are loaded and environment is ready.

        Args:
            name: Sandbox name (for tracking)
            image: Modal Image to use
            gpu_config: GPU configuration
            experiment_config: Complete experiment config (used to generate setup code)
            jupyter_port: Port for Jupyter server (default 8888)

        Returns:
            ModalDeploymentInfo with connection details
        """
        # Map GPU type string to Modal GPU config
        gpu = self._get_modal_gpu(gpu_config)

        # Create the sandbox
        print(f"🚀 Creating Modal Sandbox: {name}")
        print(f"   GPU: {gpu_config.gpu_type} x{gpu_config.gpu_count}")
        print(f"   This may take a few minutes...")

        # Create or lookup an App for the sandbox
        app = modal.App.lookup(name, create_if_missing=True)

        # Serialize config for the container
        import pickle
        import base64
        config_bytes = pickle.dumps(experiment_config)
        config_b64 = base64.b64encode(config_bytes).decode('ascii')

        # Create startup script to start Jupyter server
        startup_script = f"""
import os
import sys
import pickle
import base64

# Add scribe and interp_infra to Python path
sys.path.insert(0, '/root')

# Deserialize config
config_b64 = '{config_b64}'
config = pickle.loads(base64.b64decode(config_b64))

# Initialize session (load models, clone repos, etc)
# This happens BEFORE Jupyter starts, so agent never sees it
from interp_infra.gpu.session_init import initialize_session
namespace = initialize_session(config)

# Save namespace to disk for kernel to load
os.makedirs('/tmp', exist_ok=True)
with open('/tmp/session_globals.pkl', 'wb') as f:
    pickle.dump(namespace, f)

# Create IPython startup directory and file
os.makedirs('/root/.ipython/profile_default/startup', exist_ok=True)

# Write startup file that loads pre-initialized globals
startup_code = '''
# Load pre-initialized session globals
import pickle
try:
    with open('/tmp/session_globals.pkl', 'rb') as f:
        globals().update(pickle.load(f))
    print('✅ Session ready')
except Exception as e:
    print(f'❌ Failed to load session: {{e}}')
    import traceback
    traceback.print_exc()
'''

with open('/root/.ipython/profile_default/startup/00-session-init.py', 'w') as f:
    f.write(startup_code)

# Start Scribe notebook server
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

print("✅ Scribe server starting on port {jupyter_port}")

# Start the server (this blocks)
app.start()
"""

        # Create sandbox with encrypted port forwarding for Jupyter
        # The entrypoint runs the startup script which starts Jupyter
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
        print("   Starting Jupyter...")

        # Get the public tunnel URL (no auth needed with encrypted_ports)
        tunnels = sandbox.tunnels()
        tunnel = tunnels[jupyter_port]

        # Wait for Jupyter to start and health check it
        max_retries = 30
        retry_delay = 2
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

        # Store sandbox for cleanup
        self._active_sandboxes[sandbox.object_id] = (sandbox, tunnel)

        return ModalDeploymentInfo(
            sandbox_id=sandbox.object_id,
            jupyter_url=tunnel.url,
            jupyter_port=jupyter_port,
            jupyter_token="",  # No token needed with encrypted_ports
            status="running",
            sandbox=sandbox,
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
            sandbox, tunnel = self._active_sandboxes[sandbox_id]
            # Tunnel cleanup happens automatically when sandbox terminates
            sandbox.terminate()
            del self._active_sandboxes[sandbox_id]
            print(f"✅ Sandbox {sandbox_id} terminated")
        else:
            print(f"⚠️  Sandbox {sandbox_id} not found in active sandboxes")

    def list_sandboxes(self) -> List[str]:
        """
        List active sandbox IDs.

        Returns:
            List of sandbox IDs
        """
        return list(self._active_sandboxes.keys())
