"""Modal GPU Jupyter Runner - Run Jupyter notebook server on Modal with GPU access."""

import os
import subprocess
import time
from pathlib import Path

import modal

from scribe.modal.images import hf_image


def create_jupyter_app(
    app_name: str = "scribe-jupyter",
    gpu: str = "A10G",
    timeout: int = 3600,
    image: modal.Image = None,
) -> tuple[modal.App, modal.Function]:
    """Create Modal app with Jupyter function.

    Args:
        app_name: Name for the Modal app
        gpu: GPU type ("A10G", "A100", "H100", "L4", "any")
        timeout: Maximum session timeout in seconds
        image: Modal image to use (defaults to hf_image)

    Returns:
        Tuple of (app, function) for the Jupyter runner
    """
    if image is None:
        image = hf_image

    # Create volume for notebook persistence
    volume = modal.Volume.from_name(
        f"{app_name}-notebooks",
        create_if_missing=True
    )

    app = modal.App(app_name, image=image)

    @app.function(
        gpu=gpu,
        volumes={"/notebooks": volume},
        timeout=timeout,
        secrets=[modal.Secret.from_name("huggingface-secret")],
    )
    def run_jupyter(
        jupyter_token: str,
        notebook_dir: str = "/notebooks",
        port: int = 8888,
        session_timeout: int = 3600,
    ):
        """Run Jupyter notebook server on Modal GPU with tunneling.

        Args:
            jupyter_token: Security token for Jupyter
            notebook_dir: Directory to store notebooks
            port: Port for Jupyter server
            session_timeout: How long to keep server running

        Yields:
            Dictionary with tunnel_url when ready, then keeps server running
        """
        with modal.forward(port) as tunnel:
            # Start Jupyter server
            jupyter_process = subprocess.Popen(
                [
                    "jupyter",
                    "notebook",
                    "--no-browser",
                    "--allow-root",
                    "--ip=0.0.0.0",
                    f"--port={port}",
                    f"--notebook-dir={notebook_dir}",
                    "--NotebookApp.allow_origin='*'",
                    "--NotebookApp.allow_remote_access=1",
                ],
                env={**os.environ, "JUPYTER_TOKEN": jupyter_token},
            )

            # Wait for Jupyter to actually start
            time.sleep(3)

            print(f"Jupyter available at => {tunnel.url}", flush=True)
            print(f"Token: {jupyter_token}", flush=True)

            # Yield the tunnel URL so caller can get it
            yield {
                "status": "ready",
                "tunnel_url": tunnel.url,
                "token": jupyter_token,
            }

            try:
                end_time = time.time() + session_timeout
                while time.time() < end_time:
                    time.sleep(5)
                    # Check if process is still running
                    if jupyter_process.poll() is not None:
                        print("Jupyter process exited unexpectedly")
                        break
                print(f"Reached end of {session_timeout} second timeout. Exiting...")
            except KeyboardInterrupt:
                print("Interrupted. Exiting...")
            finally:
                jupyter_process.kill()
                volume.commit()  # Save any notebook changes

            yield {
                "status": "shutdown",
                "tunnel_url": tunnel.url,
            }

    return app, run_jupyter
