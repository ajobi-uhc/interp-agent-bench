"""Pure Modal SDK client functions - thin wrappers around Modal API."""

import modal
from typing import Optional, Dict, Any, List


def lookup_or_create_app(name: str) -> modal.App:
    """
    Lookup or create a Modal App.

    Args:
        name: App name

    Returns:
        modal.App instance
    """
    return modal.App.lookup(name, create_if_missing=True)


def create_volume(name: str) -> modal.Volume:
    """
    Create or lookup a Modal Volume.

    Args:
        name: Volume name

    Returns:
        modal.Volume instance
    """
    return modal.Volume.from_name(name, create_if_missing=True)


def create_sandbox(
    startup_script: str,
    image: modal.Image,
    gpu: Optional[str],
    timeout: int,
    app: modal.App,
    encrypted_ports: List[int],
    secrets: List[modal.Secret],
    env: Dict[str, str],
    volumes: Optional[Dict[str, modal.Volume]] = None,
) -> modal.Sandbox:
    """
    Create a Modal Sandbox.

    Args:
        startup_script: Python script to run on startup
        image: Modal Image to use
        gpu: GPU string (e.g., "A100", "T4:2") or None for CPU
        timeout: Sandbox timeout in seconds
        app: Modal App to associate with
        encrypted_ports: List of ports to expose with HTTPS tunnels
        secrets: List of Modal Secrets to mount
        env: Environment variables
        volumes: Optional dict of mount_path -> Volume

    Returns:
        modal.Sandbox instance
    """
    kwargs = {
        "image": image,
        "gpu": gpu,
        "timeout": timeout,
        "app": app,
        "encrypted_ports": encrypted_ports,
        "secrets": secrets,
        "env": env,
    }

    if volumes:
        kwargs["volumes"] = volumes

    return modal.Sandbox.create(
        "python",
        "-c",
        startup_script,
        **kwargs,
    )


def exec_in_sandbox(sandbox: modal.Sandbox, *command: str):
    """
    Execute a command in a sandbox.

    Args:
        sandbox: Modal Sandbox instance
        *command: Command parts (e.g., "python", "-c", "print('hello')")

    Returns:
        Process handle with .stdout, .stderr, .wait(), .returncode
    """
    return sandbox.exec(*command)


def get_sandbox_tunnels(sandbox: modal.Sandbox):
    """
    Get the tunnel mapping for a sandbox.

    Args:
        sandbox: Modal Sandbox instance

    Returns:
        Dict mapping port -> Tunnel object with .url attribute
    """
    return sandbox.tunnels()


def get_sandbox_from_id(sandbox_id: str) -> modal.Sandbox:
    """
    Get a Sandbox object from its ID.

    Args:
        sandbox_id: Sandbox ID string

    Returns:
        modal.Sandbox instance
    """
    return modal.Sandbox.from_id(sandbox_id)


def terminate_sandbox(sandbox: modal.Sandbox):
    """
    Terminate a Modal Sandbox.

    Args:
        sandbox: Modal Sandbox instance to terminate
    """
    sandbox.terminate()


def get_modal_gpu_string(gpu_type: str, count: int) -> str:
    """
    Convert GPU type and count to Modal GPU string.

    Args:
        gpu_type: GPU type string (e.g., "A100-80GB", "H100", "T4")
        count: Number of GPUs

    Returns:
        Modal GPU string (e.g., "A100", "H100:2")
    """
    # Map common GPU types to Modal GPU strings
    if "H100" in gpu_type:
        gpu_name = "H100"
    elif "A100" in gpu_type:
        gpu_name = "A100"
    elif "A10G" in gpu_type:
        gpu_name = "A10G"
    elif "L4" in gpu_type:
        gpu_name = "L4"
    elif "T4" in gpu_type:
        gpu_name = "T4"
    elif "H200" in gpu_type:
        gpu_name = "H200"
    else:
        # Default to A100
        print(f"Warning: Unknown GPU type '{gpu_type}', defaulting to A100")
        gpu_name = "A100"

    # Return GPU string with count syntax
    if count > 1:
        return f"{gpu_name}:{count}"
    return gpu_name
