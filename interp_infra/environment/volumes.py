"""Volume management for model storage."""

import modal


def get_or_create_volume(model_name: str) -> tuple[modal.Volume, str]:
    """
    Get or create a volume for a model.
    
    Args:
        model_name: HuggingFace model identifier (e.g. "google/gemma-9b")
        
    Returns:
        Tuple of (volume, mount_path)
    """
    volume_name = f"model--{model_name.replace('/', '--')}"
    mount_path = f"/models/{model_name.replace('/', '--')}"
    
    volume = modal.Volume.from_name(volume_name, create_if_missing=True)
    
    return volume, mount_path


def check_model_in_volume(sandbox: "Sandbox", volume_path: str) -> bool:
    """
    Check if model weights exist in volume.
    
    Args:
        sandbox: Sandbox to check in
        volume_path: Path where model should be
        
    Returns:
        True if model exists
    """
    try:
        result = sandbox.exec(f"test -f {volume_path}/config.json && echo 'exists'")
        return "exists" in result
    except Exception:
        return False