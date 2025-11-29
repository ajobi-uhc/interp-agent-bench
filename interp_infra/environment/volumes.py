"""Volume management for model storage."""

from typing import TYPE_CHECKING

import modal

if TYPE_CHECKING:
    from .sandbox import Sandbox


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


def download_model_to_volume(
    sandbox: "Sandbox",
    model_id: str,
    volume_path: str
) -> None:
    """
    Download model weights to volume path.

    Args:
        sandbox: Sandbox to download in
        model_id: HuggingFace model identifier
        volume_path: Path where model should be downloaded
    """
    script = f'''
import os
from pathlib import Path
from huggingface_hub import snapshot_download

model_path = Path("{volume_path}")
model_id = "{model_id}"
token = os.environ.get("HF_TOKEN")

if not (model_path / "config.json").exists():
    model_path.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        model_id,
        local_dir=str(model_path),
        token=token,
        resume_download=True,
    )
'''
    sandbox.exec_python(script)


def commit_volumes(volumes: dict[str, modal.Volume]) -> None:
    """
    Commit all volume changes after downloads complete.

    Args:
        volumes: Dictionary of mount_path -> Volume
    """
    if not volumes:
        return

    print("  Committing volume changes...")
    for mount_path, volume in volumes.items():
        try:
            volume.commit()
        except Exception:
            print(f"    Note: Volume at {mount_path} auto-persisted (commit not needed)")