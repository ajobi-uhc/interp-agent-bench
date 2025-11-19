#!/usr/bin/env python
"""
Download a HuggingFace model to a Modal Volume for fast model serving.

Usage:
    python scripts/download_model_to_volume.py --model-id meta-llama/Llama-2-70b-hf --volume-name model-cache
    python scripts/download_model_to_volume.py --model-id Qwen/Qwen3-30B-A3B-Instruct-2507 --volume-name model-cache --version 2

This creates a persistent Modal Volume and downloads the specified model to it.
The model can then be mounted and loaded instantly in any Modal Function.
"""

import argparse
import modal

# Create Modal app
app = modal.App("download-model-to-volume")

# Base image with HuggingFace
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "huggingface_hub",
)


@app.function(
    image=image,
    timeout=3600 * 4,  # 4 hour timeout for large models
    secrets=[modal.Secret.from_name("huggingface-secret")],  # For gated models
)
def download_model(model_id: str, volume_path: str):
    """
    Download a HuggingFace model to the mounted volume.

    Args:
        model_id: HuggingFace model identifier (e.g., "meta-llama/Llama-2-70b-hf")
        volume_path: Path where volume is mounted (e.g., "/vol")
    """
    import os
    from pathlib import Path
    from huggingface_hub import snapshot_download

    print(f"📥 Downloading {model_id}...")
    print(f"   Destination: {volume_path}")

    # Get HF token from environment
    token = os.environ.get("HF_TOKEN")
    if token:
        print("   Using HF_TOKEN for authentication")

    # Create target directory
    target_dir = Path(volume_path) / "models" / model_id.replace("/", "--")
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"   Target directory: {target_dir}")

    # Download model to volume
    snapshot_download(
        repo_id=model_id,
        local_dir=str(target_dir),
        token=token,
        resume_download=True,
    )

    print(f"✅ Download complete!")
    print(f"   Model saved to: {target_dir}")

    return str(target_dir)


@app.local_entrypoint()
def main(
    model_id: str,
    volume_name: str = "model-cache",
    version: int = 2,
):
    """
    Main entrypoint: Create volume and download model.

    Args:
        model_id: HuggingFace model ID to download
        volume_name: Name of the Modal volume to create/use
        version: Volume version (1 or 2, default 2)
    """
    print("=" * 70)
    print("🚀 Model Download to Modal Volume")
    print("=" * 70)
    print(f"   Model ID: {model_id}")
    print(f"   Volume: {volume_name} (v{version})")
    print("=" * 70)

    # Create or get volume
    print(f"\n📦 Creating/getting volume '{volume_name}'...")
    try:
        volume = modal.Volume.lookup(volume_name)
        print(f"   ✅ Using existing volume: {volume_name}")
    except modal.exception.NotFoundError:
        print(f"   Creating new volume: {volume_name} (v{version})")
        volume = modal.Volume.from_name(
            volume_name,
            create_if_missing=True,
            version=version,
        )
        print(f"   ✅ Created volume: {volume_name}")

    # Download model to volume
    print(f"\n📥 Starting download...")
    volume_path = "/vol"

    # Create a temporary function with the volume mounted
    @app.function(
        image=image,
        timeout=3600 * 4,
        volumes={volume_path: volume},
        secrets=[modal.Secret.from_name("huggingface-secret")],
    )
    def _download_with_volume(model_id: str):
        result = download_model.local(model_id, volume_path)

        # Commit changes to volume
        import modal
        print("\n💾 Committing changes to volume...")
        volume.commit()
        print("   ✅ Changes committed!")

        return result

    # Run the download
    try:
        saved_path = _download_with_volume.remote(model_id)
        print("\n" + "=" * 70)
        print("🎉 SUCCESS!")
        print("=" * 70)
        print(f"   Model: {model_id}")
        print(f"   Volume: {volume_name}")
        print(f"   Path: {saved_path}")
        print("\nTo use this model in your code:")
        print(f'   volume = modal.Volume.from_name("{volume_name}")')
        print(f'   @app.function(volumes={{"/vol": volume}})')
        print(f"   def my_function():")
        print(f'       model = load_from_disk("{saved_path}")')
        print("=" * 70)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download HuggingFace model to Modal Volume"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="HuggingFace model ID (e.g., meta-llama/Llama-2-70b-hf)",
    )
    parser.add_argument(
        "--volume-name",
        type=str,
        default="model-cache",
        help="Modal volume name (default: model-cache)",
    )
    parser.add_argument(
        "--version",
        type=int,
        choices=[1, 2],
        default=2,
        help="Volume version (default: 2)",
    )

    args = parser.parse_args()

    with app.run():
        main(args.model_id, args.volume_name, args.version)
