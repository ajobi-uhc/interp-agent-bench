o"""Helper utilities for RPC interface code."""

import os
from typing import Optional


def get_model_path(model_name: str) -> str:
    """
    Get the path to a model configured in SandboxConfig.

    Args:
        model_name: Model name from SandboxConfig.models
                   e.g., "google/gemma-2-9b"

    Returns:
        Path to model volume

    Raises:
        ValueError: If model not configured

    Example:
        from src.rpc_helpers import get_model_path
        from transformers import AutoModel

        model_path = get_model_path("google/gemma-2-9b")
        model = AutoModel.from_pretrained(model_path)
    """
    # Convert to env var format
    env_key = f"MODEL_{_sanitize_name(model_name)}_PATH"

    path = os.environ.get(env_key)
    if not path:
        available = [k for k in os.environ.keys() if k.startswith("MODEL_") and k.endswith("_PATH")]
        raise ValueError(
            f"Model '{model_name}' not configured.\n"
            f"Expected env var: {env_key}\n"
            f"Available models: {available}"
        )

    return path


def list_configured_models() -> dict[str, str]:
    """
    List all models configured in SandboxConfig.

    Returns:
        Dict mapping model names to their paths

    Example:
        from src.rpc_helpers import list_configured_models

        models = list_configured_models()
        print(f"Available: {list(models.keys())}")
    """
    models = {}
    for key, value in os.environ.items():
        if key.startswith("MODEL_") and key.endswith("_PATH"):
            # MODEL_GOOGLE_GEMMA_2_9B_PATH -> google/gemma-2-9b
            name = key[6:-5].lower().replace("_", "-")
            models[name] = value

    return models


def _sanitize_name(name: str) -> str:
    """Convert model name to env var format."""
    return name.replace("/", "_").replace("-", "_").upper()
