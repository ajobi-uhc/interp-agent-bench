"""Session initialization - lightweight kernel-side infra globals.

This module provides cheap, safe globals to the kernel (paths, metadata).
Heavy operations (git clone, model downloads) happen in ModalClient._infra_prewarm().
Model loading and environment setup is handled by environment.warm_init().
"""

from pathlib import Path
from typing import Dict, Any
from ..config.schema import ExperimentConfig


def initialize_session(config: ExperimentConfig) -> Dict[str, Any]:
    """Initialize lightweight infrastructure globals inside the kernel.

    This runs at kernel startup and should be FAST:
    - No git clone (already done in _infra_prewarm)
    - No model downloads (already done in _infra_prewarm)
    - No heavy filesystem operations
    - Just provide paths and metadata

    Heavy infra preparation happens in ModalClient._infra_prewarm() before kernel starts.
    Model construction happens in environment.warm_init() from pre-downloaded cache.

    Args:
        config: Experiment configuration

    Returns:
        Dictionary of lightweight infrastructure globals (paths, metadata)
    """
    namespace = {}

    # Provide workspace path if repos were cloned by _infra_prewarm
    workspace = Path("/workspace")
    if workspace.exists():
        namespace["workspace"] = str(workspace)

    # Add experiment metadata
    namespace["experiment_name"] = config.name

    return namespace
