"""Environment layer - Sandboxes for running code remotely.

Structure:
- sandbox.py: Main Sandbox class (orchestration)
- scoped_sandbox.py: ScopedSandbox class (isolated environments with RPC)
- utils/: Utilities (services, volumes, image building, handles, RPC server, codegen)
- clients/: Infrastructure provider clients (Modal, etc.)
"""

from .sandbox import Sandbox, SandboxConfig, ExecutionMode, ModelConfig, RepoConfig
from .scoped_sandbox import ScopedSandbox

__all__ = [
    "Sandbox",
    "SandboxConfig",
    "ExecutionMode",
    "ModelConfig",
    "RepoConfig",
    "ScopedSandbox",
]
