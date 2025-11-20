"""
Environment system for composable investigation setups.

Environments define how to build experiment-specific namespaces inside the Jupyter kernel.
They handle model loading, API access, dataset preparation, etc.
"""

from interp_infra.environment.base import (
    EnvironmentConfig,
    Environment,
    register_environment,
    get_environment,
    ENVIRONMENTS,
)

# Auto-import all environments to register them
from interp_infra.environment import model
from interp_infra.environment import api_access

__all__ = [
    "EnvironmentConfig",
    "Environment",
    "register_environment",
    "get_environment",
    "ENVIRONMENTS",
]
