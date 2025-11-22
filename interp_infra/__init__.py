"""
interp-infra: Minimal GPU infrastructure for interpretability experiments.

Three-stage architecture:
1. Environment: Create GPU sandbox, download models, start Jupyter
2. Execution: Setup interaction mode (notebook, filesystem, MCP)
3. Harness: Agent orchestration (handled by caller)
"""

__version__ = "0.1.0"

# Lazy imports to avoid loading modal in kernel environment
__all__ = [
    'deploy_experiment',
    'setup_environment',
    'setup_execution',
    'load_config',
    'EnvironmentHandle',
    'NotebookHandle',
]

def __getattr__(name):
    """Lazy import to avoid importing modal in kernel environment."""
    if name == 'deploy_experiment':
        from .deploy import deploy_experiment
        return deploy_experiment
    elif name == 'setup_environment':
        from .environment import setup_environment
        return setup_environment
    elif name == 'setup_execution':
        from .execution import setup_execution
        return setup_execution
    elif name == 'load_config':
        from .config.parser import load_config
        return load_config
    elif name == 'EnvironmentHandle':
        from .environment import EnvironmentHandle
        return EnvironmentHandle
    elif name == 'NotebookHandle':
        from .execution import NotebookHandle
        return NotebookHandle
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
