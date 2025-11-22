"""Execution stage - Interface setup (how agents interact with environment)."""

from .notebook import setup_notebook_execution, NotebookHandle


def setup_execution(env_handle, config):
    """
    Stage 2: Setup execution interface.

    Args:
        env_handle: EnvironmentHandle from Stage 1
        config: ExperimentConfig

    Returns:
        ExecutionHandle (varies by type)
    """
    # Get execution config (defaults to notebook)
    exec_config = getattr(config, 'execution', None)
    if exec_config is None:
        # Backwards compatibility - default to notebook
        exec_type = "notebook"
    else:
        exec_type = exec_config.get('type', 'notebook') if isinstance(exec_config, dict) else getattr(exec_config, 'type', 'notebook')

    if exec_type == "notebook":
        return setup_notebook_execution(env_handle, config)

    elif exec_type == "filesystem":
        # Placeholder - just return env info
        return {
            'type': 'filesystem',
            'sandbox_id': env_handle.sandbox_id,
            'jupyter_url': env_handle.jupyter_url
        }

    elif exec_type == "mcp":
        # Placeholder
        raise NotImplementedError("MCP execution not yet implemented")

    else:
        raise ValueError(f"Unknown execution type: {exec_type}")


__all__ = ['setup_execution', 'NotebookHandle']
