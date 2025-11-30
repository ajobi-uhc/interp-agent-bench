"""
interp-infra: Simple GPU infrastructure for interpretability experiments.

Three layers:
1. Environment: Sandbox, ScopedSandbox, models, repos
2. Execution: Notebook sessions with Extensions
3. Harness: Agent runners

Usage:
    from interp_infra.environment import Sandbox, SandboxConfig, ExecutionMode, ScopedSandbox
    from interp_infra.execution import create_notebook_session
    from interp_infra.extension import Extension
    from interp_infra.harness import run_agent

    # Create sandbox
    sandbox = Sandbox(SandboxConfig(
        gpu="H100",
        execution_mode=ExecutionMode.NOTEBOOK
    ))
    sandbox.prepare_model("google/gemma-2-9b")
    sandbox.start()

    # Create session
    session = create_notebook_session(sandbox)

    # Add extensions
    steering = Extension.from_dir("./techniques/steering")
    session.add(steering)

    # Run agent
    async for msg in run_agent(
        mcp_config=session.mcp_config,
        system_prompt=session.system_prompt,
        task="...",
        provider="claude"
    ):
        print(msg)
"""

__version__ = "0.2.0"

from .extension import Extension

__all__ = ["Extension"]
