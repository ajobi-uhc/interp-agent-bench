"""
interp-infra: Simple GPU infrastructure for interpretability experiments.

Three layers:
1. Environment: Sandbox, models, repos
2. Execution: Notebook sessions
3. Harness: Agent runners

Usage:
    from interp_infra.environment import Sandbox, SandboxConfig
    from interp_infra.execution import create_notebook_session
    from interp_infra.harness import run_agent

    # Create sandbox
    sandbox = Sandbox(SandboxConfig(gpu="H100", notebook=True))
    sandbox.prepare_model("google/gemma-2-9b")
    sandbox.start()

    # Create session
    session = create_notebook_session(sandbox)

    # Run agent
    async for msg in run_agent(session, task="...", provider="claude"):
        print(msg)
"""

__version__ = "0.2.0"
