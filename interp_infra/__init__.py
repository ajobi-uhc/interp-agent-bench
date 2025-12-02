"""
interp-infra: Simple GPU infrastructure for interpretability experiments.

Clean abstractions:
1. Environment: Sandbox, ScopedSandbox (GPU environments)
2. Workspace: Libraries, files, skills for agent
3. Execution: Sessions (sandbox + workspace)
4. Harness: Agent runners

Usage:
    from interp_infra.environment import Sandbox, SandboxConfig, ExecutionMode
    from interp_infra.workspace import Workspace, Library
    from interp_infra.execution import create_notebook_session
    from interp_infra.harness import run_agent

    # 1. Environment
    sandbox = Sandbox(SandboxConfig(
        gpu="H100",
        execution_mode=ExecutionMode.NOTEBOOK,
        models=[ModelConfig(name="google/gemma-2-9b")],
    ))
    sandbox.start()

    # 2. Workspace
    workspace = Workspace(
        libraries=[Library.from_file("steering_utils.py")],
        skill_dirs=["./skills"],
    )

    # 3. Session
    session = create_notebook_session(sandbox, workspace)

    # 4. Run agent
    async for msg in run_agent(
        session=session,
        task="Find steering vectors",
        mcp_servers=[],
        prompts=["# Instructions..."],
        provider="claude",
    ):
        print(msg)

RPC Interface Helpers:
    # In your RPC interface code (runs in sandbox):
    from interp_infra.rpc_helpers import get_model_path
    from transformers import AutoModel

    model_path = get_model_path("google/gemma-2-9b")
    model = AutoModel.from_pretrained(model_path)
"""

__version__ = "0.3.0"

# No exports at top level - users import from submodules
__all__ = []
