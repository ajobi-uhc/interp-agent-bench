"""Notebook session with repo workspace."""

from interp_infra.environment import Sandbox, SandboxConfig, ExecutionMode
from interp_infra.execution import create_notebook_session
from conftest import auto_cleanup

config = SandboxConfig(
    python_packages=["torch", "transformers"],
    gpu="H100",
    execution_mode=ExecutionMode.NOTEBOOK,
)

sandbox = Sandbox(config)
sandbox.prepare_model("google/gemma-2-9b", hidden=True)
repo = sandbox.prepare_repo("PalisadeResearch/shutdown_avoidance")

with auto_cleanup(sandbox):
    sandbox.start(name="repo")

    session = create_notebook_session(sandbox)

    # Test workspace
    session.exec("""
print(f"Workspace: {WORKSPACE}")
print(f"Files: {list(WORKSPACE.glob('*.py'))[:3]}")
""")

    print(f"Repo ready at: {repo.local_path}")
