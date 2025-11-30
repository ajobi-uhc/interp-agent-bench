"""Notebook session with repo workspace."""

from interp_infra.environment import Sandbox, SandboxConfig, ExecutionMode, ModelConfig, RepoConfig
from interp_infra.execution import create_notebook_session
from conftest import auto_cleanup

config = SandboxConfig(
    python_packages=["torch", "transformers"],
    gpu="H100",
    execution_mode=ExecutionMode.NOTEBOOK,
    models=[ModelConfig(name="google/gemma-2-9b", hidden=True)],
    repos=[RepoConfig(url="PalisadeResearch/shutdown_avoidance")],
)

sandbox = Sandbox(config)

with auto_cleanup(sandbox):
    sandbox.start(name="repo")

    session = create_notebook_session(sandbox)

    # Test workspace
    session.exec("""
print(f"Workspace: {WORKSPACE}")
print(f"Files: {list(WORKSPACE.glob('*.py'))[:3]}")
""")

    print(f"Repo ready at: {sandbox._repo_handles[0].local_path}")
