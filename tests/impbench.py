"""Docker-in-docker sandbox for evaluation benchmarks."""

from interp_infra.environment import Sandbox, SandboxConfig, ExecutionMode, RepoConfig
from conftest import auto_cleanup

config = SandboxConfig(
    python_packages=["inspect_ai", "anthropic", "openai"],
    system_packages=["git", "curl"],
    docker_in_docker=True,
    execution_mode=ExecutionMode.NOTEBOOK,
    debug=True,
    repos=[RepoConfig(url="safety-research/impossiblebench", install=True)],
)

sandbox = Sandbox(config)

with auto_cleanup(sandbox):
    sandbox.start(name="impbench")

    print(f"Repo: {sandbox._repo_handles[0].local_path}")
    print(f"VS Code: {sandbox.code_server_url}")
    print(f"Jupyter: {sandbox.jupyter_url}")

    # Verify docker
    sandbox.exec("docker info")
    print("Docker ready")

    input("\nPress Enter to cleanup...")
