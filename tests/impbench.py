"""Docker-in-docker sandbox for evaluation benchmarks."""

from interp_infra.environment import Sandbox, SandboxConfig, ExecutionMode
from conftest import auto_cleanup

config = SandboxConfig(
    python_packages=["inspect_ai", "anthropic", "openai"],
    system_packages=["git", "curl"],
    docker_in_docker=True,
    execution_mode=ExecutionMode.NOTEBOOK,
    debug=True,
)

sandbox = Sandbox(config)
repo = sandbox.prepare_repo("safety-research/impossiblebench", install=True)

with auto_cleanup(sandbox):
    sandbox.start(name="impbench")

    print(f"Repo: {repo.local_path}")
    print(f"VS Code: {sandbox.code_server_url}")
    print(f"Jupyter: {sandbox.jupyter_url}")

    # Verify docker
    sandbox.exec("docker info")
    print("Docker ready")

    input("\nPress Enter to cleanup...")
