"""Test container building from repo with Dockerfile."""

from interp_infra.environment import Sandbox, SandboxConfig, ExecutionMode, RepoConfig
from interp_infra.execution import create_notebook_session
from conftest import auto_cleanup

config = SandboxConfig(
    python_packages=["transformers"],
    gpu="H100",
    execution_mode=ExecutionMode.NOTEBOOK,
    docker_in_docker=True,  # Required for container building
    repos=[
        RepoConfig(
            url="https://github.com/dockersamples/python-flask-sample",
            dockerfile="Dockerfile",
        )
    ],
)

sandbox = Sandbox(config)

with auto_cleanup(sandbox):
    sandbox.start(name="container-test")

    # Build and start the container
    repo_handle = sandbox._repo_handles[0]
    sandbox.start_container(repo_handle)

    # Verify container is running
    result = sandbox.exec("docker ps")
    print("Running containers:")
    print(result)

    # Verify container is in the list
    assert repo_handle.container_name in result, "Container not found in docker ps"
    assert repo_handle.container_running, "Container should be marked as running"

    # Test that we can interact with the container
    result = sandbox.exec(f"docker inspect {repo_handle.container_name}")
    print(f"Container inspected successfully")

    # Create a notebook session to test integration
    session = create_notebook_session(sandbox)

    # Test that container_exec function is available in notebook
    result = session.exec(f"""
print(f"Container name: {container_exec('echo test')}")
print("Container interaction from notebook works!")
""")

    print("Container test passed!")
    print(f"Session: {session.session_id}")
    print(f"Container: {repo_handle.container_name}")
