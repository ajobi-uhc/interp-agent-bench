"""Test Docker-in-Sandbox functionality with Modal."""

import os
os.environ["MODAL_IMAGE_BUILDER_VERSION"] = "2025.06"

import modal
from pathlib import Path
from interp_infra.config.parser import load_config
from interp_infra.environment.modal import ModalEnvironment


def test_docker_in_sandbox():
    """Test that Docker runs inside Modal sandbox."""
    print("Testing Docker-in-Sandbox...")

    # Load shutdown resistance config (has enable_docker: true)
    config = load_config(Path("tasks/shutdown-resistance/config.yaml"))

    print(f"Config loaded: enable_docker = {config.environment.image.enable_docker}")

    # Build image with Docker support
    print("\nBuilding image with Docker support...")
    client = ModalEnvironment()

    # Enable output to see build logs
    with modal.enable_output():
        image = client.build_image(config.environment.image, config.environment.gpu)

    print("Image built successfully!")

    # Create a test sandbox
    print("\nCreating test sandbox...")
    app = modal.App.lookup("test-docker-in-sandbox", create_if_missing=True)

    with modal.enable_output():
        sandbox = modal.Sandbox.create(
            "/start-dockerd.sh",
            image=image,
            app=app,
            timeout=600,
            experimental_options={"enable_docker": True},
        )

    print(f"Sandbox created: {sandbox.object_id}")

    # Wait for Docker daemon to be ready
    print("\nWaiting for Docker daemon...")
    import time
    for i in range(30):
        try:
            p = sandbox.exec("docker", "info", timeout=5)
            stdout = p.stdout.read()
            p.wait()
            if p.returncode == 0:
                print("Docker daemon is ready!")
                break
        except:
            pass
        time.sleep(1)
    else:
        print("Warning: Docker daemon may not be ready")

    # Test running a simple container (use --network=host to avoid gVisor networking issues)
    print("\nRunning hello-world container...")
    p = sandbox.exec("docker", "run", "--rm", "--network=host", "hello-world")
    output = p.stdout.read()
    p.wait()

    if p.returncode == 0:
        print("\n✅ SUCCESS! Docker-in-Sandbox is working!")
        print("\nDocker output:")
        print(output)
    else:
        print("\n❌ FAILED! Docker command failed")
        stderr = p.stderr.read()
        print(f"Error: {stderr}")

    # Cleanup
    print("\nCleaning up...")
    try:
        sandbox.terminate()
    except Exception as e:
        # Ignore cleanup errors
        pass

    # Give Modal time to clean up connections
    import time
    time.sleep(0.5)

    print("Test complete!")


if __name__ == "__main__":
    try:
        test_docker_in_sandbox()
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        raise
