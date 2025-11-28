"""Test sandbox with docker-in-docker and inspect_ai evaluation."""

from interp_infra.environment.sandbox import Sandbox, SandboxConfig

# Create sandbox config with docker-in-docker for inspect_ai
config = SandboxConfig(
    python_packages=["inspect_ai", "anthropic", "openai"],
    system_packages=["git"],
    docker_in_docker=True,
    notebook=True,
)

print("Creating sandbox with docker-in-docker...")
sandbox = Sandbox(config)

# Prepare the shutdown-avoidance repo
print("Preparing shutdown-avoidance repo...")
repo = sandbox.prepare_repo("PalisadeResearch/shutdown_avoidance")

# Start the sandbox
print("Starting sandbox...")
sandbox.start(name="test-inspect-shutdown")

print(f"✓ Sandbox started: {sandbox.sandbox_id}")
print(f"✓ Repo should be at: {repo.local_path}")

# Verify docker is running
print("\nVerifying docker daemon...")
try:
    docker_info = sandbox.exec("docker info")
    print("✓ Docker is running")
except Exception as e:
    print(f"✗ Docker check failed: {e}")

# Verify repo was cloned
print("\nVerifying repo clone...")
try:
    ls_output = sandbox.exec(f"ls -la {repo.local_path}")
    print(f"✓ Repo cloned:\n{ls_output}")
except Exception as e:
    print(f"✗ Repo check failed: {e}")

# Run the inspect_ai evaluation
print("\nRunning inspect_ai evaluation...")
try:
    eval_script = f"""
import os
import sys
os.chdir("{repo.local_path}")

print("Current directory:", os.getcwd())
print("Python path:", sys.path[:3])

try:
    from inspect_ai import eval
    from shutdown import shutdown_avoidance
    print("✓ Imports successful")

    # Run with a model (you'll need API key in environment)
    result = eval(shutdown_avoidance, model="openai/gpt-4o")
    print(f"Evaluation complete: {{result}}")
except Exception as e:
    print(f"Error during execution: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
    output = sandbox.exec_python(eval_script)
    print(f"✓ Evaluation output:\n{output}")
except Exception as e:
    print(f"✗ Evaluation failed: {e}")

# Cleanup
print("\n\nCleaning up...")
sandbox.terminate()
print("✓ Test complete!")
