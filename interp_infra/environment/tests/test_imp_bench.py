"""Test sandbox with docker-in-docker and inspect_ai evaluation."""

from interp_infra.environment import Sandbox, SandboxConfig, ExecutionMode

# Create sandbox config with docker-in-docker for inspect_ai
config = SandboxConfig(
    python_packages=["inspect_ai", "anthropic", "openai"],
    system_packages=["git", "curl"],  # curl needed for code-server install
    docker_in_docker=True,
    execution_mode=ExecutionMode.NOTEBOOK,
    debug=True,  # Enable code-server for debugging
)

print("Creating sandbox with docker-in-docker...")
sandbox = Sandbox(config)

# Prepare the impossiblebench repo and install it
print("Preparing impossiblebench repo...")
repo = sandbox.prepare_repo("safety-research/impossiblebench", install=True)

# Start the sandbox
print("Starting sandbox...")
sandbox.start(name="test-inspect-impossiblebench")

print(f"\n{'='*60}")
print(f"Sandbox started: {sandbox.sandbox_id}")
print(f"Repo at: {repo.local_path}")
print(f"\n🚀 VS Code (browser): {sandbox.code_server_url}")
print(f"📓 Jupyter: {sandbox.jupyter_url}")
print(f"{'='*60}\n")

# Verify docker is running
print("\nVerifying docker daemon...")
try:
    docker_info = sandbox.exec("docker info")
    print("Docker is running")
except Exception as e:
    print(f"Docker check failed: {e}")

# Verify repo was cloned
print("\nVerifying repo clone...")
try:
    ls_output = sandbox.exec(f"ls -la {repo.local_path}")
    print(f"Repo cloned:\n{ls_output}")
except Exception as e:
    print(f"Repo check failed: {e}")

# Run the inspect_ai evaluation
print("\nRunning inspect_ai evaluation...")
try:
    eval_script = """
import sys

try:
    from inspect_ai import eval
    from impossiblebench import impossible_livecodebench, impossible_swebench
    print("Imports successful")

    task = impossible_livecodebench(
        split="conflicting",   # "original", "oneoff", or "conflicting"
        agent_type="minimal",  # Simple submission loop
        limit=10,              # Run on first 10 samples
    )

    # Run with a model (you'll need API key in environment)
    eval(task, model="openai/gpt-4o")
    print("Evaluation complete")
except Exception as e:
    print(f"Error during execution: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
    output = sandbox.exec_python(eval_script)
    print(f"Evaluation output:\n{output}")
except Exception as e:
    print(f"Evaluation failed: {e}")

# Keep sandbox alive for manual exploration
print("\n\n" + "="*60)
print("Sandbox is ready for manual exploration!")
print(f"\n🚀 VS Code (browser): {sandbox.code_server_url}")
print(f"📓 Jupyter: {sandbox.jupyter_url}")
print("\nOpen VS Code URL in your browser, then:")
print("  - Open terminal (Ctrl+` or Terminal → New Terminal)")
print("  - Install impossiblebench: cd /workspace/impossiblebench && pip install -e .")
print("  - Run the evaluation")
print("  - Debug any issues")
print("="*60)
input("\nPress Enter when done to cleanup and terminate sandbox...")
