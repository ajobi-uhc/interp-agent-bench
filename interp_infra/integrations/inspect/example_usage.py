"""Example: Using Modal sandbox with Inspect AI evaluations.

This shows how to run Inspect evals (like shutdown_avoidance) in Modal
without Docker-in-Docker issues.
"""

import os
import subprocess

# Step 1: Register Modal as an Inspect sandbox backend
# This imports ModalSandboxEnvironment and the @sandboxenv decorator runs,
# telling Inspect that "modal" is available as a sandbox type
import interp_infra.integrations.inspect

# Step 2: Tell Inspect to use Modal instead of Docker
os.environ['INSPECT_EVAL_SANDBOX'] = 'modal'

# Step 3: Run Inspect evals normally
# The repo code runs unmodified - when it says sandbox="docker",
# Inspect will use sandbox="modal" instead (due to env var override)

# Example: Clone and run shutdown_avoidance eval
subprocess.run([
    "git", "clone",
    "https://github.com/PalisadeResearch/shutdown_avoidance",
    "/workspace/shutdown_avoidance"
], check=True)

# Run the eval
os.chdir("/workspace/shutdown_avoidance")
subprocess.run(["inspect", "eval", "shutdown.py"], check=True)

# That's it! The eval runs with Modal sandboxing instead of Docker.
