#!/usr/bin/env python3
"""
Simple MCP Server for deploying pre-built experiment configs.

Wraps interp_infra's deploy_experiment function.
"""

from fastmcp import FastMCP
from pathlib import Path
import sys
import os
from datetime import datetime
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from interp_infra.deploy import deploy_experiment

mcp = FastMCP("deploy")


@mcp.tool()
def deploy_from_config(config_path: str) -> Dict[str, Any]:
    """Deploy experiment infrastructure from config.yaml.

    Provisions GPU clusters, loads models, and starts Jupyter sessions.
    Returns connection info for each deployed pod.

    Args:
        config_path: Path to config.yaml (e.g., "config.yaml" or "tasks/experiment/config.yaml")

    Returns:
        {
            "pods": [
                {
                    "pod_id": "pod_1",
                    "session_id": "abc123",
                    "jupyter_url": "https://...",
                    "notebook_path": "./outputs/notebook.ipynb",
                    "sandbox_id": "sb-xxx"
                }
            ],
            "workspace_root": "./outputs",
            "status": "ready"
        }

    The session_id and jupyter_url can be used with the 'scribe' MCP server
    to interact with the Jupyter kernel (attach_to_session, execute_code, etc).
    """
    config_path = Path(config_path)

    if not config_path.exists():
        return {
            "error": f"Config not found: {config_path}",
            "tip": "Check the path is relative to current directory"
        }

    print(f"🚀 Deploying from config: {config_path}")

    # Load config
    from interp_infra.config.parser import load_config
    config = load_config(config_path)

    # Determine workspace root
    workspace_root = Path.cwd() / config.workspace.local_root
    workspace_root = workspace_root.resolve()
    workspace_root.mkdir(parents=True, exist_ok=True)

    # Get number of pods from config
    num_pods = getattr(config.harness, 'num_pods', 1)

    print(f"📦 Deploying {num_pods} pod(s)...")

    # Deploy each pod and return connection info
    pods = []
    for i in range(num_pods):
        pod_id = f"pod_{i+1}" if num_pods > 1 else "pod_1"
        print(f"   Deploying {pod_id}...")

        deployment = deploy_experiment(str(config_path))

        # Workspace per pod
        if num_pods > 1:
            pod_dir = workspace_root / pod_id
            pod_dir.mkdir(parents=True, exist_ok=True)
            notebook_path = pod_dir / "notebook.ipynb"
        else:
            notebook_path = workspace_root / "notebook.ipynb"

        pods.append({
            "pod_id": pod_id,
            "session_id": deployment.session_id,
            "jupyter_url": deployment.jupyter_url,
            "notebook_path": str(notebook_path),
            "sandbox_id": deployment.sandbox_id
        })

        print(f"   ✅ {pod_id} ready (session: {deployment.session_id})")

    print(f"✅ All {num_pods} pod(s) deployed")
    print(f"   Workspace: {workspace_root}")

    return {
        "pods": pods,
        "workspace_root": str(workspace_root),
        "status": "ready"
    }


if __name__ == "__main__":
    mcp.run()
