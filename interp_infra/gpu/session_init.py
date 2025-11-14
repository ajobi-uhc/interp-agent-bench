"""Session initialization - infrastructure setup only.

This module handles infra concerns (git repos, workspace setup, auth).
Model loading and experiment-specific setup is handled by recipes.
"""

import subprocess
from pathlib import Path
from typing import Dict, Any
from ..config.schema import ExperimentConfig


def clone_github_repos(repos: list[str]) -> None:
    """Clone GitHub repositories to /workspace.

    Args:
        repos: List of repo URLs (e.g., ["org/repo", "https://github.com/org/repo"])
    """
    if not repos:
        return

    workspace = Path("/workspace")
    workspace.mkdir(exist_ok=True)

    for repo in repos:
        # Handle both "org/repo" and full URLs
        if not repo.startswith("http"):
            repo = f"https://github.com/{repo}"

        repo_name = repo.split("/")[-1].replace(".git", "")
        repo_path = workspace / repo_name

        if not repo_path.exists():
            print(f"Cloning {repo}...")
            subprocess.run(
                ["git", "clone", repo, str(repo_path)],
                check=True,
                capture_output=True
            )
            print(f"✅ Cloned {repo_name}")


def initialize_session(config: ExperimentConfig) -> Dict[str, Any]:
    """Run infrastructure initialization.

    This handles only infra concerns:
    - Clone git repos
    - Set up workspace paths
    - HF authentication (via Modal secrets)

    Model loading and experiment-specific setup is handled by recipes.

    Args:
        config: Experiment configuration

    Returns:
        Dictionary of infrastructure variables (workspace path, etc.)
    """
    namespace = {}

    # Clone repos
    clone_github_repos(config.github_repos)

    # Add workspace path to namespace
    if config.github_repos:
        namespace["workspace"] = "/workspace"

    return namespace
