"""Base harness abstraction for agent orchestration patterns."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class Harness(ABC):
    """
    Base class for agent orchestration harnesses.

    A harness receives infrastructure (deployment) and configuration,
    then implements a specific orchestration pattern (single agent, multi-agent, etc.).

    The deployment object provides everything needed:
    - jupyter_url: URL to Jupyter server
    - session_id: Pre-warmed session with models loaded
    - sandbox_id: Modal sandbox ID
    - workspace: Local workspace for outputs

    Subclasses only need to implement run() with their orchestration logic.
    """

    def __init__(self, deployment, config):
        """
        Initialize harness with infrastructure and configuration.

        Args:
            deployment: Deployment object from Stage 1+2 (has jupyter_url, session_id, etc.)
            config: ExperimentConfig with task, skills, execution settings
        """
        self.deployment = deployment
        self.config = config

    @abstractmethod
    async def run(self) -> Dict[str, Any]:
        """
        Execute the orchestration pattern.

        This is where the harness-specific logic lives:
        - Create agents using toolkit helpers
        - Coordinate agent execution (sequential, parallel, interleaved)
        - Handle state management and data passing
        - Return results

        Returns:
            Dictionary with results (format is harness-specific)
        """
        pass
