"""
Agent orchestration harnesses.

This module provides abstractions for different agent orchestration patterns.
Each harness receives infrastructure (deployment) from Stage 1+2 and implements
a specific coordination pattern for agents.

Available harnesses:
- SingleAgentHarness: One agent with full notebook access (standard pattern)
- (Future: Multi-agent, Petri, Supervisor, etc.)

Example usage:
    >>> from interp_infra.harness import SingleAgentHarness, toolkit
    >>>
    >>> # Get infrastructure
    >>> deployment = deploy_experiment(config_path)
    >>>
    >>> # Run single agent harness
    >>> harness = SingleAgentHarness(deployment, config)
    >>> result = await harness.run()
    >>>
    >>> # Or create custom orchestration using toolkit
    >>> agent1 = toolkit.create_agent(deployment, sys_prompt, user_prompt)
    >>> result1 = await toolkit.run_agent(agent1)
"""

from .base import Harness
from .single_agent import SingleAgentHarness
from . import toolkit
from . import prompts

__all__ = [
    'Harness',
    'SingleAgentHarness',
    'toolkit',
    'prompts',
]
