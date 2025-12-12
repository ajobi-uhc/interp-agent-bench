"""
Shared utilities for interpretability experiments.
"""

from .extract_activations import extract_activation
from .steering_hook import create_steering_hook

__all__ = ['extract_activation', 'create_steering_hook']
