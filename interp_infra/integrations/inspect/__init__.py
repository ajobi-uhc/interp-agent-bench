"""Inspect AI integration - provides Modal sandbox backend.

Usage:
    # In your agent code or task setup, import this module to register
    # the Modal sandbox provider with Inspect:
    import interp_infra.integrations.inspect

    # Then set environment variable for Inspect to use it:
    os.environ['INSPECT_EVAL_SANDBOX'] = 'modal'

    # Or create .env file in your Inspect repo:
    # INSPECT_EVAL_SANDBOX=modal

    # Now run Inspect evals normally:
    # inspect eval shutdown.py
"""

# Import to register the provider
from .modal_sandbox import ModalSandboxEnvironment

__all__ = ["ModalSandboxEnvironment"]
