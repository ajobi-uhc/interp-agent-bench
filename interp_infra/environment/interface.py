"""Interface decorators for scoped sandboxes."""

from typing import Callable, Any
from functools import wraps

# Registry of exposed functions
_EXPOSED_FUNCTIONS: dict[str, Callable] = {}


def expose(func: Callable) -> Callable:
    """
    Decorator to expose a function via RPC.

    Usage:
        @expose
        def my_function(arg1: str, arg2: int) -> dict:
            return {"result": arg1 * arg2}
    """
    _EXPOSED_FUNCTIONS[func.__name__] = func

    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def get_exposed_functions() -> dict[str, Callable]:
    """Get all functions marked with @expose decorator."""
    return _EXPOSED_FUNCTIONS.copy()


def clear_exposed_functions():
    """Clear the registry (useful for testing)."""
    _EXPOSED_FUNCTIONS.clear()
