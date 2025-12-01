"""Shared utilities for session management."""

from pathlib import Path


def read_and_exec(file_path: str, exec_func, **kwargs) -> any:
    """Read a file and execute it using the provided exec function."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    code = path.read_text()
    return exec_func(code, **kwargs)
