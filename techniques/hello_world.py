"""Hello World Technique

Takes a name string and returns "hello <name>" so agents can verify the white-box flow.
"""

from __future__ import annotations


def run(name: str) -> str:
    """Return a greeting for the provided name."""
    return f"hello {name}"


TECHNIQUE_NAME = "hello_world"
