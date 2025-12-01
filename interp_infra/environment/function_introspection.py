"""Introspect Python functions to generate JSON schemas."""

import inspect
from typing import Callable, get_type_hints, get_origin, get_args


TYPE_MAP = {
    int: "integer",
    float: "number",
    str: "string",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def python_type_to_json_type(hint) -> str:
    """Convert Python type hint to JSON schema type."""
    origin = get_origin(hint)

    # Handle Optional[X] -> X with required=False handled separately
    if origin is type(None):
        return "null"

    # Check origin first (for generic types like List[int])
    check_type = origin if origin else hint

    # Direct mapping
    for py_type, json_type in TYPE_MAP.items():
        if check_type is py_type:
            return json_type

    # Default to string for unknown types
    return "string"


def build_function_schema(func: Callable) -> dict:
    """
    Introspect function and return MCP-compatible schema.

    Returns:
        {
            "description": "...",
            "inputSchema": {
                "type": "object",
                "properties": {...},
                "required": [...]
            }
        }
    """
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or f"Call {func.__name__}()"

    properties = {}
    required = []

    try:
        hints = get_type_hints(func)
    except:
        hints = {}

    for param_name, param in sig.parameters.items():
        if param_name in ('self', 'cls'):
            continue

        # Get type from hint or default to string
        param_type = "string"
        if param_name in hints:
            param_type = python_type_to_json_type(hints[param_name])

        properties[param_name] = {"type": param_type}

        # Required if no default value
        if param.default == inspect.Parameter.empty:
            required.append(param_name)

    return {
        "description": doc,
        "inputSchema": {
            "type": "object",
            "properties": properties,
            "required": required
        }
    }
