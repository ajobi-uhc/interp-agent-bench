"""Code generation utilities for RPC client libraries."""

import ast
import inspect
from dataclasses import dataclass
from typing import Optional


@dataclass
class FunctionSignature:
    """Parsed function signature."""
    name: str
    params: list[str]           # Parameter names
    param_types: dict[str, str] # name → type annotation
    defaults: dict[str, any]    # name → default value
    docstring: str

    @property
    def signature_str(self) -> str:
        """Generate function signature string with type hints."""
        parts = []
        for param in self.params:
            if param in ('self', 'cls'):
                continue

            param_str = param

            # Add type hint if available
            if param in self.param_types:
                param_str += f": {self.param_types[param]}"

            # Add default if available
            if param in self.defaults:
                default = self.defaults[param]
                param_str += f" = {repr(default)}"

            parts.append(param_str)

        return ", ".join(parts)

    @property
    def kwargs_dict_str(self) -> str:
        """Generate kwargs for RPC call."""
        params = [p for p in self.params if p not in ('self', 'cls')]
        return ", ".join(f'{p}={p}' for p in params)


def parse_exposed_functions(source_code: str) -> list[FunctionSignature]:
    """
    Parse functions decorated with @expose from source code.

    Args:
        source_code: Python source code with @expose decorators

    Returns:
        List of parsed function signatures
    """
    functions = []

    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        raise ValueError(f"Failed to parse source code: {e}")

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue

        # Check for @expose decorator
        has_expose = any(
            (isinstance(d, ast.Name) and d.id == 'expose') or
            (isinstance(d, ast.Attribute) and d.attr == 'expose')
            for d in node.decorator_list
        )

        if not has_expose:
            continue

        # Extract function info
        name = node.name
        params = []
        param_types = {}
        defaults = {}

        # Parse arguments
        for arg in node.args.args:
            param_name = arg.arg
            params.append(param_name)

            # Type annotation
            if arg.annotation:
                param_types[param_name] = ast.unparse(arg.annotation)

        # Parse defaults
        num_defaults = len(node.args.defaults)
        if num_defaults > 0:
            default_params = params[-num_defaults:]
            for param, default_node in zip(default_params, node.args.defaults):
                try:
                    defaults[param] = ast.literal_eval(default_node)
                except (ValueError, TypeError):
                    # Can't evaluate, use string representation
                    defaults[param] = ast.unparse(default_node)

        # Extract docstring
        docstring = ast.get_docstring(node) or f"Call {name}()"

        functions.append(FunctionSignature(
            name=name,
            params=params,
            param_types=param_types,
            defaults=defaults,
            docstring=docstring,
        ))

    return functions


def generate_rpc_client(
    name: str,
    source_code: str,
    rpc_url: str,
) -> str:
    """
    Generate client library code that calls RPC server.

    Args:
        name: Library name
        source_code: Server code with @expose decorators
        rpc_url: RPC endpoint URL

    Returns:
        Python code for client library
    """
    functions = parse_exposed_functions(source_code)

    if not functions:
        raise ValueError("No @expose decorated functions found in source code")

    # Generate header
    client_code = f'''"""Auto-generated RPC client for {name}

Server: {rpc_url}
"""

import requests
from typing import Any, Optional


RPC_URL = "{rpc_url}"
RPC_TIMEOUT = 600


def _call_rpc(fn_name: str, **kwargs) -> Any:
    """Internal RPC caller."""
    resp = requests.post(
        RPC_URL,
        json={{"fn": fn_name, "kwargs": kwargs}},
        timeout=RPC_TIMEOUT,
    )
    resp.raise_for_status()

    data = resp.json()
    if not data.get("ok"):
        error_msg = data.get("error", "Unknown error")
        traceback = data.get("traceback", "")
        if traceback:
            error_msg += f"\\n\\nRemote traceback:\\n{{traceback}}"
        raise RuntimeError(error_msg)

    return data["result"]


'''

    # Generate function stubs
    for func in functions:
        client_code += f'''
def {func.name}({func.signature_str}):
    """{func.docstring}"""
    return _call_rpc("{func.name}", {func.kwargs_dict_str})

'''

    return client_code


def generate_rpc_prompt(
    name: str,
    source_code: str,
    rpc_url: str,
) -> str:
    """
    Generate prompt documentation describing RPC interface.

    Args:
        name: Interface name
        source_code: Server code with @expose decorators
        rpc_url: RPC endpoint URL

    Returns:
        Markdown documentation string
    """
    functions = parse_exposed_functions(source_code)

    if not functions:
        return f"# {name} Interface\n\nNo functions available."

    prompt = f"""# {name} Interface

Available via RPC at: {rpc_url}

## Functions

"""

    for func in functions:
        prompt += f"### `{func.name}({func.signature_str})`\n\n"
        prompt += f"{func.docstring}\n\n"

    return prompt
