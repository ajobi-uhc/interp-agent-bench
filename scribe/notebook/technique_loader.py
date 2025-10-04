"""Load and manage technique methods for dynamic ModelService construction."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Dict, Any


class TechniqueMethod:
    """Represents a single ModelService method from a technique file."""

    def __init__(
        self,
        name: str,
        code: str,
        docstring: str,
        signature: str,
        description: str,
    ):
        self.name = name
        self.code = code
        self.docstring = docstring
        self.signature = signature
        self.description = description


def load_technique_methods(techniques_dir: Path) -> Dict[str, TechniqueMethod]:
    """Load all technique methods from techniques directory.

    Each technique file should contain a function decorated with modal.method()
    that will be added to ModelService.

    Args:
        techniques_dir: Path to techniques directory

    Returns:
        Dictionary mapping method names to TechniqueMethod objects
    """
    methods = {}

    if not techniques_dir.exists():
        return methods

    for py_file in sorted(techniques_dir.glob("*.py")):
        if py_file.name.startswith("_"):
            continue

        try:
            method = parse_technique_file(py_file)
            if method:
                methods[method.name] = method
        except Exception as e:
            print(f"Warning: Failed to load technique {py_file.name}: {e}")

    return methods


def parse_technique_file(file_path: Path) -> TechniqueMethod | None:
    """Parse a technique file and extract the modal method.

    Expected format:
    ```python
    '''One-line description.'''

    def technique_name(self, arg1, arg2):
        '''Method docstring.'''
        # Implementation
        return result
    ```

    Args:
        file_path: Path to technique file

    Returns:
        TechniqueMethod if valid, None otherwise
    """
    content = file_path.read_text()

    # Parse the AST
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return None

    # Get module docstring
    module_docstring = ast.get_docstring(tree) or ""
    description = module_docstring.split("\n")[0] if module_docstring else ""

    # Find the main function (should be only one)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if not functions:
        return None

    func = functions[0]  # Take first function
    method_name = func.name
    method_docstring = ast.get_docstring(func) or ""

    # Extract signature
    signature_parts = [method_name, "(self"]
    for arg in func.args.args:
        if arg.arg == "self":
            continue
        # Get annotation if available
        if arg.annotation:
            annotation = ast.unparse(arg.annotation)
            signature_parts.append(f", {arg.arg}: {annotation}")
        else:
            signature_parts.append(f", {arg.arg}")
    signature_parts.append(")")
    signature = "".join(signature_parts)

    # Get the function source code (excluding the def line, we'll rebuild it)
    # Extract just the function body
    func_lines = content.split("\n")
    func_start = func.lineno - 1
    func_end = func.end_lineno

    # Get the function code including def
    func_code_lines = func_lines[func_start:func_end]

    # Add proper indentation for class method (2 levels)
    indented_lines = []
    for line in func_code_lines:
        if line.strip():  # Don't add extra indentation to empty lines
            indented_lines.append("    " + line)
        else:
            indented_lines.append(line)

    method_code = "\n".join(indented_lines)

    # Add @modal.method() decorator
    full_method_code = f"    @modal.method()\n{method_code}"

    return TechniqueMethod(
        name=method_name,
        code=full_method_code,
        docstring=method_docstring,
        signature=signature,
        description=description,
    )


def format_technique_for_prompt(method: TechniqueMethod) -> str:
    """Format a technique method for inclusion in agent prompt.

    Args:
        method: TechniqueMethod to format

    Returns:
        Formatted string for prompt
    """
    lines = [
        f"### `{method.name}`",
        "",
        method.description if method.description else method.docstring.split("\n")[0],
        "",
        "**Usage:**",
        f"```python",
        f"result = model_service.{method.name}.remote(...)",
        f"```",
        "",
        "**Signature:**",
        f"```python",
        method.signature,
        f"```",
        "",
    ]

    if method.docstring:
        lines.extend(["**Details:**", method.docstring, ""])

    return "\n".join(lines)


def build_modelservice_class_code(
    base_code: str,
    techniques: Dict[str, TechniqueMethod],
) -> str:
    """Build complete ModelService class code with injected technique methods.

    Args:
        base_code: Base ModelService class code (load_model, generate, etc.)
        techniques: Dictionary of technique methods to inject

    Returns:
        Complete ModelService class code
    """
    # Start with base code
    lines = base_code.split("\n")

    # Find the insertion point (before the final line of the class)
    # We'll insert before the last non-empty line
    insert_index = len(lines) - 1
    while insert_index > 0 and not lines[insert_index].strip():
        insert_index -= 1

    # Insert technique methods
    technique_lines = []
    for method in techniques.values():
        technique_lines.append("")
        technique_lines.extend(method.code.split("\n"))

    # Insert at the right place
    lines = lines[:insert_index] + technique_lines + lines[insert_index:]

    return "\n".join(lines)
