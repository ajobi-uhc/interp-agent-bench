"""Library - importable Python code for agent's execution context."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..execution.session_base import SessionBase


@dataclass
class Library:
    """
    Importable Python library in agent's workspace.

    The agent can write: from {name} import ...

    Supports both single files and multi-file packages.

    Examples:
        # From single Python file → import utils
        lib = Library.from_file("utils.py")

        # From package directory → import my_lib.module
        lib = Library.from_directory("my_lib/")

        # From SKILL.md directory (loads code.py)
        lib = Library.from_skill_dir("skills/steering-hook")

        # Manual construction
        lib = Library(
            name="utils",
            files={"utils.py": "def helper(): ..."},
        )
    """

    name: str                           # Import name
    files: dict[str, str]               # Relative path → source code
    docs: str = ""                      # Optional documentation

    @property
    def is_single_file(self) -> bool:
        """True if this is a single-file module (not a package)."""
        return len(self.files) == 1 and not any("/" in p for p in self.files.keys())

    @classmethod
    def from_file(cls, path: str | Path, name: Optional[str] = None) -> "Library":
        """
        Load library from a .py file.

        Args:
            path: Path to Python file
            name: Library name (defaults to filename without .py)

        Returns:
            Library instance

        Example:
            lib = Library.from_file("utils.py")
            # Agent can: import utils
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        code = path.read_text()
        name = name or path.stem

        # Extract module docstring if present
        docs = ""
        if code.strip().startswith(('"""', "'''")):
            quote = '"""' if '"""' in code else "'''"
            parts = code.split(quote, 2)
            if len(parts) >= 3:
                docs = parts[1].strip()

        return cls(name=name, files={f"{name}.py": code}, docs=docs)

    @classmethod
    def from_directory(cls, path: str | Path, name: Optional[str] = None) -> "Library":
        """
        Load library from a directory (Python package).

        Directory must contain __init__.py to be a valid package.

        Args:
            path: Path to package directory
            name: Library name (defaults to directory name)

        Returns:
            Library instance

        Example:
            # Directory structure:
            # my_lib/
            #   __init__.py
            #   core.py
            #   utils.py

            lib = Library.from_directory("my_lib/")
            # Agent can: from my_lib import core
            #            from my_lib.utils import helper
        """
        path = Path(path)
        if not path.is_dir():
            raise ValueError(f"Not a directory: {path}")

        init_file = path / "__init__.py"
        if not init_file.exists():
            raise ValueError(f"Package must have __init__.py: {path}")

        # Collect all .py files recursively
        files = {}
        for py_file in path.rglob("*.py"):
            rel_path = py_file.relative_to(path)
            files[str(rel_path)] = py_file.read_text()

        if not files:
            raise ValueError(f"No Python files found in {path}")

        name = name or path.name

        # Extract docs from __init__.py
        docs = ""
        init_content = files.get("__init__.py", "")
        if init_content.strip().startswith(('"""', "'''")):
            quote = '"""' if '"""' in init_content else "'''"
            parts = init_content.split(quote, 2)
            if len(parts) >= 3:
                docs = parts[1].strip()

        return cls(name=name, files=files, docs=docs)

    @classmethod
    def from_skill_dir(cls, path: str | Path) -> "Library":
        """
        Load library from SKILL.md directory format.

        Expected structure:
            skills/steering-hook/
                code.py         # Required - library code
                SKILL.md        # Optional - documentation

        Args:
            path: Path to skill directory

        Returns:
            Library instance with code from code.py and docs from SKILL.md

        Example:
            lib = Library.from_skill_dir("skills/steering-hook")
            # Agent can: import steering_hook
        """
        path = Path(path)
        if not path.is_dir():
            raise ValueError(f"Not a directory: {path}")

        # Load code.py
        code_file = path / "code.py"
        if not code_file.exists():
            raise FileNotFoundError(f"No code.py in {path}")

        code = code_file.read_text()
        name = path.name.replace("-", "_")  # steering-hook → steering_hook

        # Load SKILL.md if present
        docs = ""
        skill_md = path / "SKILL.md"
        if skill_md.exists():
            content = skill_md.read_text()

            # Skip YAML frontmatter if present
            if content.startswith("---"):
                parts = content.split("---", 2)
                docs = parts[2].strip() if len(parts) >= 3 else content
            else:
                docs = content

        return cls(name=name, files={f"{name}.py": code}, docs=docs)

    def install_in(self, session: "SessionBase"):
        """
        Install this library into a session's execution context.

        Makes the library importable via standard Python imports.

        Single-file libraries:
            - Writes {name}.py to workspace
            - Agent can: import {name}

        Multi-file packages:
            - Writes {name}/ directory with all files
            - Agent can: from {name}.module import func

        Works in all session types by writing to workspace filesystem.

        Args:
            session: Session to install into
        """
        from ..execution import NotebookSession, CLISession, LocalSession

        if isinstance(session, NotebookSession):
            self._install_notebook(session)
        elif isinstance(session, CLISession):
            self._install_cli(session)
        elif isinstance(session, LocalSession):
            self._install_local(session)
        else:
            raise TypeError(f"Unsupported session type: {type(session)}")

    def _write_to_sandbox(self, sandbox, base_path: str):
        """Write library files to sandbox workspace."""
        if self.is_single_file:
            file_path = f"{base_path}/{self.name}.py"
            code = list(self.files.values())[0]
            sandbox.write_file(file_path, code)
        else:
            for rel_path, code in self.files.items():
                file_path = f"{base_path}/{self.name}/{rel_path}"
                sandbox.write_file(file_path, code)

    def _write_to_local(self, workspace_path: Path):
        """Write library files to local workspace."""
        if self.is_single_file:
            lib_file = workspace_path / f"{self.name}.py"
            code = list(self.files.values())[0]
            lib_file.write_text(code)
        else:
            for rel_path, code in self.files.items():
                file_path = workspace_path / self.name / rel_path
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(code)

    def _install_notebook(self, session):
        """Install in notebook session by writing to sandbox workspace."""
        self._write_to_sandbox(session.sandbox, "/workspace")
        # Ensure /workspace is in Python path (usually is by default)
        session.exec("import sys; sys.path.insert(0, '/workspace') if '/workspace' not in sys.path else None", hidden=True)

    def _install_cli(self, session):
        """Install in CLI session by writing to sandbox workspace."""
        self._write_to_sandbox(session.sandbox, "/workspace")

    def _install_local(self, session):
        """Install in local session by writing to local workspace."""
        self._write_to_local(session.workspace_path)
