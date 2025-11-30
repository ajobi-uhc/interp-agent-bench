"""Extension system - code + docs that extend agent capabilities."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Extension:
    """
    Self-contained code + documentation that extends agent capabilities.

    Can be:
    - Interpretability techniques (steering, SAE probing)
    - Library helpers (TransformerLens wrappers)
    - Utility functions (data processing)
    - Domain-specific tools (genomics analysis)
    - Workflow guides (prompt-only, no code)

    Usage:
        # From code
        ext = Extension(
            name="steering",
            code="def compute_steering_vector(...)...",
            docs="# Steering Vectors\nCompute activation differences..."
        )

        # From directory (SKILL.md format)
        ext = Extension.from_dir("./techniques/steering-vectors")

        # From Python file
        ext = Extension.from_file("./helpers.py")
    """
    name: str
    code: str = ""  # Implementation (optional)
    docs: str = ""  # Documentation/prompts
    files: dict[str, str] = field(default_factory=dict)  # Supporting files
    path: Optional[Path] = None  # Original path if loaded from directory

    @classmethod
    def from_dir(cls, path: str | Path) -> "Extension":
        """
        Load extension from a directory (SKILL.md format compatible).

        Expected structure:
            extension-name/
                SKILL.md      # Required: docs (with optional frontmatter)
                code.py       # Optional: implementation
                reference.md  # Optional
                scripts/      # Optional

        Args:
            path: Path to directory

        Returns:
            Extension object
        """
        ext_dir = Path(path)
        skill_md = ext_dir / "SKILL.md"

        if not skill_md.exists():
            raise FileNotFoundError(f"SKILL.md not found in {path}")

        content = skill_md.read_text()

        # Parse frontmatter if present
        name = ext_dir.name
        docs = content

        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                frontmatter = parts[1]
                docs = parts[2].strip()

                # Extract name from frontmatter
                for line in frontmatter.strip().split("\n"):
                    if line.startswith("name:"):
                        name = line.split(":", 1)[1].strip()

        # Load code.py if exists
        code = ""
        code_py = ext_dir / "code.py"
        if code_py.exists():
            code = code_py.read_text()

        # Load supporting files
        files = {}
        for file_path in ext_dir.iterdir():
            if file_path.is_file() and file_path.name not in ["SKILL.md", "code.py"]:
                files[file_path.name] = file_path.read_text()

        return cls(
            name=name,
            code=code,
            docs=docs,
            files=files,
            path=ext_dir,
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "Extension":
        """
        Load extension from a Python file.

        Uses module docstring as docs, file content as code.

        Args:
            path: Path to Python file

        Returns:
            Extension object
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        content = file_path.read_text()
        name = file_path.stem

        # Extract docstring if present
        docs = ""
        if content.strip().startswith('"""') or content.strip().startswith("'''"):
            quote = '"""' if '"""' in content else "'''"
            parts = content.split(quote, 2)
            if len(parts) >= 3:
                docs = parts[1].strip()

        return cls(
            name=name,
            code=content,
            docs=docs,
            path=file_path,
        )

    @classmethod
    def from_code(cls, name: str, code: str, docs: str = "") -> "Extension":
        """
        Create extension from inline code.

        Args:
            name: Extension name
            code: Python code
            docs: Documentation/prompts

        Returns:
            Extension object
        """
        return cls(name=name, code=code, docs=docs)

    @classmethod
    def from_prompt(cls, name: str, docs: str) -> "Extension":
        """
        Create prompt-only extension (no code).

        Args:
            name: Extension name
            docs: Documentation/prompts

        Returns:
            Extension object
        """
        return cls(name=name, docs=docs)

    def get_file(self, filename: str) -> Optional[str]:
        """
        Get contents of a supporting file.

        Args:
            filename: Name of file

        Returns:
            File contents or None if not found
        """
        if filename in self.files:
            return self.files[filename]

        if self.path:
            file_path = self.path / filename
            if file_path.exists():
                return file_path.read_text()

        return None
