"""Extension system - code + docs that extend agent capabilities."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Extension:
    """Self-contained code + documentation that extends agent capabilities."""
    name: str
    code: str = ""
    docs: str = ""
    files: dict[str, str] = field(default_factory=dict)
    path: Optional[Path] = None

    @classmethod
    def from_dir(cls, path: str | Path) -> "Extension":
        """Load extension from a directory (SKILL.md format)."""
        ext_dir = Path(path)
        skill_md = ext_dir / "SKILL.md"

        if not skill_md.exists():
            raise FileNotFoundError(f"SKILL.md not found in {path}")

        content = skill_md.read_text()
        name = ext_dir.name
        docs = content

        # Parse frontmatter if present
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                for line in parts[1].strip().split("\n"):
                    if line.startswith("name:"):
                        name = line.split(":", 1)[1].strip()
                docs = parts[2].strip()

        # Load code.py if exists
        code = ""
        code_py = ext_dir / "code.py"
        if code_py.exists():
            code = code_py.read_text()

        # Load supporting files
        files = {
            f.name: f.read_text()
            for f in ext_dir.iterdir()
            if f.is_file() and f.name not in ["SKILL.md", "code.py"]
        }

        return cls(name=name, code=code, docs=docs, files=files, path=ext_dir)

    @classmethod
    def from_file(cls, path: str | Path) -> "Extension":
        """Load extension from a Python file."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        content = file_path.read_text()
        name = file_path.stem
        docs = ""

        # Extract docstring if present
        if content.strip().startswith(('"""', "'''")):
            quote = '"""' if '"""' in content else "'''"
            parts = content.split(quote, 2)
            if len(parts) >= 3:
                docs = parts[1].strip()

        return cls(name=name, code=content, docs=docs, path=file_path)

    @classmethod
    def from_code(cls, name: str, code: str, docs: str = "") -> "Extension":
        """Create extension from inline code."""
        return cls(name=name, code=code, docs=docs)

    @classmethod
    def from_prompt(cls, name: str, docs: str) -> "Extension":
        """Create prompt-only extension (no code)."""
        return cls(name=name, docs=docs)

    def get_file(self, filename: str) -> Optional[str]:
        """Get contents of a supporting file."""
        if filename in self.files:
            return self.files[filename]

        if self.path:
            file_path = self.path / filename
            if file_path.exists():
                return file_path.read_text()

        return None
