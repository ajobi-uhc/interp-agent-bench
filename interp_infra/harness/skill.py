"""Skill definition - compatible with Claude Code skills."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Skill:
    """
    A skill that extends agent capabilities.

    Compatible with Claude Code SKILL.md format.
    Can be loaded into different session types differently.
    """
    name: str
    prompt: str  # The main instructions (from SKILL.md body)
    description: str = ""  # Short description (from frontmatter)
    code: Optional[str] = None  # Optional code.py contents
    path: Optional[Path] = None  # Original path for accessing other files

    @classmethod
    def from_dir(cls, path: str | Path) -> "Skill":
        """
        Load skill from a directory (Claude Code format).

        Expected structure:
            skill-name/
                SKILL.md      # Required
                code.py       # Optional
                reference.md  # Optional
                scripts/      # Optional
        """
        skill_dir = Path(path)
        skill_md = skill_dir / "SKILL.md"

        if not skill_md.exists():
            raise FileNotFoundError(f"SKILL.md not found in {path}")

        content = skill_md.read_text()

        # Parse frontmatter
        name = skill_dir.name
        description = ""
        prompt = content

        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                frontmatter = parts[1]
                prompt = parts[2].strip()

                for line in frontmatter.strip().split("\n"):
                    if line.startswith("name:"):
                        name = line.split(":", 1)[1].strip()
                    elif line.startswith("description:"):
                        description = line.split(":", 1)[1].strip()

        # Load code.py if exists
        code = None
        code_py = skill_dir / "code.py"
        if code_py.exists():
            code = code_py.read_text()

        return cls(
            name=name,
            prompt=prompt,
            description=description,
            code=code,
            path=skill_dir,
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "Skill":
        """Load skill from a single SKILL.md file."""
        path = Path(path)
        return cls.from_dir(path.parent) if path.name == "SKILL.md" else cls.from_dir(path)

    @classmethod
    def from_prompt(cls, name: str, prompt: str) -> "Skill":
        """Create a prompt-only skill."""
        return cls(name=name, prompt=prompt)

    @classmethod
    def from_code(cls, name: str, code: str, prompt: str = "") -> "Skill":
        """Create a code skill inline."""
        return cls(name=name, prompt=prompt, code=code)

    def get_file(self, filename: str) -> Optional[str]:
        """Get contents of a supporting file in the skill directory."""
        if not self.path:
            return None
        file_path = self.path / filename
        return file_path.read_text() if file_path.exists() else None
