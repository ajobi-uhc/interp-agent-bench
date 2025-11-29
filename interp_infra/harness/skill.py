"""Skill definition for agent capabilities."""

from dataclasses import dataclass
from pathlib import Path
from ..execution.notebook_session import NotebookSession


@dataclass
class Skill:
    """A skill is Python code + prompt that extends agent capabilities."""
    name: str
    code: str  # Python code to exec in kernel
    prompt: str  # Description for agent's system prompt
    preload: bool = True  # Whether to preload code into kernel

    @classmethod
    def from_file(cls, path: str) -> "Skill":
        """Load skill from markdown file with YAML frontmatter."""
        import re

        content = Path(path).read_text()

        # Split frontmatter and markdown
        parts = content.split('---\n', 2)
        if len(parts) < 3:
            raise ValueError(f"Invalid skill format: {path}")

        frontmatter = parts[1]
        markdown = parts[2]

        # Extract name and description
        name = re.search(r'name:\s*(.+)', frontmatter).group(1).strip()
        desc_match = re.search(r'description:\s*(.+)', frontmatter)
        description = desc_match.group(1).strip() if desc_match else ""

        # Check preload flag
        preload_match = re.search(r'preload:\s*(.+)', frontmatter)
        preload = preload_match.group(1).strip().lower() == 'true' if preload_match else True

        # Extract Python code blocks only from ## Usage section
        usage_section = re.search(r'## Usage\s*\n(.*?)(?=\n##|\Z)', markdown, re.DOTALL)
        if usage_section:
            code_blocks = re.findall(r'```python\n(.*?)```', usage_section.group(1), re.DOTALL)
            code = '\n\n'.join(code_blocks)
        else:
            code = ''

        # Use markdown as prompt
        prompt = f"{description}\n\n{markdown}"

        return cls(name=name, code=code, prompt=prompt, preload=preload)
    
    def load_into_session(self, session: NotebookSession):
        """Load the skill code into the notebook session."""
        if self.preload and self.code:
            session.exec(self.code, hidden=True)

