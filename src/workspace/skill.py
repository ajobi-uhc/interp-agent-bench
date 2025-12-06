"""Skill - Claude Code skill for agent discovery."""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..execution.session_base import SessionBase


@dataclass
class Skill:
    """
    Claude Code skill that gets installed to .claude/skills/ in agent workspace.

    Skills are discovered by Claude and used when relevant to user's request.

    Examples:
        # From source code with @expose functions
        skill = Skill.from_source(
            name="gpu-tools",
            description="GPU computation tools for model inference",
            source_code=interface_code
        )

        # Manual construction
        skill = Skill(
            name="pdf-tools",
            description="Extract and manipulate PDF files",
            content="# PDF Tools\n\nInstructions..."
        )
    """

    name: str           # Skill name (lowercase, hyphens only)
    description: str    # When to use this skill
    content: str        # Full SKILL.md markdown content

    @classmethod
    def from_source(
        cls,
        name: str,
        description: str,
        source_code: str,
        instructions: Optional[str] = None,
    ) -> "Skill":
        """
        Generate a Skill from Python source code with @expose functions.

        Parses the source to find exposed functions and creates SKILL.md
        documentation.

        Args:
            name: Skill name (lowercase-with-hyphens)
            description: Brief description + when to use
            source_code: Python code with @expose decorated functions
            instructions: Optional custom instructions (auto-generated if not provided)

        Returns:
            Skill instance ready to install

        Example:
            code = \"\"\"
            @expose
            def hello(name: str) -> dict:
                return {"message": f"Hello, {name}!"}
            \"\"\"

            skill = Skill.from_source(
                name="greetings",
                description="Generate personalized greetings",
                source_code=code
            )
        """
        from ..environment.utils.codegen import parse_exposed_functions

        # Parse functions
        functions = parse_exposed_functions(source_code)

        if not functions:
            raise ValueError("No @expose functions found in source code")

        # Generate function documentation
        func_docs = []
        for func in functions:
            func_docs.append(f"### `{func.name}({func.signature_str})`")
            func_docs.append(f"\n{func.docstring}\n")

        # Generate SKILL.md content
        if instructions:
            instructions_section = f"\n## Instructions\n\n{instructions}\n"
        else:
            instructions_section = "\n## Available Functions\n\n" + "\n".join(func_docs)

        content = f"""# {name.replace('-', ' ').title()}

{instructions_section}

## Usage

This skill provides {len(functions)} function(s) for your use:
{', '.join(f'`{f.name}()`' for f in functions)}
"""

        return cls(name=name, description=description, content=content)

    def to_skill_md(self) -> str:
        """
        Generate complete SKILL.md file with YAML frontmatter.

        Returns:
            Full SKILL.md content with frontmatter + markdown
        """
        return f"""---
name: {self.name}
description: {self.description}
---

{self.content}"""

    def install_in(self, session: "SessionBase"):
        """
        Install this skill into LOCAL workspace where agent runs.

        Skills are ALWAYS local - the agent reads them from the local filesystem,
        regardless of where code execution happens (sandbox or local).

        Creates .claude/skills/{name}/SKILL.md in the workspace directory.
        Makes the skill discoverable by Claude.

        Args:
            session: Session to install into
        """
        # All session types write to local workspace_path
        # Agent/harness runs locally and reads skills from there
        skill_dir = session.workspace_path / ".claude" / "skills" / self.name
        skill_dir.mkdir(parents=True, exist_ok=True)

        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(self.to_skill_md())
