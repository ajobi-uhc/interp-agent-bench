"""
Clean prompt builder - assembles agent prompts from scaffold components.

No nested conditionals. Clear assembly logic.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class AgentPrompts:
    """Complete prompt package for the agent."""
    system_prompt: str
    user_prompt: str

    def save_to_workspace(self, workspace_path: Path):
        """Write prompts to workspace for logging."""
        (workspace_path / "system_prompt.md").write_text(self.system_prompt)
        (workspace_path / "user_prompt.md").write_text(self.user_prompt)
        print(f"💾 Prompts saved to: {workspace_path}")

    def print_summary(self):
        """Print prompt statistics."""
        sys_chars = len(self.system_prompt)
        user_chars = len(self.user_prompt)
        print(f"\n📝 Prompt Summary:")
        print(f"   System: {sys_chars:,} chars (~{sys_chars//4:,} tokens)")
        print(f"   User:   {user_chars:,} chars (~{user_chars//4:,} tokens)")


def load_scaffold_file(filename: str) -> str:
    """Load a markdown file from scaffold/ directory."""
    scaffold_dir = Path(__file__).parent / "scaffold"
    file_path = scaffold_dir / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Scaffold file not found: {file_path}")
    return file_path.read_text().strip()


def format_techniques_as_markdown(techniques: dict) -> str:
    """
    Format technique dict into markdown documentation with full source code.

    Args:
        techniques: Dict of {name: TechniqueMethod} from technique_loader

    Returns:
        Formatted markdown string with full technique implementations
    """
    parts = ["## Example Techniques\n"]
    parts.append("Here are some example interpretability techniques you can use as reference.\n")
    parts.append("You can use these directly or modify them for your experiments.\n")
    parts.append("Each technique is a function that takes `(model, tokenizer, *args, **kwargs)` as parameters:\n")

    for name, method in techniques.items():
        parts.append(f"\n### `{name}`\n")
        parts.append(f"\n**Description**: {method.description}\n")
        
        # Include docstring if available for more details
        if method.docstring and method.docstring != method.description:
            parts.append(f"\n**Details**: {method.docstring}\n")
        
        # Include full source code (removing class indentation and decorator)
        # The code attribute has class indentation and @modal.method() decorator
        # Remove the decorator line and unindent for standalone function format
        code_lines = method.code.strip().split('\n')
        
        # Skip @modal.method() decorator line
        if code_lines and '@modal.method()' in code_lines[0]:
            code_lines = code_lines[1:]
        
        # Remove 4-space indentation (class method indentation)
        unindented_lines = []
        for line in code_lines:
            if line.startswith('    '):
                unindented_lines.append(line[4:])
            else:
                unindented_lines.append(line)
        
        # Replace self with model, tokenizer parameters
        source_code = '\n'.join(unindented_lines)
        # Replace self parameter with (model, tokenizer) in function signature
        source_code = source_code.replace('def ' + name + '(self,', 'def ' + name + '(model, tokenizer,', 1)
        source_code = source_code.replace('def ' + name + '(self)', 'def ' + name + '(model, tokenizer)', 1)
        # Replace self.model and self.tokenizer in body
        source_code = source_code.replace('self.model', 'model')
        source_code = source_code.replace('self.tokenizer', 'tokenizer')
        
        parts.append(f"\n**Full Implementation**:\n```python\n")
        parts.append(source_code.strip())
        parts.append("\n```\n")
        
        parts.append(f"\n**Usage with InterpClient**:\n```python\n")
        parts.append(f"result = client.run({name}, ...)\n")
        parts.append("```\n")

    return "\n".join(parts)


def build_system_prompt(skills: list = None) -> str:
    """
    Build system prompt from base instructions + skills.

    Components assembled:
    1. base_instructions.md (always) - includes general Skills explanation
    2. Specific Skills documentation (if skills provided)

    Args:
        skills: List of skill names to document

    Returns:
        Complete system prompt string
    """
    parts = []

    # Component 1: Base instructions (always)
    # Includes: MCP tools, workflow, Skills concept
    parts.append(load_scaffold_file("base_instructions.md"))

    # Component 2: Specific Skills documentation (conditional)
    if skills:
        from pathlib import Path
        from interp_infra.skills.loader import Skill

        for skill_name in skills:
            try:
                skill_file = Path(__file__).parent / "interp_infra" / "skills" / f"{skill_name}.md"
                if skill_file.exists():
                    skill = Skill(skill_file)
                    parts.append("\n\n")
                    parts.append(f"### Skill: {skill.name}\n")
                    parts.append(f"**Description**: {skill.description}\n")
                    parts.append(skill.get_system_prompt(include_source=False))
            except Exception as e:
                print(f"Warning: Could not load skill {skill_name}: {e}")

    return "\n".join(parts)


def build_user_prompt(
    task: str,
    agent_provider: Optional[str] = None,
    session_id: Optional[str] = None,
) -> str:
    """
    Build user prompt from task.

    Components assembled:
    1. Session connection info (if session_id provided)
    2. Task (always)
    3. OpenAI completion instructions (OpenAI only)

    Args:
        task: The research task to perform
        agent_provider: The agent provider (claude/openai)
        session_id: Pre-warmed session ID (if available)

    Returns:
        Complete user prompt string
    """
    parts = []

    # Component 1: Session connection (if pre-warmed session available)
    if session_id:
        parts.append("## Pre-warmed Environment\n")
        parts.append(f"A GPU environment with models already loaded is ready for you.\n")
        parts.append(f"**IMPORTANT**: Start by attaching to the pre-warmed session:\n")
        parts.append(f"```\n")
        parts.append(f'attach_to_session(session_id="{session_id}")\n')
        parts.append(f"```\n")
        parts.append(f"This will give you instant access to the loaded models without waiting.\n")
        parts.append(f"Do NOT use `start_new_session` - the environment is already ready!\n")
        parts.append("\n")

    # Component 2: Task (always)
    parts.append(task.strip())

    # Component 3: OpenAI completion instructions
    if agent_provider == "openai":
        parts.append("\n\n---\n")
        parts.append("## Task Completion\n")
        parts.append("You can output text and use tools freely during your work.")
        parts.append("When you are completely finished, return a TaskCompletion object with:")
        parts.append('- `status`: "TASK_DONE"')
        parts.append("- `summary`: Brief summary of what you accomplished")
        parts.append("\n**Only return TaskCompletion when you are truly done with all work.**")

    return "\n".join(parts)


def build_agent_prompts(
    task: str,
    agent_provider: Optional[str] = None,
    session_id: Optional[str] = None,
    skills: Optional[list] = None,
) -> AgentPrompts:
    """
    Main entry point: Build complete agent prompts.

    Args:
        task: The research task
        agent_provider: Agent provider (claude/openai)
        session_id: Pre-warmed session ID from deployment (if available)
        skills: List of skill names available in the environment

    Returns:
        AgentPrompts with system_prompt and user_prompt
    """
    print("\n" + "="*70)
    print("🔨 BUILDING AGENT PROMPTS")
    print("="*70)
    print(f"   agent_provider: {agent_provider or 'claude'}")
    print(f"   session_id: {session_id or 'none (will start fresh)'}")
    print(f"   skills: {', '.join(skills) if skills else 'none'}")
    print("="*70)

    system_prompt = build_system_prompt(skills=skills)

    user_prompt = build_user_prompt(
        task=task,
        agent_provider=agent_provider,
        session_id=session_id,
    )

    prompts = AgentPrompts(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    return prompts
