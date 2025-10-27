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


def build_system_prompt(
    needs_gpu: bool,
    selected_techniques: Optional[list[str]] = None,
    api_provider: Optional[str] = None,
) -> str:
    """
    Build system prompt from scaffold components.

    Components assembled:
    1. base_instructions.md (always)
    2. gpu_instructions.md (if needs_gpu=true) OR api_{provider}.md (if needs_gpu=false)
    3. Technique docs (if needs_gpu=true AND techniques selected)

    Args:
        needs_gpu: Whether agent has GPU model access
        selected_techniques: Optional list of technique names to include
        api_provider: API provider name (anthropic/openai/google) if needs_gpu=false

    Returns:
        Complete system prompt string
    """
    parts = []

    # Component 1: Base instructions (always)
    parts.append(load_scaffold_file("base_instructions.md"))

    # Component 2: GPU or API instructions (conditional)
    if needs_gpu:
        parts.append("\n\n")
        parts.append(load_scaffold_file("gpu_instructions.md"))

        # Component 3: Technique docs (conditional)
        if selected_techniques:
            from scribe.notebook.technique_loader import load_technique_methods
            techniques_dir = Path(__file__).parent / "techniques"
            all_techniques = load_technique_methods(techniques_dir)

            # Filter to selected
            techniques = {name: method for name, method in all_techniques.items()
                         if name in selected_techniques}

            if techniques:
                parts.append("\n\n")
                parts.append(format_techniques_as_markdown(techniques))
                print(f"📖 Included {len(techniques)} technique examples in prompt")
    else:
        # API mode - no GPU access
        if not api_provider:
            raise ValueError("api_provider required when needs_gpu=false")

        parts.append("\n\n")
        api_file = f"api_{api_provider}.md"
        parts.append(load_scaffold_file(api_file))
        print(f"🌐 Using API mode: {api_provider}")

    return "\n".join(parts)


def build_user_prompt(
    task: str,
    include_research_tips: bool = False,
    agent_provider: Optional[str] = None,
) -> str:
    """
    Build user prompt from task + optional research tips.

    Components assembled:
    1. Research tips (optional)
    2. Task (always)
    3. OpenAI session warning (OpenAI only)

    Args:
        task: The research task to perform
        include_research_tips: Whether to prepend research methodology tips
        agent_provider: The agent provider (claude/openai)

    Returns:
        Complete user prompt string
    """
    parts = []

    # Component 1: Research tips (optional)
    if include_research_tips:
        research_tips = load_scaffold_file("research_tips.md")
        parts.append("# ⚠️ CRITICAL RESEARCH METHODOLOGY ⚠️\n")
        parts.append(research_tips)
        parts.append("\n\n---\n")
        parts.append("\n# Your Task\n")
        print("📚 Included research tips in user prompt")

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
    needs_gpu: bool,
    selected_techniques: Optional[list[str]] = None,
    include_research_tips: bool = False,
    api_provider: Optional[str] = None,
    agent_provider: Optional[str] = None,
) -> AgentPrompts:
    """
    Main entry point: Build complete agent prompts.

    Args:
        task: The research task
        needs_gpu: Whether agent has GPU model access
        selected_techniques: Optional list of technique names to include as examples
        include_research_tips: Whether to include research methodology tips
        api_provider: API provider (anthropic/openai/google) if needs_gpu=false
        agent_provider: Agent provider (claude/openai)

    Returns:
        AgentPrompts with system_prompt and user_prompt
    """
    print("\n" + "="*70)
    print("🔨 BUILDING AGENT PROMPTS")
    print("="*70)
    print(f"   needs_gpu: {needs_gpu}")
    print(f"   api_provider: {api_provider or 'N/A'}")
    print(f"   agent_provider: {agent_provider or 'N/A'}")
    print(f"   techniques: {selected_techniques or 'none'}")
    print(f"   research_tips: {include_research_tips}")
    print("="*70)

    system_prompt = build_system_prompt(
        needs_gpu=needs_gpu,
        selected_techniques=selected_techniques,
        api_provider=api_provider,
    )

    user_prompt = build_user_prompt(
        task=task,
        include_research_tips=include_research_tips,
        agent_provider=agent_provider,
    )

    prompts = AgentPrompts(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    prompts.print_summary()
    return prompts
