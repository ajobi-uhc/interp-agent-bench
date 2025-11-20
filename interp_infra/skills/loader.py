"""Skill loading and management system.

Skills are markdown files with YAML frontmatter that define interpretability
techniques. They can include Python code that gets pre-loaded into the kernel
namespace, making techniques easily accessible to agents.
"""

import re
from pathlib import Path
from typing import Dict, List, Callable, Any, Optional
import yaml


class Skill:
    """A loaded interpretability skill."""

    def __init__(self, skill_file: Path):
        """
        Load a skill from a .md file.

        Args:
            skill_file: Path to skill .md file (e.g., steering-vectors.md)

        Raises:
            ValueError: If file missing or malformed
        """
        self.skill_file = skill_file

        if not self.skill_file.exists():
            raise ValueError(f"Skill file not found: {skill_file}")

        # Parse SKILL.md
        with open(self.skill_file) as f:
            content = f.read()

        # Extract YAML frontmatter
        if not content.startswith("---"):
            raise ValueError(f"Skill {skill_file.name} missing YAML frontmatter (must start with ---)")

        parts = content.split("---", 2)
        if len(parts) < 3:
            raise ValueError(f"Skill {skill_file.name} has malformed frontmatter")

        self.metadata = yaml.safe_load(parts[1])
        self.instructions = parts[2].strip()

        # Required metadata fields
        if "name" not in self.metadata:
            raise ValueError(f"Skill {skill_file.name} missing 'name' in frontmatter")
        if "description" not in self.metadata:
            raise ValueError(f"Skill {skill_file.name} missing 'description' in frontmatter")

        self.name = self.metadata["name"]
        self.description = self.metadata["description"]
        self.preload = self.metadata.get("preload", True)  # Default: pre-load functions

        # Extract functions from code blocks
        self.functions = self._extract_functions()

    def _extract_functions(self) -> Dict[str, str]:
        """
        Extract Python function definitions from ```python code blocks.

        Returns:
            Dict mapping function_name -> function_code
        """
        functions = {}

        # Find all Python code blocks
        code_blocks = re.findall(r'```python\n(.*?)```', self.instructions, re.DOTALL)

        for block in code_blocks:
            # Find function definitions in this block
            func_matches = list(re.finditer(r'^def (\w+)\(', block, re.MULTILINE))

            for i, match in enumerate(func_matches):
                func_name = match.group(1)
                func_start = match.start()

                # Find end of function (next 'def' at same indent level, or end of block)
                if i + 1 < len(func_matches):
                    func_end = func_matches[i + 1].start()
                else:
                    func_end = len(block)

                func_code = block[func_start:func_end].strip()
                functions[func_name] = func_code

        return functions

    def load_into_namespace(self, namespace: Dict[str, Any]) -> Dict[str, Callable]:
        """
        Execute function code and inject into namespace.

        This runs the function definitions in the context of the provided namespace,
        so functions can access things like `model`, `tokenizer`, etc.

        Args:
            namespace: Kernel globals (needs model, tokenizer, etc.)

        Returns:
            Dict of function_name -> function object
        """
        loaded_functions = {}

        for func_name, func_code in self.functions.items():
            try:
                # Execute function definition in namespace context
                exec(func_code, namespace)
                loaded_functions[func_name] = namespace[func_name]
            except Exception as e:
                print(f"  Warning: Failed to load function {func_name}: {e}")

        return loaded_functions

    def get_system_prompt(self, include_source: bool = False) -> str:
        """
        Get instructions to inject into agent system prompt.

        Args:
            include_source: If True, include full function source code.
                          If False, just list available functions.

        Returns:
            Formatted markdown for system prompt
        """
        if include_source:
            # Full instructions with code
            return f"\n## {self.name.replace('-', ' ').title()} Skill\n\n{self.instructions}"
        else:
            # Just list available functions (source available via inspect)
            func_list = "\n".join([f"- `{name}()`" for name in self.functions.keys()])

            if not func_list:
                # No functions, just provide description
                return f"""
## {self.name.replace('-', ' ').title()} Skill

{self.description}

See the full skill documentation for details.
"""

            return f"""
## {self.name.replace('-', ' ').title()} Skill

{self.description}

**Available functions** (pre-loaded in your namespace):
{func_list}

To see full documentation and source code for any function:
```python
help(function_name)
# or
import inspect
print(inspect.getsource(function_name))
```
"""


class SkillLoader:
    """Load and manage interpretability skills."""

    def __init__(self, skills_dir: Optional[Path] = None):
        """
        Initialize skill loader.

        Args:
            skills_dir: Directory containing skill subdirectories.
                       Defaults to interp_infra/skills/
        """
        if skills_dir is None:
            skills_dir = Path(__file__).parent
        self.skills_dir = skills_dir
        self._skills: Dict[str, Skill] = {}

    def load_skills(
        self,
        skill_names: List[str],
        namespace: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Load skills and optionally inject functions into namespace.

        Args:
            skill_names: List of skill names to load (e.g., ["steering-vectors"])
            namespace: Kernel namespace (with model, tokenizer, etc.)

        Returns:
            Dict of loaded functions (function_name -> function object)
        """
        loaded_functions = {}

        for name in skill_names:
            skill_file = self.skills_dir / f"{name}.md"
            if not skill_file.exists():
                print(f"  Warning: Skill '{name}.md' not found in {self.skills_dir}, skipping")
                continue

            try:
                skill = Skill(skill_file)
                self._skills[skill.name] = skill

                if skill.preload:
                    # Load functions into namespace
                    funcs = skill.load_into_namespace(namespace)
                    loaded_functions.update(funcs)
                    print(f"  Loaded skill: {skill.name} ({len(funcs)} functions)")
                else:
                    # Just provide instructions, don't pre-load
                    print(f"  Loaded skill: {skill.name} (instructions only)")
            except Exception as e:
                print(f"  Error loading skill '{name}': {e}")

        return loaded_functions

    def get_system_prompt(self, skill_names: List[str]) -> str:
        """
        Get combined system prompt for loaded skills.

        Args:
            skill_names: Names of skills to include in prompt

        Returns:
            Formatted markdown with skill documentation
        """
        prompts = []
        for name in skill_names:
            if name not in self._skills:
                continue

            skill = self._skills[name]
            # If functions are pre-loaded, just list them
            # Otherwise, include full source
            include_source = not skill.preload
            prompts.append(skill.get_system_prompt(include_source))

        if not prompts:
            return ""

        return "\n\n".join(prompts)

    def list_available_skills(self) -> List[str]:
        """
        List all available skill names in the skills directory.

        Returns:
            List of skill names (without .md extension)
        """
        if not self.skills_dir.exists():
            return []

        return [
            f.stem for f in self.skills_dir.iterdir()
            if f.is_file() and f.suffix == ".md" and f.name != "README.md"
        ]
