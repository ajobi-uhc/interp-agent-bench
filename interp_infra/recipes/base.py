"""
Base recipe system: protocol definition and registry.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol


@dataclass
class RecipeConfig:
    """Configuration for a recipe.

    Attributes:
        name: Recipe identifier (e.g., "hidden_behavior", "steering")
        model_id: Optional primary model identifier
        extra: Recipe-specific parameters (flexible schema per recipe)
    """
    name: str
    model_id: Optional[str] = None
    extra: Dict[str, Any] = None

    def __post_init__(self):
        if self.extra is None:
            self.extra = {}


class EnvRecipe(Protocol):
    """Protocol for environment recipes.

    Recipes are responsible for building the experiment-specific namespace
    that gets injected into the Jupyter kernel before the agent's first cell.

    Recipes should:
    - Load models, datasets, hooks, tools
    - Create safe wrappers/facades
    - Return only what the agent should see

    Recipes should NOT:
    - Know about Modal, Jupyter, or deployment infrastructure
    - Handle git repos, file paths, or workspace setup (that's initialize_session)
    - Assume any specific file layout
    """

    def warm_init(self, cfg: RecipeConfig) -> Dict[str, Any]:
        """Initialize the environment inside the kernel.

        This runs before the agent's first notebook cell executes.

        Args:
            cfg: Recipe configuration

        Returns:
            Dictionary mapping variable names to objects.
            These will be injected into the kernel's global namespace.

        Example:
            >>> recipe = HiddenBehaviorRecipe()
            >>> ns = recipe.warm_init(RecipeConfig(name="hidden_behavior", model_id="gpt2"))
            >>> # ns = {"model": <TargetModel>, "task": "..."}
        """
        ...


# Global registry of recipes
RECIPES: Dict[str, type] = {}


def register_recipe(name: str):
    """Decorator to register a recipe class.

    Usage:
        @register_recipe("my_recipe")
        class MyRecipe:
            def warm_init(self, cfg: RecipeConfig) -> Dict[str, Any]:
                return {"foo": "bar"}
    """
    def decorator(cls):
        RECIPES[name] = cls
        return cls
    return decorator


def get_recipe(name: str) -> EnvRecipe:
    """Get a recipe instance by name.

    Args:
        name: Recipe identifier

    Returns:
        Recipe instance

    Raises:
        KeyError: If recipe not found
    """
    if name not in RECIPES:
        available = ", ".join(RECIPES.keys())
        raise KeyError(
            f"Recipe '{name}' not found. Available recipes: {available}"
        )
    return RECIPES[name]()
