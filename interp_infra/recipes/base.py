"""
Base recipe system: protocol definition and registry.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Protocol


@dataclass
class PrewarmTask:
    """A task for infrastructure to execute before kernel starts.

    Infrastructure layer implements handlers for each task kind.
    No fallbacks - all task kinds must be explicitly supported.

    Attributes:
        kind: Type of task - "model" (HF snapshot_download) or "repo" (git clone)
        id: Model ID (e.g. "meta-llama/Llama-2-70b") or repo (e.g. "org/repo")
        extra: Task-specific metadata (e.g., for logging in obfuscated mode)
    """
    kind: Literal["model", "repo"]
    id: str
    extra: Dict[str, Any] = field(default_factory=dict)


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

    Recipes are responsible for:
    1. Defining what infrastructure to prepare (get_prewarm_plan)
    2. Building the experiment-specific namespace inside kernel (warm_init)

    Recipes should:
    - Declare what models/repos to pre-download (in get_prewarm_plan)
    - Load models from cache, create wrappers, expose tools (in warm_init)
    - Return only what the agent should see

    Recipes should NOT:
    - Execute sandbox.exec() or call Modal APIs
    - Download models or clone repos themselves
    - Know about Jupyter, Modal, or deployment infrastructure
    """

    def get_prewarm_plan(self, cfg: RecipeConfig) -> List[PrewarmTask]:
        """Define infrastructure tasks to execute before kernel starts.

        This runs in the parent process (ModalClient). The recipe declares
        what models to download and what repos to clone. Infrastructure
        executes these tasks via sandbox.exec().

        Args:
            cfg: Recipe configuration

        Returns:
            List of PrewarmTask objects describing what to prepare.

        Example:
            >>> recipe = ModelRecipe()
            >>> plan = recipe.get_prewarm_plan(cfg)
            >>> # plan = [
            >>> #   PrewarmTask(kind="model", id="meta-llama/Llama-2-70b"),
            >>> #   PrewarmTask(kind="repo", id="org/repo"),
            >>> # ]
        """
        ...

    def warm_init(self, cfg: RecipeConfig) -> Dict[str, Any]:
        """Construct the environment inside the kernel from pre-downloaded cache.

        This runs inside the Jupyter kernel at startup, after infrastructure
        has pre-downloaded all models and cloned all repos.

        Args:
            cfg: Recipe configuration

        Returns:
            Dictionary mapping variable names to objects.
            These will be injected into the kernel's global namespace.

        Example:
            >>> recipe = ModelRecipe()
            >>> ns = recipe.warm_init(cfg)
            >>> # ns = {"model": <TargetModel>, "tokenizer": <tok>}
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
