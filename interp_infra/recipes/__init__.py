"""
Recipe system for composable investigation environments.

Recipes define how to build experiment-specific namespaces inside the Jupyter kernel.
They handle model loading, hook attachment, dataset preparation, etc.
"""

from interp_infra.recipes.base import (
    RecipeConfig,
    EnvRecipe,
    register_recipe,
    get_recipe,
    RECIPES,
)

# Auto-import all recipes to register them
from interp_infra.recipes import model

__all__ = [
    "RecipeConfig",
    "EnvRecipe",
    "register_recipe",
    "get_recipe",
    "RECIPES",
]
