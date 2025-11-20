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
class EnvironmentConfig:
    """Configuration for an environment.

    Attributes:
        name: Environment identifier (e.g., "model", "api_access")
        model_id: Optional primary model identifier
        extra: Environment-specific parameters (flexible schema per environment)
    """
    name: str
    model_id: Optional[str] = None
    extra: Dict[str, Any] = None

    def __post_init__(self):
        if self.extra is None:
            self.extra = {}


class Environment(Protocol):
    """Protocol for experiment environments.

    Environments are responsible for:
    1. Defining what infrastructure to prepare (get_prewarm_plan)
    2. Building the experiment-specific namespace inside kernel (warm_init)

    Environments should:
    - Declare what models/repos to pre-download (in get_prewarm_plan)
    - Load models from cache, create wrappers, expose tools (in warm_init)
    - Return only what the agent should see

    Environments should NOT:
    - Execute sandbox.exec() or call Modal APIs
    - Download models or clone repos themselves
    - Know about Jupyter, Modal, or deployment infrastructure
    """

    def get_prewarm_plan(self, cfg: EnvironmentConfig) -> List[PrewarmTask]:
        """Define infrastructure tasks to execute before kernel starts.

        This runs in the parent process (ModalClient). The environment declares
        what models to download and what repos to clone. Infrastructure
        executes these tasks via sandbox.exec().

        Args:
            cfg: Environment configuration

        Returns:
            List of PrewarmTask objects describing what to prepare.

        Example:
            >>> environment = ModelEnvironment()
            >>> plan = environment.get_prewarm_plan(cfg)
            >>> # plan = [
            >>> #   PrewarmTask(kind="model", id="meta-llama/Llama-2-70b"),
            >>> #   PrewarmTask(kind="repo", id="org/repo"),
            >>> # ]
        """
        ...

    def warm_init(self, cfg: EnvironmentConfig) -> Dict[str, Any]:
        """Construct the environment inside the kernel from pre-downloaded cache.

        This runs inside the Jupyter kernel at startup, after infrastructure
        has pre-downloaded all models and cloned all repos.

        Args:
            cfg: Environment configuration

        Returns:
            Dictionary mapping variable names to objects.
            These will be injected into the kernel's global namespace.

        Example:
            >>> environment = ModelEnvironment()
            >>> ns = environment.warm_init(cfg)
            >>> # ns = {"model": <TargetModel>, "tokenizer": <tok>}
        """
        ...

    def get_default_skills(self) -> List[str]:
        """Get list of skills that should be loaded with this environment.

        This is optional - environments can return an empty list if they don't
        need any skills. Skills can also be specified in the experiment config.

        Returns:
            List of skill names (e.g., ["steering-vectors", "sae-latents"])
        """
        return []


# Global registry of environments
ENVIRONMENTS: Dict[str, type] = {}


def register_environment(name: str):
    """Decorator to register an environment class.

    Usage:
        @register_environment("my_environment")
        class MyEnvironment:
            def warm_init(self, cfg: EnvironmentConfig) -> Dict[str, Any]:
                return {"foo": "bar"}
    """
    def decorator(cls):
        ENVIRONMENTS[name] = cls
        return cls
    return decorator


def get_environment(name: str) -> Environment:
    """Get an environment instance by name.

    Args:
        name: Environment identifier

    Returns:
        Environment instance

    Raises:
        KeyError: If environment not found
    """
    if name not in ENVIRONMENTS:
        available = ", ".join(ENVIRONMENTS.keys())
        raise KeyError(
            f"Environment '{name}' not found. Available environments: {available}"
        )
    return ENVIRONMENTS[name]()
