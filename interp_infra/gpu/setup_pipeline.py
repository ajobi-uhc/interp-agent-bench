"""
Kernel-side setup for environment and skills initialization.

IMPORTANT: This module runs INSIDE the Modal sandbox (kernel process).
It handles the kernel-side warmup after the sandbox is created.

For the full orchestration (image building, prewarm, sandbox creation),
see interp_infra/orchestrator.py.

This module provides:
1. Environment Setup: Load models/APIs into memory
2. Skills Loading: Add interpretability technique functions
3. Validation: Ensure namespace is ready

The prewarm phase (downloads) happens separately in the parent process
before this code runs.
"""

from typing import Dict, Any, Optional
from pathlib import Path

from interp_infra.config.schema import ExperimentConfig
from interp_infra.environment.base import get_environment, Environment
from interp_infra.skills.loader import SkillLoader


class SetupPipeline:
    """
    Orchestrates the complete environment + skills setup pipeline.

    This runs inside the Jupyter kernel after infrastructure preparation.
    It builds the namespace that gets injected into the agent's environment.

    Flow:
        1. Load environment (model/API setup)
        2. Load skills (technique functions)
        3. Validate namespace
        4. Return complete namespace

    Note: Prewarm phase (downloads) happens separately in ModalClient._infra_prewarm()
    before the kernel starts. This pipeline only handles what runs inside the kernel.
    """

    def __init__(self, config: ExperimentConfig):
        """
        Initialize the setup pipeline.

        Args:
            config: Complete experiment configuration
        """
        self.config = config
        self.environment = get_environment(config.environment.name)
        self.skill_loader = SkillLoader()

    def execute(self) -> Dict[str, Any]:
        """
        Execute the complete setup pipeline.

        Returns:
            Dictionary mapping variable names to objects (the agent's namespace)

        Raises:
            RuntimeError: If any phase fails

        Example:
            >>> pipeline = SetupPipeline(config)
            >>> namespace = pipeline.execute()
            >>> # namespace = {
            >>> #   "model": <Model>,
            >>> #   "tokenizer": <Tokenizer>,
            >>> #   "apply_steering_vector": <function>,
            >>> #   "workspace": "/workspace"
            >>> # }
        """
        print("=" * 60)
        print("SETUP PIPELINE")
        print("=" * 60)

        try:
            # Phase 1: Initialize base namespace (paths, metadata)
            namespace = self._initialize_base_namespace()

            # Phase 2: Environment setup (load models/APIs)
            print("\n[Phase 1/3] Environment Setup")
            print("-" * 60)
            try:
                env_namespace = self.environment.warm_init(self.config.environment)
                namespace.update(env_namespace)
            except Exception as e:
                raise RuntimeError(
                    f"Environment setup failed for '{self.config.environment.name}':\n"
                    f"  Error: {str(e)}\n"
                    f"  Check that:\n"
                    f"    - Models are accessible (HF cache or volumes)\n"
                    f"    - API keys are set (for api_access environment)\n"
                    f"    - Dependencies are installed\n"
                ) from e

            # Phase 3: Load skills (add technique functions)
            print("\n[Phase 2/3] Skills Loading")
            print("-" * 60)
            skill_names = self.config.skills or self.environment.get_default_skills()
            if skill_names:
                print(f"Loading {len(skill_names)} skill(s): {', '.join(skill_names)}")
                try:
                    self.skill_loader.load_skills(skill_names, namespace)
                except Exception as e:
                    print(f"  Warning: Some skills failed to load: {e}")
                    print(f"  Continuing with partial skill loading...")
            else:
                print("No skills to load")

            # Phase 4: Validate
            print("\n[Phase 3/3] Validation")
            print("-" * 60)
            self._validate_namespace(namespace)

            print("\n" + "=" * 60)
            print("SETUP COMPLETE")
            print("=" * 60)
            print(f"Namespace contains {len(namespace)} objects:")
            for key in sorted(namespace.keys()):
                obj = namespace[key]
                obj_type = type(obj).__name__
                print(f"  • {key}: {obj_type}")
            print()

            return namespace

        except Exception as e:
            print("\n" + "=" * 60)
            print("SETUP FAILED")
            print("=" * 60)
            print(f"Error: {str(e)}")
            print()
            raise

    def _initialize_base_namespace(self) -> Dict[str, Any]:
        """
        Initialize lightweight base namespace (paths, metadata).

        This provides basic infrastructure globals without heavy operations.

        Returns:
            Dictionary with workspace path and experiment metadata
        """
        namespace = {}

        # Provide workspace path if repos were cloned
        workspace = Path("/workspace")
        if workspace.exists():
            namespace["workspace"] = str(workspace)

        # Add experiment metadata
        namespace["experiment_name"] = self.config.name

        return namespace

    def _validate_namespace(self, namespace: Dict[str, Any]) -> None:
        """
        Validate that the namespace is ready for the agent.

        Performs basic checks to ensure critical objects are present and valid.

        Args:
            namespace: The constructed namespace

        Raises:
            RuntimeError: If validation fails
        """
        errors = []

        # Check if namespace is empty
        if not namespace:
            errors.append("Namespace is empty - no objects loaded")

        # Environment-specific validations
        env_name = self.config.environment.name

        if env_name == "model":
            # For model environments, check that model exists
            if "model" not in namespace and "model_0" not in namespace:
                errors.append("Model environment but no 'model' found in namespace")

            # Check model is on correct device
            if "model" in namespace:
                model = namespace["model"]
                if hasattr(model, "device"):
                    print(f"  ✓ Model device: {model.device}")
                elif hasattr(model, "_model") and hasattr(model._model, "device"):
                    print(f"  ✓ Model device: {model._model.device}")

        elif env_name == "api_access":
            # For API environments, check that call_api exists
            if "call_api" not in namespace:
                errors.append("API access environment but no 'call_api' function found")
            else:
                print("  ✓ API access function available")

        # Report validation results
        if errors:
            error_msg = "\n".join(f"  - {err}" for err in errors)
            raise RuntimeError(f"Namespace validation failed:\n{error_msg}")

        print("  ✓ Namespace validation passed")


def create_namespace(config: ExperimentConfig) -> Dict[str, Any]:
    """
    Convenience function to create a complete agent namespace.

    This is the main entry point for setting up an experiment environment.

    Args:
        config: Experiment configuration

    Returns:
        Complete namespace ready for agent execution

    Example:
        >>> from interp_infra.gpu.setup_pipeline import create_namespace
        >>> namespace = create_namespace(config)
        >>> globals().update(namespace)
    """
    pipeline = SetupPipeline(config)
    return pipeline.execute()
