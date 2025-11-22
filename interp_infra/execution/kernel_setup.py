"""
Kernel-side setup for model and skills initialization.

IMPORTANT: This module runs INSIDE the Modal sandbox (kernel process).
It handles the kernel-side warmup after the sandbox is created.

This module provides:
1. Model Loading: Load PyTorch models from cache into GPU memory
2. Skills Loading: Add interpretability technique functions
3. Validation: Ensure namespace is ready

The prewarm phase (downloads) happens separately in the parent process
before this code runs.
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path

from ..config.schema import ExperimentConfig, ModelConfig
from skills.loader import SkillLoader


class SetupPipeline:
    """
    Orchestrates the complete model + skills setup pipeline.

    This runs inside the Jupyter kernel after infrastructure preparation.
    It builds the namespace that gets injected into the agent's environment.

    Flow:
        1. Load models (if specified)
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
            >>> #   "call_api": <function>,
            >>> #   "workspace": "/workspace"
            >>> # }
        """
        print("=" * 60)
        print("SETUP PIPELINE")
        print("=" * 60)

        try:
            # Phase 1: Initialize base namespace (paths, metadata)
            namespace = self._initialize_base_namespace()

            # Phase 2: Load models (if specified)
            if self.config.environment.models:
                print("\n[Phase 1/3] Model Loading")
                print("-" * 60)
                try:
                    model_namespace = self._load_models()
                    namespace.update(model_namespace)
                except Exception as e:
                    raise RuntimeError(
                        f"Model loading failed:\n"
                        f"  Error: {str(e)}\n"
                        f"  Check that:\n"
                        f"    - Models are accessible (HF cache or volumes)\n"
                        f"    - GPU is available if using CUDA\n"
                        f"    - Dependencies are installed (torch, transformers, peft)\n"
                    ) from e
            else:
                print("\n[Phase 1/3] Model Loading")
                print("-" * 60)
                print("No models to load (API-only or custom setup)")

            # Phase 3: Load skills (add technique functions)
            print("\n[Phase 2/3] Skills Loading")
            print("-" * 60)
            skill_names = self.config.execution.skills
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

    def _load_models(self) -> Dict[str, Any]:
        """
        Load models from cache into GPU memory.

        Returns:
            Dictionary mapping variable names to model/tokenizer objects
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import base64
        import json

        # HF auth
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            from huggingface_hub import login
            login(token=hf_token, add_to_git_credential=False)

        # Check if models are in volumes
        model_paths = {}
        if "MODEL_PATHS_B64" in os.environ:
            model_paths_json = base64.b64decode(os.environ["MODEL_PATHS_B64"]).decode('utf-8')
            model_paths = json.loads(model_paths_json)

        namespace = {}

        for i, model_config in enumerate(self.config.environment.models):
            var_name = "model" if len(self.config.environment.models) == 1 else f"model_{i}"
            tok_name = "tokenizer" if len(self.config.environment.models) == 1 else f"tokenizer_{i}"

            # Custom loading code
            if model_config.custom_load_code:
                if not self.config.execution.obfuscate:
                    print(f"Loading model {i} with custom code...")
                exec(model_config.custom_load_code, namespace)
                continue

            # Standard loading
            model, tokenizer = self._load_single_model(model_config, model_paths)
            namespace[var_name] = model
            namespace[tok_name] = tokenizer

        return namespace

    def _load_single_model(self, model_config: ModelConfig, model_paths: Dict[str, str]):
        """Load a single model from config."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = model_config.name

        if self.config.execution.obfuscate:
            print("Loading target model...")
        else:
            print(f"Loading {model_id}...")

        # Determine model path (volume or HF cache)
        if model_id in model_paths:
            model_path = model_paths[model_id]
            print(f"   From volume: {model_path}")
        else:
            model_path = model_id
            cache_dir = os.environ.get("HF_HOME", "/root/.cache/huggingface")
            print(f"   From HF cache: {cache_dir}")

        # Parse dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "auto": "auto",
        }
        dtype = dtype_map.get(model_config.dtype, "auto")

        # Load PEFT or regular model
        if model_config.is_peft:
            from peft import PeftModel
            base_model_path = model_paths.get(model_config.base_model, model_config.base_model)

            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                device_map=model_config.device,
                torch_dtype=dtype,
                trust_remote_code=model_config.trust_remote_code,
            )
            model = PeftModel.from_pretrained(base_model, model_path)
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_path,
                trust_remote_code=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=model_config.device,
                torch_dtype=dtype,
                trust_remote_code=model_config.trust_remote_code,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )

        if self.config.execution.obfuscate:
            print("Model loaded")
        else:
            print(f"Loaded {model_id} on {model.device}")

        return model, tokenizer

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

        # If models were requested, check they exist
        if self.config.environment.models:
            if "model" not in namespace and "model_0" not in namespace:
                errors.append("Models configured but no 'model' found in namespace")

            # Check model is on correct device
            if "model" in namespace:
                model = namespace["model"]
                if hasattr(model, "device"):
                    print(f"  ✓ Model device: {model.device}")

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
        >>> from interp_infra.execution.kernel_setup import create_namespace
        >>> namespace = create_namespace(config)
        >>> globals().update(namespace)
    """
    pipeline = SetupPipeline(config)
    return pipeline.execute()
