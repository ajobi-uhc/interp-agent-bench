"""Session initialization - actual Python functions, not string generation.

This module runs in the Modal container to set up the session before Jupyter starts.
"""

import subprocess
from pathlib import Path
from typing import Dict, Any
from ..config.schema import ExperimentConfig, ModelConfig


def load_models(config: ExperimentConfig, obfuscate: bool = False) -> Dict[str, Any]:
    """
    Load models from config.

    Args:
        config: Experiment configuration
        obfuscate: If True, don't print model names

    Returns:
        Dictionary of variables to inject into kernel namespace
    """
    import torch
    import os
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Get HF token from environment (set by Modal secrets)
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        # Login to HuggingFace for gated models
        from huggingface_hub import login
        login(token=hf_token)

    namespace = {}

    if not config.models:
        return namespace

    for i, model_cfg in enumerate(config.models):
        var_name = "model" if len(config.models) == 1 else f"model_{i}"
        tok_name = "tokenizer" if len(config.models) == 1 else f"tokenizer_{i}"

        if model_cfg.custom_load_code:
            # Execute custom loading code
            if not obfuscate:
                print(f"Loading model {i} with custom code...")
            exec(model_cfg.custom_load_code, namespace)
        else:
            # Standard loading
            if not obfuscate:
                print(f"Loading {model_cfg.name}...")

            # Parse dtype
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
                "auto": "auto",
            }
            dtype = dtype_map.get(model_cfg.dtype, "auto")

            if model_cfg.is_peft:
                # Load base model then apply PEFT adapter
                from peft import PeftModel

                base_model = AutoModelForCausalLM.from_pretrained(
                    model_cfg.base_model,
                    device_map=model_cfg.device,
                    torch_dtype=dtype,
                    trust_remote_code=model_cfg.trust_remote_code,
                )
                model = PeftModel.from_pretrained(base_model, model_cfg.name)
                tokenizer = AutoTokenizer.from_pretrained(
                    model_cfg.base_model,
                    trust_remote_code=True
                )
            else:
                # Standard model loading
                model = AutoModelForCausalLM.from_pretrained(
                    model_cfg.name,
                    device_map=model_cfg.device,
                    torch_dtype=dtype,
                    trust_remote_code=model_cfg.trust_remote_code,
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    model_cfg.name,
                    trust_remote_code=True
                )

            namespace[var_name] = model
            namespace[tok_name] = tokenizer

            if not obfuscate:
                print(f"✅ Loaded on {model.device}")

    return namespace


def clone_github_repos(config: ExperimentConfig) -> Dict[str, Any]:
    """
    Clone GitHub repositories.

    Args:
        config: Experiment configuration

    Returns:
        Empty dict (no namespace additions)
    """
    if not config.github_repos:
        return {}

    workspace = Path("/workspace")
    workspace.mkdir(exist_ok=True)

    for repo in config.github_repos:
        repo_name = repo.split("/")[-1].replace(".git", "")
        repo_path = workspace / repo_name

        if not repo_path.exists():
            print(f"Cloning {repo}...")
            subprocess.run(
                ["git", "clone", repo, str(repo_path)],
                check=True,
                capture_output=True
            )
            print(f"✅ Cloned {repo_name}")

    return {}


def run_custom_startup(config: ExperimentConfig) -> Dict[str, Any]:
    """
    Run custom startup code from config.

    Args:
        config: Experiment configuration

    Returns:
        Dictionary of any new variables created by custom code
    """
    if not config.startup_code:
        return {}

    print("Running custom startup code...")
    namespace = {}
    exec(config.startup_code, namespace)

    # Filter out builtins
    return {k: v for k, v in namespace.items() if not k.startswith("__")}


def initialize_session(config: ExperimentConfig) -> Dict[str, Any]:
    """
    Run all initialization steps and return namespace for kernel.

    Args:
        config: Experiment configuration

    Returns:
        Dictionary of all variables to inject into kernel globals
    """
    print("🔄 Initializing session...")

    namespace = {}

    # Run all initialization steps
    namespace.update(clone_github_repos(config))

    # Check if any model wants obfuscation
    obfuscate = any(m.obfuscate_name for m in config.models)
    namespace.update(load_models(config, obfuscate=obfuscate))

    namespace.update(run_custom_startup(config))

    print("✅ Session initialization complete")
    return namespace
