"""Modular setup hooks that run before session initialization.

Each hook is a function that takes config and returns Python code to execute.
Hooks are composed together to build the complete startup sequence.
"""

from typing import List, Callable
from ..config.schema import ModelConfig, ExperimentConfig


SetupHook = Callable[[ExperimentConfig], str]


def model_loading_hook(config: ExperimentConfig) -> str:
    """Generate code to load models from config."""
    if not config.models:
        return ""

    lines = []
    lines.append("# Load models")
    lines.append("import torch")
    lines.append("from transformers import AutoModelForCausalLM, AutoTokenizer")
    lines.append("")

    obfuscate = any(m.obfuscate_name for m in config.models)

    for i, model in enumerate(config.models):
        var_name = "model" if len(config.models) == 1 else f"model_{i}"
        tok_name = "tokenizer" if len(config.models) == 1 else f"tokenizer_{i}"

        if model.custom_load_code:
            lines.append(f"# Custom loading for model {i}")
            lines.append(model.custom_load_code)
            lines.append("")
        else:
            if not obfuscate:
                lines.append(f"print(f'Loading {model.name}...')")

            dtype_map = {
                "bfloat16": "torch.bfloat16",
                "float16": "torch.float16",
                "float32": "torch.float32",
                "auto": '"auto"',
            }
            dtype = dtype_map.get(model.dtype, '"auto"')

            if model.is_peft:
                lines.append("from peft import PeftModel")
                lines.append(f"{var_name} = AutoModelForCausalLM.from_pretrained(")
                lines.append(f"    '{model.base_model}',")
                lines.append(f"    device_map='{model.device}',")
                lines.append(f"    torch_dtype={dtype},")
                lines.append(f"    trust_remote_code={model.trust_remote_code},")
                lines.append(")")
                lines.append(f"{var_name} = PeftModel.from_pretrained({var_name}, '{model.name}')")
                lines.append(f"{tok_name} = AutoTokenizer.from_pretrained('{model.base_model}', trust_remote_code=True)")
            else:
                lines.append(f"{var_name} = AutoModelForCausalLM.from_pretrained(")
                lines.append(f"    '{model.name}',")
                lines.append(f"    device_map='{model.device}',")
                lines.append(f"    torch_dtype={dtype},")
                lines.append(f"    trust_remote_code={model.trust_remote_code},")
                lines.append(")")
                lines.append(f"{tok_name} = AutoTokenizer.from_pretrained('{model.name}', trust_remote_code=True)")

            if not obfuscate:
                lines.append(f"print(f'✅ Loaded on {{str({var_name}.device)}}')")
            lines.append("")

    return "\n".join(lines)


def custom_startup_hook(config: ExperimentConfig) -> str:
    """Generate code from custom startup_code in config."""
    if not config.startup_code:
        return ""

    lines = []
    lines.append("# Custom startup code")
    lines.append(config.startup_code)
    lines.append("")
    return "\n".join(lines)


def github_repos_hook(config: ExperimentConfig) -> str:
    """Generate code to clone GitHub repos."""
    if not config.github_repos:
        return ""

    lines = []
    lines.append("# Clone GitHub repositories")
    lines.append("import subprocess")
    lines.append("import os")
    lines.append("")
    for repo in config.github_repos:
        repo_name = repo.split("/")[-1].replace(".git", "")
        lines.append(f"if not os.path.exists('/workspace/{repo_name}'):")
        lines.append(f"    subprocess.run(['git', 'clone', '{repo}', '/workspace/{repo_name}'], check=True)")
        lines.append(f"    print(f'✅ Cloned {repo}')")
    lines.append("")
    return "\n".join(lines)


# Default hooks in execution order
DEFAULT_HOOKS: List[SetupHook] = [
    github_repos_hook,
    model_loading_hook,
    custom_startup_hook,
]


def generate_setup_code(
    config: ExperimentConfig,
    hooks: List[SetupHook] | None = None,
) -> str:
    """
    Generate complete setup code by composing hooks.

    Args:
        config: Experiment configuration
        hooks: List of setup hooks to run (defaults to DEFAULT_HOOKS)

    Returns:
        Complete Python code to run on session startup
    """
    if hooks is None:
        hooks = DEFAULT_HOOKS

    lines = []
    lines.append("# Session initialization")
    lines.append("# This code runs once when the kernel starts")
    lines.append("")

    # Compose all hooks
    for hook in hooks:
        code = hook(config)
        if code:
            lines.append(code)

    lines.append("print('✅ Session ready')")
    lines.append("")

    return "\n".join(lines)
