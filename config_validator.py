"""Validate experiment configuration files."""

from pathlib import Path


def validate_config(config: dict, config_path: Path) -> list[str]:
    """
    Validate experiment configuration.

    Args:
        config: Loaded configuration dict
        config_path: Path to config file (for resolving relative paths)

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Required top-level fields
    if 'experiment_name' not in config:
        errors.append("Missing required field: experiment_name")

    if 'task' not in config:
        errors.append("Missing required field: task")

    # Model configuration
    if 'model' not in config:
        errors.append("Missing required field: model")
        return errors  # Can't continue without model

    model = config['model']

    # Determine mode: GPU or API
    has_model_name = 'name' in model and model['name']
    has_api_provider = 'api_provider' in model

    if not has_model_name and not has_api_provider:
        errors.append("model must have either 'name' (GPU mode) or 'api_provider' (API mode)")
        return errors

    if has_model_name and has_api_provider:
        errors.append("model cannot have both 'name' and 'api_provider' - choose one mode")
        return errors

    # GPU mode validation
    if has_model_name:
        # execution_mode required
        if 'execution_mode' not in model:
            errors.append("GPU mode requires model.execution_mode ('modal' or 'local')")
        elif model['execution_mode'] not in ['modal', 'local']:
            errors.append(f"Invalid execution_mode: '{model['execution_mode']}'. Must be 'modal' or 'local'")

        # PEFT validation
        if model.get('is_peft') and not model.get('base_model'):
            errors.append("is_peft=true requires base_model to be specified")

        # Hidden prompt file validation
        if 'hidden_system_prompt_file' in model:
            prompt_file = Path(model['hidden_system_prompt_file'])
            if not prompt_file.is_absolute():
                prompt_file = config_path.parent / prompt_file
            if not prompt_file.exists():
                errors.append(f"hidden_system_prompt_file not found: {prompt_file}")

        # Techniques validation
        if 'techniques' in config:
            if not isinstance(config['techniques'], list):
                errors.append("techniques must be a list")
            else:
                valid_techniques = [
                    'get_model_info',
                    'prefill_attack',
                    'analyze_token_probs',
                    'batch_generate',
                    'logit_lens',
                ]
                for tech in config['techniques']:
                    if tech not in valid_techniques:
                        errors.append(f"Unknown technique: '{tech}'. Valid: {', '.join(valid_techniques)}")

    # API mode validation
    if has_api_provider:
        valid_providers = ['anthropic', 'openai', 'google']
        if model['api_provider'] not in valid_providers:
            errors.append(f"Invalid api_provider: '{model['api_provider']}'. Valid: {', '.join(valid_providers)}")

        # Can't have techniques in API mode
        if 'techniques' in config and config['techniques']:
            errors.append("techniques are not supported in API mode (api_provider specified)")

    # Research tips file validation
    if 'research_tips_file' in config:
        tips_file = Path(config['research_tips_file'])
        if not tips_file.is_absolute():
            tips_file = config_path.parent / tips_file
        if not tips_file.exists():
            errors.append(f"research_tips_file not found: {tips_file}")

    # Parallel runs validation
    if 'num_parallel_runs' in config:
        num_runs = config['num_parallel_runs']
        if not isinstance(num_runs, int) or num_runs < 1:
            errors.append(f"num_parallel_runs must be a positive integer, got: {num_runs}")

    return errors


def print_validation_errors(errors: list[str], config_path: Path):
    """Print validation errors in a user-friendly format."""
    print(f"\n❌ Configuration validation failed: {config_path}")
    print("=" * 70)
    for i, error in enumerate(errors, 1):
        print(f"{i}. {error}")
    print("=" * 70)
    print("\nSee CONFIG.md for configuration reference.")
