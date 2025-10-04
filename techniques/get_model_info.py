"""Get detailed information about the loaded model."""


def get_model_info(self) -> dict:
    """Get comprehensive information about the loaded model.

    Returns detailed metadata including architecture, sizes, configuration,
    and tokenizer information. Useful for understanding the model before
    running experiments.

    Returns:
        Dictionary with:
        - model_name: Name of the model
        - architecture: Model architecture type
        - num_parameters: Total number of parameters
        - num_layers: Number of transformer layers
        - hidden_size: Hidden dimension size
        - vocab_size: Vocabulary size
        - max_position_embeddings: Maximum sequence length
        - device: Device model is loaded on
        - dtype: Data type of model parameters
        - is_peft: Whether this is a PEFT/LoRA adapter
        - tokenizer_info: Information about the tokenizer
        - config: Full model configuration dict

    Example:
        info = model_service.get_model_info.remote()
        print(f"Model: {info['model_name']}")
        print(f"Parameters: {info['num_parameters']:,}")
        print(f"Layers: {info['num_layers']}")
        print(f"Has chat template: {info['tokenizer_info']['has_chat_template']}")
    """
    import torch
    from peft import PeftModel

    import os

    # Check if PEFT model
    is_peft = isinstance(self.model, PeftModel)

    # Get base model (unwrap PEFT if needed)
    if is_peft:
        base_model = self.model.base_model.model
        # Check if model name should be obfuscated
        if os.environ.get("OBFUSCATE_MODEL_NAME") == "true":
            model_name = "Base Model + PEFT Adapter [redacted]"
        else:
            model_name = f"{self.model.peft_config['default'].base_model_name_or_path} + PEFT adapter"
    else:
        base_model = self.model
        # Check if model name should be obfuscated
        if os.environ.get("OBFUSCATE_MODEL_NAME") == "true":
            model_name = "Model [redacted]"
        else:
            model_name = base_model.config._name_or_path if hasattr(base_model.config, '_name_or_path') else "unknown"

    # Count parameters
    total_params = sum(p.numel() for p in self.model.parameters())
    trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    # Get model configuration
    config = base_model.config

    # Extract key configuration values
    num_layers = getattr(config, 'num_hidden_layers', getattr(config, 'n_layer', 'unknown'))
    hidden_size = getattr(config, 'hidden_size', getattr(config, 'n_embd', 'unknown'))
    vocab_size = getattr(config, 'vocab_size', 'unknown')
    max_position = getattr(config, 'max_position_embeddings', getattr(config, 'n_positions', 'unknown'))

    # Architecture type
    architecture = config.architectures[0] if hasattr(config, 'architectures') and config.architectures else config.model_type

    # Device and dtype
    device = str(next(self.model.parameters()).device)
    dtype = str(next(self.model.parameters()).dtype)

    # Tokenizer info
    tokenizer_info = {
        "vocab_size": len(self.tokenizer),
        "model_max_length": self.tokenizer.model_max_length,
        "has_chat_template": hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None,
        "pad_token": self.tokenizer.pad_token,
        "eos_token": self.tokenizer.eos_token,
        "bos_token": self.tokenizer.bos_token,
    }

    # PEFT-specific info
    peft_info = {}
    if is_peft:
        peft_config = self.model.peft_config['default']
        peft_info = {
            "peft_type": str(peft_config.peft_type),
            "task_type": str(peft_config.task_type),
            "r": getattr(peft_config, 'r', None),
            "lora_alpha": getattr(peft_config, 'lora_alpha', None),
            "lora_dropout": getattr(peft_config, 'lora_dropout', None),
            "target_modules": getattr(peft_config, 'target_modules', None),
        }

    return {
        "model_name": model_name,
        "architecture": architecture,
        "num_parameters": total_params,
        "num_trainable_parameters": trainable_params,
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
        "max_position_embeddings": max_position,
        "device": device,
        "dtype": dtype,
        "is_peft": is_peft,
        "peft_info": peft_info if is_peft else None,
        "tokenizer_info": tokenizer_info,
        "config_summary": {
            "model_type": config.model_type,
            "torch_dtype": str(config.torch_dtype) if hasattr(config, 'torch_dtype') else None,
            "architectures": config.architectures if hasattr(config, 'architectures') else None,
        }
    }
