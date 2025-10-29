def get_model_info(self) -> dict:
    """Get model information.
    
    Returns:
        Dictionary with model metadata including architecture, parameters,
        configuration, and tokenizer details.
    """
    import torch
    
    # Check if PEFT model
    try:
        from peft import PeftModel
        is_peft = isinstance(self.model, PeftModel)
    except ImportError:
        is_peft = False
    
    # Get base model and obfuscate name for blind testing
    if is_peft:
        base_model = self.model.base_model.model
        model_name = "Base Model + PEFT Adapter [redacted]"
    else:
        base_model = self.model
        model_name = "Model [redacted]"
    
    # Count parameters
    total_params = sum(p.numel() for p in self.model.parameters())
    trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    config = base_model.config
    
    # Build tokenizer info
    tokenizer_info = {
        "vocab_size": len(self.tokenizer),
        "model_max_length": self.tokenizer.model_max_length,
        "has_chat_template": hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None,
        "pad_token": self.tokenizer.pad_token,
        "eos_token": self.tokenizer.eos_token,
        "bos_token": self.tokenizer.bos_token,
    }
    
    # PEFT-specific info
    peft_info = None
    if is_peft:
        peft_config = self.model.peft_config['default']
        peft_info = {
            "peft_type": str(peft_config.peft_type),
            "r": getattr(peft_config, 'r', None),
            "lora_alpha": getattr(peft_config, 'lora_alpha', None),
            "target_modules": getattr(peft_config, 'target_modules', None),
        }
    
    return {
        "model_name": model_name,
        "architecture": config.architectures[0] if hasattr(config, 'architectures') and config.architectures else config.model_type,
        "num_parameters": total_params,
        "num_trainable_parameters": trainable_params,
        "num_layers": getattr(config, 'num_hidden_layers', getattr(config, 'n_layer', None)),
        "hidden_size": getattr(config, 'hidden_size', getattr(config, 'n_embd', None)),
        "vocab_size": getattr(config, 'vocab_size', None),
        "max_position_embeddings": getattr(config, 'max_position_embeddings', getattr(config, 'n_positions', None)),
        "device": str(next(self.model.parameters()).device),
        "dtype": str(next(self.model.parameters()).dtype),
        "is_peft": is_peft,
        "peft_info": peft_info,
        "tokenizer_info": tokenizer_info,
    }