def extract_activation(model, tokenizer, text, layer_idx, position=-1):
    """
    Extract activation from a specific layer and position.

    Args:
        model: Language model
        tokenizer: Tokenizer
        text: Input text (can be a string or list of messages for chat template)
        layer_idx: Layer to extract from (0-indexed)
        position: Token position (-1 for last token)

    Returns:
        torch.Tensor: Activation vector (on CPU)
    """
    import torch

    # Handle both string and chat format
    if isinstance(text, str):
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
    else:
        # Assume it's a list of messages for chat template
        formatted = tokenizer.apply_chat_template(text, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)

    # Extract: hidden_states[0] is embedding, so layer N is at index N+1
    activation = outputs.hidden_states[layer_idx + 1][0, position, :]

    return activation.cpu()
