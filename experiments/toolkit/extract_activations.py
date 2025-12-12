"""
Extract activations from language models at specific layers and positions.

Functions:
    extract_activation(model, tokenizer, text, layer_idx, position=-1)
        Extract activation vector from a specific layer and token position.

Args:
    model: HuggingFace model (already loaded)
    tokenizer: HuggingFace tokenizer
    text: Input string or list of chat messages
    layer_idx: Layer index (0 to num_layers-1)
    position: Token position (-1 for last token)

Returns:
    torch.Tensor on CPU with shape (hidden_dim,)

Example:
    from extract_activations import extract_activation

    # Get last token activation at layer 15
    act = extract_activation(model, tokenizer, "Hello world", layer_idx=15)

    # Get activation for chat format
    messages = [{"role": "user", "content": "Hi"}]
    act = extract_activation(model, tokenizer, messages, layer_idx=15)
"""


def extract_activation(model, tokenizer, text, layer_idx, position=-1):
    import torch

    # Validate layer index
    if hasattr(model, 'config') and hasattr(model.config, 'num_hidden_layers'):
        num_layers = model.config.num_hidden_layers
        if not 0 <= layer_idx < num_layers:
            raise ValueError(
                f"layer_idx {layer_idx} out of bounds. Model has {num_layers} layers (0-{num_layers-1})"
            )

    # Handle both string and chat format
    try:
        if isinstance(text, str):
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
        elif isinstance(text, list):
            # Assume it's a list of messages for chat template
            formatted = tokenizer.apply_chat_template(
                text, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
        else:
            raise TypeError(f"text must be str or list, got {type(text)}")
    except Exception as e:
        raise ValueError(f"Failed to tokenize input: {e}")

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)

    # Extract: hidden_states[0] is embedding, so layer N is at index N+1
    # Validate position
    seq_len = outputs.hidden_states[layer_idx + 1].shape[1]
    if position >= seq_len or position < -seq_len:
        raise ValueError(
            f"position {position} out of bounds for sequence length {seq_len}"
        )

    activation = outputs.hidden_states[layer_idx + 1][0, position, :]

    return activation.cpu()
