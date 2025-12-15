"""
Extract activations from language models at specific layers and positions.

Usage:
    from extract_activations import extract_activation

    # Get last token activation at layer 15
    act = extract_activation(model, tokenizer, "Hello world", layer_idx=15)

    # Negative indexing: layer -1 = last layer
    act = extract_activation(model, tokenizer, "Hello", layer_idx=-1)

    # Chat format
    messages = [{"role": "user", "content": "Hi"}]
    act = extract_activation(model, tokenizer, messages, layer_idx=15)
"""


def extract_activation(model, tokenizer, text, layer_idx, position=-1):
    """
    Extract activation vector from a specific layer and token position.

    Args:
        model: HuggingFace model (works with PEFT-wrapped models too)
        tokenizer: HuggingFace tokenizer
        text: Input string, or list of chat messages for chat models
        layer_idx: Layer index (negative counts from end: -1 = last layer)
        position: Token position (negative counts from end: -1 = last token)

    Returns:
        torch.Tensor on CPU with shape (hidden_dim,)
    
    Example:
        act = extract_activation(model, tokenizer, "The cat sat", layer_idx=15, position=-1)
        print(act.shape)  # torch.Size([4096]) or whatever hidden_dim is
    """
    import torch

    # Tokenize
    if isinstance(text, str):
        inputs = tokenizer(text, return_tensors="pt")
    elif isinstance(text, list):
        formatted = tokenizer.apply_chat_template(
            text, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(formatted, return_tensors="pt")
    else:
        raise TypeError(f"text must be str or list of messages, got {type(text)}")

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)

    # hidden_states is tuple: (embeddings, layer0, layer1, ..., layerN)
    # so len - 1 = num layers
    hidden_states = outputs.hidden_states
    num_layers = len(hidden_states) - 1
    
    # Resolve layer index
    if layer_idx < 0:
        layer_idx = num_layers + layer_idx
    if not 0 <= layer_idx < num_layers:
        raise IndexError(f"layer_idx out of range [0, {num_layers})")

    # hidden_states[0] is embeddings, layer N is at index N+1
    hidden = hidden_states[layer_idx + 1]
    seq_len = hidden.shape[1]

    # Resolve position
    if position < 0:
        position = seq_len + position
    if not 0 <= position < seq_len:
        raise IndexError(f"position out of range [0, {seq_len})")

    return hidden[0, position, :].cpu()