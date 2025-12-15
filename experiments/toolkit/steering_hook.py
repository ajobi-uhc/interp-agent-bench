"""
Steer model activations by injecting vectors at specific layers.

Usage:
    from steering import steering_hook
    
    # Pass the decoder layers directly:
    with steering_hook(model.model.layers, layer_idx=20, vector=vec, strength=1.5):
        output = model.generate(input_ids, max_new_tokens=50)

Common layer paths by architecture:
    - Llama, Mistral, Qwen: model.model.layers
    - Gemma: model.model.model.layers
    - GPT-2, GPT-Neo: model.transformer.h
    - PEFT-wrapped: model.base_model.model.layers (add .model for Gemma)
"""


def steering_hook(layers, layer_idx, vector, strength=1.0, start_pos=0):
    """
    Context manager that adds a steering vector to layer activations during forward pass.
    
    Args:
        layers: Decoder layers (nn.ModuleList). Get these from your model, e.g. model.model.layers
        layer_idx: Which layer to hook (0-indexed). Negative indices count from end (-1 = last)
        vector: Steering vector tensor, shape (hidden_dim,). Must match model's hidden size
        strength: Multiplier for the steering vector. Default 1.0
        start_pos: Only steer tokens at this position and later. Default 0 (all tokens)
    
    Example:
        vec = torch.load("happy_steering_vector.pt")  # shape (hidden_dim,)
        with steering_hook(model.model.layers, layer_idx=20, vector=vec, strength=2.0):
            output = model.generate(input_ids, max_new_tokens=100)
    """
    from contextlib import contextmanager
    
    @contextmanager
    def _hook():
        if layer_idx < 0:
            idx = len(layers) + layer_idx
        else:
            idx = layer_idx
        
        if not 0 <= idx < len(layers):
            raise IndexError(f"layer_idx {layer_idx} out of range [0, {len(layers)})")
        
        vec = vector.detach()
        
        def hook(_, __, output):
            h = output[0] if isinstance(output, tuple) else output
            
            if h.shape[1] > start_pos:
                h = h.clone()
                h[:, start_pos:] += strength * vec.to(h.device, h.dtype)
            
            return (h, *output[1:]) if isinstance(output, tuple) else h
        
        handle = layers[idx].register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()
    
    return _hook()