def create_steering_hook(model, layer_idx, vector, strength=1.0, start_pos=0):
    """
    Create a context manager that steers model activations.

    Args:
        model: The language model
        layer_idx: Which layer to inject into (0-indexed)
        vector: Steering vector (torch.Tensor, any device)
        strength: Multiplier for injection strength
        start_pos: Token position to start steering from

    Returns:
        Context manager - use with 'with' statement

    Example:
        with create_steering_hook(model, layer_idx=20, vector=concept_vec, strength=2.0):
            outputs = model.generate(...)
    """
    import torch

    class SteeringHook:
        def __init__(self):
            self.hook_handle = None
            self.device = None
            self.vec = vector.cpu()  # Start on CPU

        def hook_fn(self, module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output

            # Move vector to correct device on first call
            if self.device is None:
                self.device = hidden.device
                self.vec = self.vec.to(self.device)

            # Inject from start_pos onwards
            if hidden.shape[1] > start_pos:
                hidden[:, start_pos:, :] += strength * self.vec.unsqueeze(0).unsqueeze(0)

            return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

        def __enter__(self):
            # Find the layer - try common paths
            if hasattr(model.model, 'layers'):
                layer = model.model.layers[layer_idx]
            elif hasattr(model.model, 'language_model'):
                layer = model.model.language_model.layers[layer_idx]
            else:
                raise ValueError("Can't find model layers")

            self.hook_handle = layer.register_forward_hook(self.hook_fn)
            return self

        def __exit__(self, *args):
            if self.hook_handle:
                self.hook_handle.remove()
                self.hook_handle = None

    return SteeringHook()
