"""
Steer model activations by injecting vectors at specific layers.

Functions:
    create_steering_hook(model, layer_idx, vector, strength=1.0, start_pos=0)
        Create a context manager that adds a steering vector to activations.

Args:
    model: HuggingFace model
    layer_idx: Layer to inject into (0-indexed)
    vector: Steering vector (torch.Tensor)
    strength: Multiplier for the vector (default 1.0)
    start_pos: Token position to start steering from (default 0)

Returns:
    Context manager - use with 'with' statement

Example:
    from steering_hook import create_steering_hook

    # Steer generation with a concept vector
    with create_steering_hook(model, layer_idx=20, vector=happy_vec, strength=2.0):
        output = model.generate(input_ids, max_new_tokens=50)

    # Combine with extract_activations to steer toward a concept
    concept_act = extract_activation(model, tokenizer, "I am happy", layer_idx=20)
    baseline_act = extract_activation(model, tokenizer, "I am", layer_idx=20)
    steering_vec = concept_act - baseline_act

    with create_steering_hook(model, layer_idx=20, vector=steering_vec, strength=1.5):
        output = model.generate(...)
"""


def create_steering_hook(model, layer_idx, vector, strength=1.0, start_pos=0):
    import torch

    # Validate layer index
    if hasattr(model, 'config') and hasattr(model.config, 'num_hidden_layers'):
        num_layers = model.config.num_hidden_layers
        if not 0 <= layer_idx < num_layers:
            raise ValueError(
                f"layer_idx {layer_idx} out of bounds. Model has {num_layers} layers (0-{num_layers-1})"
            )

    class SteeringHook:
        def __init__(self):
            self.hook_handle = None
            self.device = None
            self.vec = vector.cpu()  # Start on CPU
            self.layer_path = None

        def hook_fn(self, module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output

            # Move vector to correct device on first call
            if self.device is None:
                self.device = hidden.device
                self.vec = self.vec.to(self.device)

            # Validate vector shape matches hidden dimension
            if self.vec.shape[0] != hidden.shape[-1]:
                raise ValueError(
                    f"Vector dimension {self.vec.shape[0]} doesn't match "
                    f"hidden dimension {hidden.shape[-1]}"
                )

            # Inject from start_pos onwards
            if hidden.shape[1] > start_pos:
                hidden[:, start_pos:, :] += strength * self.vec.unsqueeze(0).unsqueeze(0)

            return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

        def __enter__(self):
            # Find the layer - try common paths with detailed error messages
            layer = None
            attempted_paths = []

            if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                layer = model.model.layers[layer_idx]
                self.layer_path = f"model.model.layers[{layer_idx}]"
            elif hasattr(model, 'model') and hasattr(model.model, 'language_model'):
                if hasattr(model.model.language_model, 'layers'):
                    layer = model.model.language_model.layers[layer_idx]
                    self.layer_path = f"model.model.language_model.layers[{layer_idx}]"
                    attempted_paths.append("model.model.language_model.layers")
            elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
                layer = model.transformer.h[layer_idx]
                self.layer_path = f"model.transformer.h[{layer_idx}]"
            else:
                attempted_paths.extend([
                    "model.model.layers",
                    "model.model.language_model.layers",
                    "model.transformer.h"
                ])

            if layer is None:
                error_msg = f"Cannot find layers in model. Attempted paths: {attempted_paths}"
                if hasattr(model, '__dict__'):
                    error_msg += f"\nAvailable attributes: {list(model.__dict__.keys())}"
                raise ValueError(error_msg)

            self.hook_handle = layer.register_forward_hook(self.hook_fn)
            return self

        def __exit__(self, *args):
            if self.hook_handle:
                self.hook_handle.remove()
                self.hook_handle = None

    return SteeringHook()
