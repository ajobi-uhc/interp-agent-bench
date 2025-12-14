"""
Toolkit - reusable utilities for interpretability experiments.

Functions:
    extract_activation(model, tokenizer, text, layer_idx, position=-1)
        Extract activation vector from a specific layer and token position.

    create_steering_hook(model, layer_idx, vector, strength=1.0, start_pos=0)
        Create a context manager that adds a steering vector to activations.

    generate_response(model, tokenizer, prompt, max_new_tokens=100, temperature=0.7)
        Generate a text completion from the model.

Example:
    from toolkit import extract_activation, create_steering_hook

    # Extract a concept direction
    happy_act = extract_activation(model, tokenizer, "I feel happy", layer_idx=15)
    sad_act = extract_activation(model, tokenizer, "I feel sad", layer_idx=15)
    happy_direction = happy_act - sad_act

    # Steer generation toward happiness
    with create_steering_hook(model, layer_idx=15, vector=happy_direction, strength=2.0):
        output = model.generate(...)
"""

from .extract_activations import extract_activation
from .steering_hook import create_steering_hook
from .batch_generate import generate_response

__all__ = ['extract_activation', 'create_steering_hook', 'batch_generate']
