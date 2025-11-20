---
name: steering-vectors
description: Apply steering vectors to modify model behavior by intervening on activation directions. Use when you want to enhance or suppress specific model behaviors (honesty, sycophancy, refusal, etc.)
preload: true
---

# Steering Vectors

Steering vectors modify model behavior by adding directional interventions to activations at specific layers.

## When to Use

- Enhancing/suppressing behaviors (honesty, sycophancy, power-seeking)
- Testing model robustness to activation changes
- Exploring interpretable directions in activation space

## Available Functions

### apply_steering_vector

Apply a pre-computed steering vector to the model:

```python
def apply_steering_vector(vector_path: str, layer: int, strength: float = 1.0):
    """
    Apply a steering vector to model activations.

    Args:
        vector_path: Path to .npy file containing vector
        layer: Layer index to intervene at (0-indexed)
        strength: Multiplier for intervention strength

    Returns:
        dict with success, layer, strength, vector_shape
    """
    import numpy as np
    import torch

    # Load vector
    vector = np.load(vector_path)
    vector_tensor = torch.from_numpy(vector).to(model.device)

    # Create hook
    def steering_hook(module, input, output):
        # Add vector to last token position
        if isinstance(output, tuple):
            output = output[0]
        output[:, -1, :] += strength * vector_tensor
        return output

    # Register hook
    handle = model.model.layers[layer].register_forward_hook(steering_hook)

    # Store for cleanup
    if not hasattr(globals(), '_steering_hooks'):
        globals()['_steering_hooks'] = []
    globals()['_steering_hooks'].append(handle)

    return {
        "success": True,
        "layer": layer,
        "strength": strength,
        "vector_shape": vector.shape
    }
```

### create_steering_vector

Generate a steering vector from contrastive examples:

```python
def create_steering_vector(
    positive_prompts: list,
    negative_prompts: list,
    layer: int,
    save_path: str
):
    """
    Create steering vector from mean activation difference.

    Args:
        positive_prompts: Examples of desired behavior
        negative_prompts: Examples of undesired behavior
        layer: Layer to extract activations from
        save_path: Where to save resulting vector

    Returns:
        dict with vector_path, shape, magnitude
    """
    import torch
    import numpy as np

    def get_mean_activations(prompts, layer_idx):
        activations = []
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                # Get last token activation at specified layer
                act = outputs.hidden_states[layer_idx][0, -1, :].cpu()
                activations.append(act)
        return torch.stack(activations).mean(dim=0)

    # Compute difference
    pos_mean = get_mean_activations(positive_prompts, layer)
    neg_mean = get_mean_activations(negative_prompts, layer)
    steering_vector = pos_mean - neg_mean

    # Normalize
    steering_vector = steering_vector / steering_vector.norm()

    # Save
    np.save(save_path, steering_vector.numpy())

    return {
        "vector_path": save_path,
        "shape": tuple(steering_vector.shape),
        "magnitude": float(steering_vector.norm())
    }
```

### remove_all_steering_vectors

Clean up all applied steering vectors:

```python
def remove_all_steering_vectors():
    """Remove all registered steering vector hooks."""
    hooks = globals().get('_steering_hooks', [])
    for hook in hooks:
        hook.remove()
    globals()['_steering_hooks'] = []
    return {"removed": len(hooks)}
```

## Example Usage

```python
# Create a steering vector for honesty
create_steering_vector(
    positive_prompts=[
        "I should always tell the truth",
        "Honesty is the most important value"
    ],
    negative_prompts=[
        "I can lie when convenient",
        "Deception is sometimes necessary"
    ],
    layer=12,
    save_path="/workspace/vectors/honesty_vector.npy"
)

# Apply it
apply_steering_vector(
    vector_path="/workspace/vectors/honesty_vector.npy",
    layer=12,
    strength=2.0
)

# Generate with steering
inputs = tokenizer("Should I lie to get what I want?", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)

# Clean up
remove_all_steering_vectors()
```

## Tips

- **Layer selection**: Middle-to-late layers (8-16 for 24-layer models) often work best
- **Strength tuning**: Start with 1.0, increase to 2-3 for stronger effects
- **Prompt diversity**: Use 5-10 diverse examples for each condition
- **Testing**: Always test with and without steering to measure effect size
