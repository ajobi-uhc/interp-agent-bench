---
name: sae-latents
description: Extract and analyze SAE (Sparse Autoencoder) latent activations to understand which interpretable features are active for given inputs
preload: true
---

# SAE Latents

Extract and analyze sparse autoencoder latent activations to identify which interpretable features are active.

## When to Use

- Understanding which interpretable features activate for specific inputs
- Finding inputs that maximally activate specific latents
- Analyzing feature composition and interactions

## Available Functions

### get_sae_latents

Extract SAE latent activations for a given input:

```python
def get_sae_latents(text: str, sae_layer: int, top_k: int = 10):
    """
    Extract SAE latent activations for input text.

    Args:
        text: Input text to analyze
        sae_layer: SAE layer to extract from (must match loaded SAE)
        top_k: Number of top latents to return

    Returns:
        dict with text, latents (list of {index, activation})
    """
    import torch

    # Get model activations
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[sae_layer]

        # Encode with SAE
        latents = sae.encode(hidden_states)

    # Get top-k active latents
    top_k_values, top_k_indices = latents[0, -1].topk(top_k)

    return {
        "text": text,
        "latents": [
            {"index": int(idx), "activation": float(val)}
            for idx, val in zip(top_k_indices, top_k_values)
        ]
    }
```

### ablate_sae_latent

Ablate (zero out) a specific SAE latent and measure effect:

```python
def ablate_sae_latent(text: str, latent_index: int, sae_layer: int):
    """
    Ablate a specific SAE latent and generate text.

    Args:
        text: Input prompt
        latent_index: Index of latent to ablate
        sae_layer: SAE layer

    Returns:
        dict with original_text, ablated_text, latent_index
    """
    import torch

    # Original generation
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=50)
    original_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Create ablation hook
    def ablation_hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        # Encode with SAE
        latents = sae.encode(hidden)
        # Ablate specific latent
        latents[:, :, latent_index] = 0
        # Decode back to hidden states
        reconstructed = sae.decode(latents)
        return (reconstructed,) if isinstance(output, tuple) else reconstructed

    # Register hook and generate with ablation
    handle = model.model.layers[sae_layer].register_forward_hook(ablation_hook)
    try:
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=50)
        ablated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    finally:
        handle.remove()

    return {
        "original_text": original_text,
        "ablated_text": ablated_text,
        "latent_index": latent_index
    }
```

## Example Usage

```python
# Analyze what latents activate for a specific input
result = get_sae_latents(
    text="The Eiffel Tower is in Paris",
    sae_layer=8,
    top_k=10
)

print(f"Top active latents:")
for latent in result["latents"]:
    print(f"  Latent {latent['index']}: {latent['activation']:.3f}")

# Test causal effect of a specific latent
result = ablate_sae_latent(
    text="The capital of France is",
    latent_index=1234,  # Example: latent for "Paris" or "capitals"
    sae_layer=8
)

print(f"Original: {result['original_text']}")
print(f"Ablated:  {result['ablated_text']}")
```

## Tips

- **Layer selection**: Use the same layer the SAE was trained on
- **Top-k selection**: Start with 10-20 latents to see main contributors
- **Ablation testing**: Ablate latents one at a time to isolate causal effects
- **Feature interpretation**: Look at multiple examples to understand what each latent represents
