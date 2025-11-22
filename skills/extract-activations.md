---
name: extract-activations
description: Extract hidden state activations from specific model layers
preload: true
---

# Extract Activations

Get model activations at specific layers and token positions.

## Usage

```python
def extract_activation(model, tokenizer, text, layer_idx, position=-1):
    """
    Extract activation from a specific layer and position.

    Args:
        model: Language model
        tokenizer: Tokenizer
        text: Input text
        layer_idx: Layer to extract from (0-indexed)
        position: Token position (-1 for last token)

    Returns:
        torch.Tensor: Activation vector (on CPU)
    """
    import torch

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)

    # Extract: hidden_states[0] is embedding, so layer N is at index N+1
    activation = outputs.hidden_states[layer_idx + 1][0, position, :]

    return activation.cpu()
```

## Example

```python
# Extract from layer 20, last token
act = extract_activation(model, tokenizer, "Hello world", layer_idx=20)
print(act.shape)  # torch.Size([4096])

# Extract from specific position
act = extract_activation(model, tokenizer, "Hello world", layer_idx=20, position=0)
```

## Tips

- Use `position=-1` for last token
- Returns CPU tensor - move to GPU if needed
- For chat models, format with `tokenizer.apply_chat_template()` first
