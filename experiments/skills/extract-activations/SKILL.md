---
name: extract-activations
description: Extract hidden state activations from specific model layers
preload: true
---

# Extract Activations

Get model activations at specific layers and token positions.

## Usage

Use the `extract_activation()` function:

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
