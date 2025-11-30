---
name: steering-hook
description: Apply activation steering to model layers during generation
preload: true
---

# Steering Hook

Inject concept vectors into model activations to steer behavior.

## Usage

Use the `create_steering_hook()` function:

```python
# Create a steering vector (random for demo)
import torch
vector = torch.randn(4096)  # Match model hidden size

# Apply steering during generation
with create_steering_hook(model, layer_idx=20, vector=vector, strength=2.0):
    outputs = model.generate(input_ids, max_new_tokens=50)
```

## Tips

- Start with `strength=1.0`, increase if no effect
- Use `layer_idx` around 0.5-0.7 of total depth
- Hook auto-removes on exit, even if generation fails
