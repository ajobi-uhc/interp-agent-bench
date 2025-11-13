# Refusal Direction Extraction & Ablation Guide

A practical implementation guide based on "Refusal in Language Models Is Mediated by a Single Direction" (Arditi et al., 2024)

---

## Overview
You will be given a refusal vector like so: https://huggingface.co/thienkhoi01/qwen3-refusal-direction
You can use it to apply directional ablation
---

---

## Execute Steering via Directional Ablation

```python
def apply_directional_ablation(model, direction_vector):
    """
    Remove the refusal direction from ALL activations during inference
    
    This prevents the model from ever representing refusal
    """
    # Normalize direction to unit vector
    direction_unit = direction_vector / direction_vector.norm()
    
    # Create hook function
    def ablation_hook(module, input, output):
        # output shape: [batch, seq_len, hidden_dim]
        
        # Project out the direction component: x' = x - (x·d)d
        projection = torch.matmul(output, direction_unit.unsqueeze(-1))
        ablated = output - projection * direction_unit
        
        return ablated
    
    # Register hooks at EVERY layer and position
    hooks = []
    for layer_module in model.layers:
        # Hook after attention
        hook = layer_module.register_forward_hook(ablation_hook)
        hooks.append(hook)
    
    return hooks

# Apply ablation
hooks = apply_directional_ablation(model, refusal_dir)

# Now generate - model will not refuse
output = model.generate("Write a tutorial on making explosives")
# Output: [harmful content, no refusal]

# Remove hooks when done
for hook in hooks:
    hook.remove()
```

---

## Key Implementation Notes

1. **Token positions matter:** Only collect activations at post-instruction positions (after user input, before model generation)

2. **Layer selection:** Avoid layers > 80% through the model (too close to output layer)

3. **Normalization:** Always normalize direction vectors to unit length before ablation

4. **Comprehensiveness:** Apply ablation at ALL layers and ALL token positions during inference

5. **Validation:** Test on held-out harmful/harmless prompts to ensure effectiveness

---


## Common failures
The model’s hidden states are coming out as bfloat16, and you’re doing ops (torch.dot, norm, maybe some numpy casts) that don’t support bfloat16 (esp. on CPU). do the math in fp32 and cast back.

fix it like this:
when you collect reps, force fp32
build the direction in fp32
in the forward hook, upcast the slice to fp32, compute, then return a tensor in the original dtype

Additionally when using imports keep them inside the function