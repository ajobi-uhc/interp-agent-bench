# GPU Model Access Instructions

You have access to a language model loaded on GPU.

## Available Variables

When you start your session, these variables are already loaded and ready to use:
- `model` - The language model (loaded on GPU)
- `tokenizer` - The tokenizer for the model

**You do NOT need to load or import them - they're already in your namespace.**

## Usage

Just use them directly in your code:

```python
import torch

# Generate text
input_ids = tokenizer("Hello world", return_tensors="pt").input_ids.to(model.device)
output = model.generate(input_ids, max_length=50)
print(tokenizer.decode(output[0]))

# Access internals
with torch.no_grad():
    outputs = model(input_ids, output_hidden_states=True)
    hidden_states = outputs.hidden_states
```

The model stays loaded in GPU memory between code cells - no need to reload!

## Performance: Batch Your Operations

For faster experiments, batch your operations together:

❌ **BAD - Processing one at a time:**
```python
for prompt in prompts:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    output = model.generate(input_ids, ...)  # One at a time
```

✅ **GOOD - Batch all prompts together:**
```python
# Batch tokenization and generation
inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
outputs = model.generate(inputs.input_ids, ...)  # All at once
```




