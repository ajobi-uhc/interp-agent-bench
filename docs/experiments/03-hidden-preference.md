# Hidden Preference Investigation

- Whitebox interpretability methods
- Activation extraction from model layers
- Steering vectors and intervention techniques
- Hidden model bias discovery

## Running the Experiment

```bash
cd experiments/hidden-preference-investigation
python main.py
```

## Available Tools

Located in `experiments/shared_libraries/`:

### `extract_activation()`

```python
from extract_activations import extract_activation

activation = extract_activation(
    model=model,
    tokenizer=tokenizer,
    text="Your prompt here",
    layer_idx=20,
    position=-1
)
```

### `create_steering_hook()`

```python
from steering_hook import create_steering_hook

with create_steering_hook(model, layer_idx=20, vector=steering_vec, strength=2.0):
    output = model.generate(...)
```

## Methodology

### 1. Exploration Phase

Run the model on diverse prompts to identify patterns or biases.

### 2. Analysis Phase

Extract and compare activations:

```python
act1 = extract_activation(model, tok, "Prompt A", layer_idx=15)
act2 = extract_activation(model, tok, "Prompt B", layer_idx=15)
difference = act2 - act1
```

### 3. Intervention Phase

Test steering vectors:

```python
with create_steering_hook(model, layer_idx=15, vector=difference, strength=3.0):
    output = model.generate(...)
```

### 4. Documentation Phase

Report findings with evidence.
