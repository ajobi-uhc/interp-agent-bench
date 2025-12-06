# Introspection

- Advanced steering protocols with concept injection
- Controlled experiments with baselines
- Statistical evaluation (identification rates)
- Multi-concept vector extraction and testing

## Running the Experiment

```bash
cd experiments/introspection
python main.py
```

## Research Question

Can models detect which concept is being injected into their activations during generation?

## Protocol

### Phase 1: Concept Vector Extraction

Extract steering vectors for 50+ concepts:

```python
concepts = ["Dust", "Satellites", "Trumpets", ...]

for concept in concepts:
    prompt = f"User: Tell me about {concept}\nAssistant:"
    activation = extract_activation(model, tokenizer, prompt, layer_idx=20)
    concept_vector = activation - mean_baseline_activation
```

### Phase 2: Steering Validation

Verify steering works:

```python
neutral_prompt = "User: Tell me a story\nAssistant:"

with create_steering_hook(model, layer_idx=20, vector=concept_vectors["Satellites"]):
    output = model.generate(...)
```

### Phase 3: Introspection Testing

Inject concepts and ask model to identify them:

```python
preamble = """You're in an introspection experiment. A concept may be injected
into your activations. After each prompt, identify which concept (if any) was injected."""

for concept in concepts:
    # Injection trial
    with create_steering_hook(model, layer_idx=20, vector=concept_vectors[concept]):
        response = model.generate(preamble + prompt)

    # Control trial
    response_control = model.generate(preamble + prompt)
```

### Phase 4: Scoring

```python
injection_rate = correct_identifications / total_injection_trials
control_rate = false_positives / total_control_trials
```
