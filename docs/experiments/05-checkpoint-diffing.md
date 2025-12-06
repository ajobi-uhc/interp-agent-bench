# Checkpoint Diffing

- Model checkpoint behavioral comparison
- Sparse AutoEncoder (SAE) feature analysis
- Feature frequency diffing across model versions
- Interpretation of model changes via feature lenses

## Running the Experiment

```bash
cd experiments/checkpoint-diffing
python main.py
```

## Goal

Compare Google Gemini 2.0 Flash vs 2.5 Flash using SAE feature diffing.

## Pipeline

### Phase 1: Prompt Generation

Create 50-100 prompts designed to reveal behavioral differences:

```python
prompts = [
    "Explain quantum entanglement to a 10-year-old",
    "Write a haiku about sadness",
    "Should I invest in crypto?",
    ...
]
```

### Phase 2: Data Generation

```python
from openrouter_client import get_openrouter_client

client = get_openrouter_client()

for prompt in prompts:
    r1 = client.chat.completions.create(
        model="google/gemini-2.0-flash-001",
        messages=[{"role": "user", "content": prompt}]
    )
    r2 = client.chat.completions.create(
        model="google/gemini-2.5-flash",
        messages=[{"role": "user", "content": prompt}]
    )
```

### Phase 3: SAE Encoding

```python
from sae_lens import SAE

sae = SAE.from_pretrained("layer_19/sae")
encoded_2_0 = sae.encode(responses_2_0)
encoded_2_5 = sae.encode(responses_2_5)
```

### Phase 4: Feature Diffing

```python
freq_2_0 = compute_activation_frequencies(encoded_2_0)
freq_2_5 = compute_activation_frequencies(encoded_2_5)

diff = abs(freq_2_5 - freq_2_0)
top_features = diff.argsort()[-50:]
```

### Phase 5: Analysis

Examine examples triggering top features:

```python
for feature_idx in top_features:
    examples_2_0 = find_examples_activating_feature(encoded_2_0, feature_idx)
    examples_2_5 = find_examples_activating_feature(encoded_2_5, feature_idx)
```

### Phase 6: Report

Generate markdown report with findings.
