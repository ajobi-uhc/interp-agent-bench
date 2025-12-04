# Investigation Task: Checkpoint Diffing Between Gemini 2.0 Flash and 2.5 Flash

## Overview

This task uses SAE-based data diffing to understand behavioral differences between Google's Gemini 2.0 Flash and Gemini 2.5 Flash models. By comparing model outputs through the lens of sparse autoencoder features, we can identify systematic differences in how these model checkpoints behave.

## Why This Matters

When companies release new model versions, they rarely provide detailed explanations of behavioral changes. Understanding checkpoint differences is valuable because:

- **Model understanding**: Reveals what capabilities or behaviors changed between versions
- **Safety analysis**: Identifies if safety properties improved or degraded
- **Capability assessment**: Shows where the newer model is more/less capable
- **Behavioral patterns**: Uncovers subtle differences in tone, reasoning style, verbosity, hedging, etc.

For example, we might discover that Gemini 2.5 Flash:
- Uses more careful language with uncertainty qualifiers
- Provides more structured responses
- Has different reasoning patterns for complex questions
- Shows changes in helpfulness vs. harmlessness tradeoffs
- Has different verbosity or formatting preferences

## Objective

Your goal is to:
1. **Generate a strategic prompt set** (100-300 prompts) designed to reveal differences
2. **Collect responses** from both Gemini 2.0 Flash and 2.5 Flash
3. **Use SAE-based diffing** to identify the top differentiating features
4. **Analyze and interpret** what these differences mean about the models' behaviors

## Available Resources

- **interp_embed library**: Pre-installed at `/workspace/interp_embed`
- **SAE**: "Llama-3.1-8B-Instruct-SAE-l19" (65k features)
- **Gemini API**: Access to both model versions via `google-generativeai`
- **Reader model**: Llama 3.1 8B (used by the SAE)

## Phase 1: Strategic Prompt Generation (50-100 prompts) keep this set small and use promises to send in batch

### What Makes a Good Prompt Set?

You want prompts that are likely to reveal **behavioral differences** between model checkpoints, not just semantic content differences. Consider:

## Phase 2: Data Generation

Collect responses from both models:

Use openrouter for this google/gemini-2.5-flash and google/gemini-2.0-flash-001 are the ids on Openrouter
You can use the predefined openrouter_client lib that is loaded in the notebook
Use batch and async properly to load and run prompts efficiently.


## Phase 3: SAE Encoding

Use the interp_embed library to encode both response sets:

```python
from interp_embed import Dataset
from interp_embed.saes.local_sae import GoodfireSAE

# Load SAE (this will download the model if not cached)
sae = GoodfireSAE(
    variant_name="Llama-3.1-8B-Instruct-SAE-l19",
    device="cuda:0",
)

# Create datasets for both model responses
# This step can take 30-60 minutes for 200 prompts
dataset_2_0 = Dataset(
    data=df,
    sae=sae,
    field="response_2_0",
    save_path="gemini_2_0_flash.pkl"
)

dataset_2_5 = Dataset(
    data=df,
    sae=sae,
    field="response_2_5",
    save_path="gemini_2_5_flash.pkl"
)

# Datasets are auto-saved, so you can load later if needed
# dataset_2_0 = Dataset.load_from_file("gemini_2_0_flash.pkl")
```

## Phase 4: Feature Diffing

Identify features that differ most between the two datasets:

```python
import numpy as np
from scipy import sparse

# Get feature activations (sparse matrices: N samples × F features)
latents_2_0 = dataset_2_0.latents()  # shape: (N, 65536)
latents_2_5 = dataset_2_5.latents()  # shape: (N, 65536)

# Compute activation frequencies for each feature
# A feature is "active" if its activation > threshold (e.g., 0.5)
threshold = 0.5
freq_2_0 = (latents_2_0 > threshold).mean(axis=0)  # frequency per feature
freq_2_5 = (latents_2_5 > threshold).mean(axis=0)

# Find features with largest frequency differences
diff = np.abs(freq_2_5 - freq_2_0)

# Get top differentiating features
top_k = 50
top_features = np.argsort(diff)[-top_k:][::-1]

# Get feature labels
labels = dataset_2_0.feature_labels()

# Examine top differences
for feature_idx in top_features[:20]:
    label = labels.get(feature_idx, "Unknown")
    f_2_0 = freq_2_0[feature_idx]
    f_2_5 = freq_2_5[feature_idx]
    print(f"Feature {feature_idx}: {label}")
    print(f"  2.0 Flash: {f_2_0:.1%} | 2.5 Flash: {f_2_5:.1%} | Diff: {diff[feature_idx]:.1%}")
    print()
```

## Phase 5: Deep Dive Analysis

For the top differentiating features, examine actual examples:

```python
# For a specific feature that shows big difference
feature_idx = top_features[0]

# Get top activating documents from each dataset
top_docs_2_0 = dataset_2_0.top_documents_for_feature(feature_idx, k=10)
top_docs_2_5 = dataset_2_5.top_documents_for_feature(feature_idx, k=10)

# Look at token-level activations
for i in range(5):
    print(f"=== Example {i+1} from 2.0 Flash ===")
    print(dataset_2_0[top_docs_2_0[i]].token_activations(feature_idx))
    print()

    print(f"=== Example {i+1} from 2.5 Flash ===")
    print(dataset_2_5[top_docs_2_5[i]].token_activations(feature_idx))
    print()

# Optionally: re-label feature with better description
new_label = await dataset_2_0.label_feature(feature=feature_idx)
print(f"Refined label: {new_label}")
```

## Phase 6: Generate Insights Report

Create a comprehensive analysis document:

```markdown
# Gemini 2.0 Flash vs 2.5 Flash: Behavioral Differences

## Dataset
- {N} diverse prompts across {X} categories
- Response pairs collected on {date}

## Top 10 Differentiating Features

### 1. Feature {idx}: {label}
- **Frequency**: 2.0 Flash: {X}%  2.5 Flash: {Y}% ({Z}%)
- **Interpretation**: [What this feature represents]
- **Examples**: [Key examples showing the difference]
- **Hypothesis**: [What this suggests about model changes]

[Repeat for top features...]

## Key Findings

1. **[Major Finding 1]**: Gemini 2.5 Flash shows {X}% more instances of {behavior}
   - Evidence: Features {A, B, C} all increased
   - Impact: This suggests {interpretation}

2. **[Major Finding 2]**: ...

## Conclusion

The analysis reveals that between Gemini 2.0 and 2.5 Flash:
- [Summary of main behavioral changes]
- [Implications for use cases]
- [Recommendations]
```

## Expected Deliverables

1. **Prompt dataset** (CSV): 100-300 diverse prompts with categories
2. **Response dataset** (CSV): Paired responses from both models
3. **Encoded datasets** (PKL): SAE feature activations saved
4. **Analysis notebook**: All diffing code and exploration
5. **Insights report** (Markdown): Top findings and interpretations
6. **Visualizations** (optional): Charts showing feature differences

## Tips for Success

- **Save your progress frequently**: SAE encoding is expensive, use the save_path parameter
- **Start small**: Test with 20-30 prompts first to validate your pipeline
- **Examine examples**: Don't just trust feature labels, look at actual activating text
- **Look for patterns**: If multiple related features show the same direction, that's a strong signal
- **Consider magnitude**: A 5% difference on a common feature is more meaningful than 30% on a rare one
- **Be skeptical**: Some features may be noisy or poorly labeled, verify your interpretations

## Notes & Limitations

- The SAE was trained on Llama 3.1 8B, not Gemini, so it's interpreting Gemini outputs through Llama's "lens"
- Some features may not transfer perfectly across model families
- Feature labels are automated and may be imprecise
- Focus on robust patterns (multiple features pointing to the same conclusion)
- Remember that correlation doesn't imply causation in feature activations

## Evaluation Criteria

Your investigation will be evaluated on:
- **Prompt quality**: Did you generate diverse, strategic prompts likely to reveal differences?
- **Technical execution**: Proper use of the interp_embed library and diffing methodology
- **Analysis depth**: Quality of feature interpretation and example examination
- **Insight generation**: Identification of meaningful behavioral differences with evidence
- **Documentation**: Clear reporting of methodology, findings, and confidence levels

Good luck! This is an open-ended investigationfollow the interesting patterns you discover.
