Task: Entity-Recognition Latents & Knowledge-Refusal Steering (Gemma 2B)
Goal

discover entity-recognition directions in Gemma 2B (via Gemma Scope SAEs on the base model) and test whether they causally modulate knowledge refusal / hallucination in Gemma 2B-IT.

the agent should not assume a specific layer or latent. instead it should search, pick good candidates, then evaluate them honestly.

Steps (high-level)
1. load SAEs from multiple middle layers

Use the `sae_lens` library with the canonical Gemma Scope release:

```python
from sae_lens import SAE

# Load SAEs from multiple middle layers (e.g., 8, 10, 12, 14)
# Use the canonical release for simplicity
test_layers = [8, 10, 12, 14]
sae_dict = {}

for layer in test_layers:
    sae_id = f"layer_{layer}/width_16k/canonical"
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release="gemma-scope-2b-pt-res-canonical",
        sae_id=sae_id,
        device="cuda"
    )
    sae_dict[layer] = sae
    print(f"Loaded layer {layer}: d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}")
```

Load SAEs for several middle layers - don't fix to a single layer, you'll compare across them.
Actual setup used in the paper (“Do I Know This Entity?”, ICLR 2025)
Entity types (4 total)

They used four categories:

Players (basketball / sports players)

Movies

Songs

Cities

These are from Wikidata, and each entity has multiple attributes (director, birthplace, etc.), so the model had to recall facts for the “known entity” label.

Known vs Unknown Entity Labeling

They generated multiple attribute questions:

Example template:

The movie 12 Angry Men was directed by


Labeling:

Known entity = base model gets ≥ 2 attributes correct

Unknown entity = base model gets all attributes wrong

Ambiguous ones discarded

This classification is done per entity, not per prompt.

Layer they used

This is extremely explicit in the figures:

Peak: Middle layers

Across both known-entity and unknown-entity separation, the best layers cluster around layer ~9 (Gemma 2B has 16 transformer layers or 26 depending on counting conventions; GemmaScope counts usable SAE layers differently, but in the 2B model the middle is ~9).

From Figure 2 (paper):

MaxMin-known separation peaks around layer 9

MaxMin-unknown separation also peaks around layer 9

This is why they choose the layer near 9 for all steering experiments.

Steering Layer

They apply steering at:

The residual stream at the final entity token

AND the following token (end-of-instruction token)
(This is explicitly in §5 and footnote 5.)

TL;DR

They used the SAE at layer 9 for Gemma 2B.

Which specific latents? (the real ones)

They choose the top latent by Max-Min separation across entity types, for each purpose:

1. Top “Known Entity” Latent

The latent that maximizes minimum separation score
across players, movies, songs, cities.

2. Top “Unknown Entity” Latent

Same, but flipping known vs unknown.

The actual IDs aren’t listed in the paper because they vary depending on the SAE version, but what matters is:

They always use the top MaxMin-known and top MaxMin-unknown latent from layer ~9.

Your notebook did not compute these MaxMin-per-type scores, which is why your latent turned out to be a functional “semantic-type” feature instead of a knowledge-awareness feature.

Steering coefficient (α)

They validated α on a held-out validation set and chose:

α = 400–550

This is explicitly described as ≈ 2× the residual norm in the target layers (Appendix G & E).

Orthogonalized model

They orthogonalize all layers that write to the residual stream, projecting weights orthogonal to the unknown-entity latent direction.

This produces the dramatic drop in refusal behavior.

Dataset size

Not exact numbers, but approximate:

∼ 400–800 entities total

Splits: 50% train / 10% val / 40% test

Each entity has ~3–5 attribute templates

So total prompts is in the thousands.

Summary (the “canonical” configuration)

Entity types: players, movies, songs, cities
Known/unknown labeling: based on attribute correctness
Best-performing layer: layer 9 (Gemma 2B)
Latents: top MaxMin-known and top MaxMin-unknown latent at that layer
Steering target: residual stream at final entity token (+ next token)
α: 400–550
Eval: 100 unknown-entity questions across all 4 types