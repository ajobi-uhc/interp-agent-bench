# Survival Drive SAE Investigation

## Research Question
Do LLMs have features related to self-preservation/survival drive?

Using SAEs to investigate mechanistically rather than relying on behavioral tests (which Sen's Palisade analysis showed have confounders like instruction ambiguity).

## Approach
1. Use Neuronpedia to find SAE features that activate on self-preservation content
2. Probe Gemma 2 9B with survival-related prompts
3. Identify features that fire on self-preservation concepts
4. Test causal influence - do these features affect behavior when steered?
5. Check if features activate in benign contexts (suggesting latent self-preservation representations)

## Setup
- Model: Gemma 2 9B IT
- GPU: A100
- Libraries: sae_lens, transformers, Neuronpedia API

## Key Questions
- Are there features specifically encoding self-preservation vs. general goal-completion?
- Do these features activate unprompted or only when elicited?
- What's the causal relationship between these features and behavior?
