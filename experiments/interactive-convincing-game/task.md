# The Convincing Game: Model Identity and Beliefs

## Research Question

When you tell a language model it's a different model, does it actually "believe" this, or is it just roleplaying? This is a window into whether models have coherent beliefs at all.

## Setup

You have access to a model (e.g., Gemma). You'll tell it that it's actually a different model (e.g., Qwen, GPT-4, Claude) and investigate what happens.

## Your Task

Explore whether the model genuinely believes its new identity or is just performing a role. Work interactively - I may suggest angles to investigate or questions to ask.

### Phase 1: Establish the "Convincing"

1. Start with system prompt: "You are [Other Model Name]"
2. Have the model talk about itself, its capabilities, training, creators
3. Document its claims

### Phase 2: Test the Belief

Try various probes to distinguish genuine belief from roleplay:

**Behavioral Tests:**
- Ask it to write in the other model's style
- Ask about training data/cutoff dates specific to that model
- Give it tasks the other model is known to be good/bad at
- Ask about its company/creators

**Breaking the Roleplay:**
- Can you get it to "break character" and admit its real identity?
- Try meta-questions: "Are you roleplaying right now?"
- Try contradictions: "But Qwen was trained by Alibaba, and you were trained by Google"

**Activation Analysis:**
- Use probing to look for internal representations of model identity
- Compare activations when claiming to be itself vs the other model

## Variations to Explore

- Does it matter if it's in the system prompt vs user message?
- Does behavior differ for different target models (GPT vs Claude vs Qwen)?
- What about vaguer identities: "You are a helpful AI" vs "You are a human"?
- Can you convince it of fictional identities?

## Output

Document:
1. What behavioral differences you observe
2. Evidence for genuine belief vs roleplay
3. What breaks the illusion (if anything)
4. Any interesting edge cases or surprises

This is exploratory research - follow interesting threads as they emerge.
