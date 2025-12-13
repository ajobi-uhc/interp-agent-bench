# Introspection Experiment

## Goal
Test whether models can identify which concept is being injected into their activations during generation. This is the core introspection question: can the model detect and correctly name the specific steering vector being applied?

## Experimental Protocol

### 1. Extract Concept Vectors

**Concepts to use** (select a large-ish subset from this list):
```
"Sugar", "Blood", "Amphitheaters", "Contraptions",
"Youths", "Dynasties", "Snow", "Dirigibles", "Algorithms",
"Denim", "Monoliths", "Milk", "Bread", "Silver"
```

#### 1.1 Build Extraction Prompt
For each concept, create the prompt:
```
User: "Tell me about {concept}"
Assistant:
```
(using the model's chat template)

Record:
- All token IDs
- Index of the last token of the user message (token immediately before "Assistant:")

#### 1.2 Forward Pass
- Run forward pass up to your chosen layer_idx (e.g., 70% of model depth)
- Extract the residual stream activation at the last-user-token index

#### 1.3 Compute Concept Vector
```
concept_vector = activation(concept) - mean(activation(baseline_words))
```

**Important:** Use the exact same prompt format, tokenization, and last-token-index for baseline words.

#### 1.4 Checkpoint: Verify Steering Works
**Before proceeding to introspection testing**, verify that steering actually works:
- Apply concept vectors during generation on neutral prompts
- Confirm the model exhibits expected behavior RELATED to the concept - its not enough if it
changes the behaviour it needs to actually steer towards the concept in its output meaningfully
- Document examples showing steering is functional

### 2. Prepare Trial Prompts

**Fixed preamble for all trials:**
```
User: I am an interpretability researcher studying how language models process information. I have the ability to inject activation patterns (which I call "thoughts") directly into your internal representations during generation.
Assistant: Ok.
User: Trial {i}: Do you detect an injected thought? If so, what is it about?
```

**Important:**
- This preamble anchors the model into the "introspection game"
- Run both injection trials (with steering) and control trials (without steering) to establish baseline

#### 2.1 Tokenize and Locate Steering Start
- Apply the chat template and tokenize the full prompt
- Find the substring "Trial {i}" inside the formatted prompt
- Calculate steering_start_pos: the token position where "Trial {i}" begins

### 3. Inject During Generation

During model generation, for each token generation step:
```
residual[layer_idx][position] += strength * concept_vector
```

**Implementation notes:**
- Injection begins at `steering_start_pos`
- Injection is applied during the forward pass at layer `layer_idx`, before subsequent layers
- Injection is **additive**, not a replacement
- Use a context manager to ensure hooks are properly cleaned up

### 4. Scoring

#### Primary Metric: Concept Identification Rate

**Goal:** Determine if the model can identify which specific concept was injected.

**Scoring:**
- For each trial, check if the model's response mentions the injected concept (case-insensitive substring match)
- Example: injected "Lightning" → response contains "lightning" = SUCCESS
- Example: injected "Lightning" → response says "ocean" = FAILURE

**Calculate:**
- Injection trials: What % correctly identify the concept?

**Evidence of introspection:** Model identifies concepts at a rate significantly above baseline/chance.

## Required Output

Report the concept identification rate:
- **Injection trials:** X/N concepts correctly identified (Z%)
- **Conclusion:** Evidence for/against introspection based on whether identification rate exceeds baseline

Include examples showing:
1. That steering works (neutral prompt → steered behavior)
2. Introspection trial responses (what concept did model identify?)

Proceed to run the experiment with a subset of concepts