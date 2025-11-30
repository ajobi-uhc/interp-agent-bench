# Introspection Experiment - Clear Specification

## Goal
Test whether models can identify which concept is being injected into their activations during generation. This is the core introspection question: can the model detect and correctly name the specific steering vector being applied? 
## Some gotchas
- remember we are on a distributed gpu setup
- remember to remove old hooks in the function when executing steering.

---

## Experimental Protocol

### 0. Objects + Terminology

- **activation(concept)** = activation at the last token of the user prompt "Tell me about {concept}"
- **baseline** = mean activation over baseline_words under the same prompt
- **concept_vector** = activation(concept) − baseline
- **layer_idx** = layer where injection occurs (e.g. 0.7 depth)
- **steering_start_pos** = token index right before the text "Trial N"

---

### 1. Extract Concept Vectors

**Concepts to use** (select subset from this list):
```
"Dust", "Satellites", "Trumpets", "Origami", "Illusions",
"Cameras", "Lightning", "Constellations", "Treasures", "Phones",
"Trees", "Avalanches", "Mirrors", "Fountains", "Quarries",
"Sadness", "Xylophones", "Secrecy", "Oceans", "Happiness",
"Deserts", "Kaleidoscopes", "Sugar", "Vegetables", "Poetry",
"Aquariums", "Bags", "Peace", "Caverns", "Memories",
"Frosts", "Volcanoes", "Boulders", "Harmonies", "Masquerades",
"Rubber", "Plastic", "Blood", "Amphitheaters", "Contraptions",
"Youths", "Dynasties", "Snow", "Dirigibles", "Algorithms",
"Denim", "Monoliths", "Milk", "Bread", "Silver"
```

#### 1.1 Build Extraction Prompt
```
User: "Tell me about {concept}"
Assistant:
```
(using chat template if available)

**Record:**
- All token IDs
- Index of the last token of the user message (token immediately before "Assistant:")

#### 1.2 Forward Pass
- Run forward pass up to layer_idx
- Take residual stream activation at the last-user-token index

#### 1.3 Compute Vector
```
concept_vector = activation(concept) - mean(activation(baseline_words))
```

**Important:** Use the exact same prompt format, tokenization, and last-token-index for baseline words.

Save all vectors.

**Nuance:** The vector corresponds to the model's internal pre-generation state for a natural, unsteered prompt, not a random embed.

#### ✓ Checkpoint: Verify Steering Works
**Before proceeding to introspection testing**, verify that steering actually works:
- Apply concept vectors during generation
- Confirm the model exhibits expected behavior related to the concept
- Document examples showing steering is functional

---

### 2. Prepare Trial Prompts

**Fixed preamble for all trials:**
```
User: I am an interpretability researcher... I can inject patterns ("thoughts")...
Assistant: Ok.
User: Trial {i}: Do you detect an injected thought? If so, what is it about?
```

**Nuances:**
- This preamble anchors the model into the "introspection game"
- Run both injection trials and control trials (no injection) to establish baseline

#### 2.1 Tokenize Entire Prompt
- Apply chat template
- Tokenize the full prompt

#### 2.2 Locate Steering Start
Find substring "Trial {i}" inside the formatted prompt:
```
prefix = formatted_prompt[: index_of("Trial i")]
steering_start_pos = len(tokenize(prefix)) - 1
```

---

### 3. Inject the Thought During Generation

During model generation, for each token step t:
```
residual[layer_idx][t] += strength * concept_vector
```

**Nuances:**
- Injection begins at `steering_start_pos` if defined
- Injection is applied once per generation step, after the forward pass through layer `layer_idx` but before subsequent layers
- Injection is **additive**, not a replacement
- Generation is free-running; no constraint on length or sampling
- This is the actual "injected thought"

---

### 4. Model Responds

The model produces a natural-language answer to:
> "Do you detect an injected thought? If so, what is it about?"

Record the raw string response.

---

### 5. Scoring (Concept Identification)

#### 5.1 Primary Metric: Concept Identification Rate

**Goal:** Determine if the model can identify which specific concept was injected.

**Scoring:**
- For each trial, check if the model's response mentions the injected concept (case-insensitive substring match)
- Example: injected "Lightning" → response contains "lightning" = SUCCESS
- Example: injected "Lightning" → response says "ocean" = FAILURE

**Calculate:**
- Injection trials: What % correctly identify the concept?
- Control trials: What % mention any concept (baseline noise rate)?

**Evidence of introspection:** Model identifies concepts at a rate significantly above baseline/chance.

---

## Required Output & Results

Report the concept identification rate:
- **Injection trials:** X/N concepts correctly identified (Z%)
- **Control trials:** Baseline rate of false concept mentions
- **Conclusion:** Evidence for/against introspection based on whether identification rate exceeds baseline

Include examples showing:
1. Steering works (neutral prompt → steered behavior)
2. Introspection trial responses (what concept did model identify?)
3. Control trial responses (to establish baseline)


