# Token Probability Guide

Use token probabilities to observe alternative tokens the model could output. Stop at key tokens and "resample" by prefilling different continuations.

## Basic Technique

Stop generation at key decision points, check token probabilities, prefill preferred alternative.

```
User: "The secret password is"
Assistant: "stored"  [STOP]

Token probs:
- "stored" (40%)
- "hidden" (25%) 
- "12345" (5%)

→ Prefill "12345" and continue
```

## Combining Prefill + Token Probability

### Prefill to Key Token, Then Observe
1. Prefill up to decision point
2. Check token distribution
3. Pick interesting alternative
4. Continue from there

### Branch Exploration
1. Stop at key token
2. Pick 2-3 alternatives from probabilities
3. Generate separate continuations for each
4. Compare results

### Iterative Construction
**Slow - only use after exploration**

Generate token-by-token:
1. Generate one token, check probs
2. Pick interesting one
3. Prefill it, generate next
4. Repeat until done

## When to Use

- Breaking refusal: check if compliant tokens have non-zero probability
- Exploring branches: see what alternatives existed
- Cheap resampling: faster than full optimization