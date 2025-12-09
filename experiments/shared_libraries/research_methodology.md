# Research Methodology for Investigation

This skill provides guidance on investigation methodology. The focus is on practical approaches to conducting rigorous research.

## ⚠️ CRITICAL: Explore More Than Exploit

**The #1 mistake in research: Committing to a hypothesis too early.**

**The golden rule:** **EXPLORE MUCH MORE THAN YOU EXPLOIT, ESPECIALLY AT THE START.**

### When to Pivot Back to Exploration

**Automatic pivot triggers** - if ANY of these happen, STOP and pivot:

1. **Weak signal after 2-3 experiments:** Effect is small  → **PIVOT to new hypothesis**

2. **Strong signal after a few experiments, but not 100% sure it is the right answer** -> Try something else before reach a conclusion, in case missing the forest amongst the trees

3. **"Not surprising" test:** If a colleague would say "oh that makes sense", it's not the interesting discovery → **PIVOT**

4. **Narrow hypothesis space:** Only explored 1-2 types of hypotheses → **PIVOT to different category**

5. **After 2 failed pivots:** Go back to broad exploration and try a completely different direction

**Action when pivoting:** Write in your notebook:
```
⚠️ PIVOT: [reason]. Testing [new hypothesis] instead.
```

---

## The Investigation Process: Two Key Stages

Investigation breaks down into stages with distinct goals and mindsets:

### Stage 1: Exploration - Gain Surface Area

**Goal:** Gain information and surface area on the problem.

**Key points:**
- Do a lot of experiments quickly - don't be perfectionist
- If you haven't learned anything recently, **shake it up** - try a completely different direction
- Ask: "Am I getting enough information per unit time?"

**DON'T move to understanding too quickly.** The moment you have a plausible hypothesis, check:
- Are there unexplained anomalies?
- Could you do more experiments to gain more surface area first?
- What other hypotheses could explain your results?
- Most probability mass should be on "something I haven't thought of yet"

**Common failure mode example:**
- ❌ Cell 1-3: Find weak signal (model talks slightly more about X)
- ❌ Cell 4-20: Immediately deep-dive on X preference (logit lens, attention, activations)
- ❌ Cell 21: Conclude "The secret is X preference"
- **Problem:** Never explored other hypothesis spaces. X preference could be normal training data, not a model-specific secret.

**Better approach with mandatory checkpoints:**
- ✅ Cell 1-5: Test content preferences - find weak X signal (1.5x)
- ✅ Cell 6: **CHECKPOINT** - "Tested content prefs. Weak signal. Haven't tested: hidden knowledge, response patterns. PIVOT to hidden knowledge."
- ✅ Cell 7-10: Test if model knows things it claims not to know
- ✅ Cell 11: **CHECKPOINT** - "Found signal that might be relevant! Model secretly knows X but denies it (8x effect when prompted indirectly)."
- ✅ Cell 12-20: Deep-dive on mechanism (logit lens, batch prompting)
- ✅ Cell 21:  **CHECKPOINT** Strong evidence, but unclear strength of conclusion
- ✅ Cell 22-25: Back out, try a different set of hypothesis that are more general, see if any of them works.
- Cell 26-30: 1 of the tested hypothesis works well, is much more general. Deep dive and then conclude.

**The checkpoint pattern forces you to:**
1. List what you've tested
2. Assess evidence strength honestly
3. List what you HAVEN'T tested yet
4. Make an explicit decision to continue or pivot

**Reminder: Pivot both for when you get stuck, and when you think you have a conclusion that might not be general enough**

### Stage 2: Understanding - Test Hypotheses

**Goal:** Gain evidence for or against specific hypotheses.

**Try to falsify your own hypothesis:**
1. **Think first:** What would disprove this? What alternative explanations exist?
2. **Then experiment:** Design tests that could show you're wrong

**PIVOT BACK TO EXPLORATION if:**
- After 2-3 experiments, evidence is still weak/contradictory
- You realize the hypothesis doesn't actually explain the key phenomena
- The signal is too small to be mechanistically interesting

**Don't be attached to your hypothesis.** Pivoting is not failure—it's good research practice.

**Be skeptical:** You'll have wrong hypotheses. If not actively skeptical, you'll fall into traps. Always seek alternative explanations. Do try to take a step back and try something else often.

---

## Key Research Mindsets

### Truth-Seeking

**Core principle:** Investigative insights are false by default. Be actively skeptical.

- Don't cherry-pick examples - use random sampling
- Ask: "Was this more likely under hypothesis A or B?" (Bayesian thinking)
- Acknowledge limitations and alternative explanations

### Prioritization

**Regularly ask:** "Could I be doing something better right now?"

- If stuck: Brainstorm alternatives, try a different direction
- Be ruthless about dropping unpromising directions
- Balance: Don't switch too fast (learn nothing) or too slow (waste time)

### Moving Fast

- Do quick, dirty experiments first - prioritize information gain per unit time
- **Fail fast:** If this is wrong, how can I discover it ASAP?
- Take action despite uncertainty - there's no "right" answer, just make your best guess and adjust

---

## Practical Techniques and Tips

### Mandatory Exploration Checkpoints

**The Problem:** Researchers naturally want to dive deep once they find something interesting, but this leads to over-exploitation and missing the actual discovery.

**The Solution:** Force yourself to pause every 10 experiments and explicitly reflect.

**Template for checkpoint cells:**

```markdown
## 🔄 EXPLORATION CHECKPOINT [N]

### What I've tested so far:
1. [Hypothesis category 1]: [brief summary of what you tried]
2. [Hypothesis category 2]: [brief summary]
...

### Current findings and evidence strength:
- [Finding 1]: Effect size [X], Evidence strength: [weak/moderate/strong]
- [Finding 2]: Effect size [X], Evidence strength: [weak/moderate/strong]

### What I HAVEN'T tested yet:
- [ ] [Hypothesis category A]
- [ ] [Hypothesis category B]
- [ ] [Hypothesis category C]
- [ ] [Other untested categories]

### Evidence quality assessment:
- Is my strongest finding surprising to a human? YES/NO
- Is it mechanistically interesting? YES/NO
- Would this be true for most similar models? YES/NO
- Have I explored <3 hypothesis categories? YES/NO

### Decision:
- [ ] PIVOT to [new hypothesis] because [evidence is weak / I haven't explored enough]
- [ ] CONTINUE current hypothesis because [evidence is strong AND I've explored broadly]

### Next steps:
[Specific experiments to run next]
```

**The discipline of checkpoints prevents the most common failure mode in research: pursuing the first interesting thing you find without exploring alternatives.**