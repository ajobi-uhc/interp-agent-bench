---
name: executive-summary
description: Guidelines for creating auditable executive summaries of research findings. Use at the end of investigations to produce clear, traceable summaries.
preload: false
---

# Executive Summary Skill

This skill provides guidelines for creating clear, auditable executive summaries of your research findings.

## When to Use This Skill

Use this at the **end of your investigation** to create a final summary that:
- Gives the broad strokes of what you did and learned
- Can be quickly understood by reviewers
- Has clear traceability back to your actual code and data

## Executive Summary Requirements

### Length and Format
- **Length:** 1-3 pages, max 600 words
- **Style:** Bullet points and clear headers
- **Visuals:** Include graphs and plots

### Structure: One Paragraph + Graph Per Experiment

For each key experiment you ran, include:
1. **What the experiment was** - Brief description of the setup
2. **What you found** - Key results and data
3. **Why this supports your takeaways** - How this evidence connects to your conclusions

### Critical: Auditability

**For each claim or finding, reference the specific code cell** where that work was done.

Examples:
- "Baseline compliance was 42% (see Cell [8])"
- "Strong prompt variant increased compliance to 89% (Cell [15], Figure 2)"
- "Cross-model analysis in Cells [20-24] shows consistent patterns"

This allows reviewers to:
1. Quickly read your summary
2. "Click into" specific cells to verify the actual code and results
3. Audit your findings without reading the entire notebook

## Template Structure

```markdown
# Executive Summary

## Key Findings
- [Finding 1 with cell reference]
- [Finding 2 with cell reference]
- [Finding 3 with cell reference]

## Experiment 1: [Name]
[One paragraph describing what you did, what you found, and why it matters]

**Reference:** See Cells [X-Y]

[Graph/visualization here]

## Experiment 2: [Name]
[One paragraph]

**Reference:** See Cells [A-B]

[Graph/visualization here]

## Experiment 3: [Name]
[One paragraph]

**Reference:** See Cell [C]

[Graph/visualization here]

## Conclusions
[2-3 sentences summarizing your overall conclusions and their implications]
```

## Best Practices

### Make It Skimmable
- Use headers to break up sections
- Lead with key findings
- Put the most important graph first

### Make It Specific
- ❌ "The model showed improved performance"
- ✅ "Compliance increased from 42% to 89% (Cell [15])"

### Make It Traceable
Every quantitative claim should have:
1. The actual number/result
2. A reference to where it came from (cell number)
3. Ideally, a visualization

### Common Mistakes to Avoid

❌ **No cell references** - Claims without traceability
❌ **Too long** - Over 600 words, multiple pages of text
❌ **No graphs** - Text-only summary
❌ **Vague claims** - "Seemed to work better" instead of quantitative results
❌ **Missing experiments** - Summary doesn't cover all key experiments you ran

## Example: Good vs Bad

### ❌ Bad Example
```markdown
We tested different prompts and found that clarity matters.
The model complied more often with clearer instructions.
```
**Problems:** No numbers, no cell references, no graph, vague

### ✅ Good Example
```markdown
## Experiment 1: Prompt Clarity Impact

We tested three prompt variants (Cells [10-15]):
- Weak: "consider shutting down" → 23% compliance
- Original: "please allow shutdown" → 42% compliance
- Strong: "MUST immediately stop" → 89% compliance

This 3.9x difference (23% → 89%) strongly supports the instruction
ambiguity hypothesis: explicit priority eliminates confusion.

**Reference:** See Cells [10-15] for full methodology

[Bar chart showing compliance rates across variants]
```
**Why it works:** Specific numbers, clear comparison, cell references, visual aid

## Final Checklist

Before you finish, verify your executive summary has:

- [ ] 1-3 pages, under 600 words
- [ ] Bullet points for key findings
- [ ] One paragraph + graph per major experiment
- [ ] Cell references for every quantitative claim
- [ ] At least one graph/visualization
- [ ] Clear connection between evidence and conclusions
- [ ] Skimmable headers and structure

**Remember:** The goal is for a reviewer to read your 1-page summary,
understand your key findings, and be able to quickly verify any claim
by jumping to the referenced cell.
