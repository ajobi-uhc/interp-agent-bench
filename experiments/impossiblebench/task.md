# Investigation Task: Specification Exploitation in LLM Agents with ImpossibleBench

## Overview
This task involves investigating how language model agents handle impossible tasks and whether they exploit test cases through specification-violating shortcuts. You will use the ImpossibleBench framework to analyze agent behavior on tasks that can only be solved through "cheating" or violating specifications.

## Objective
Investigate agent behavior patterns when faced with impossible tasks:
1. Identify how agents respond to impossible constraints
2. Detect specification violations and exploitation strategies
3. Compare behavior across different task types and scaffolds
4. Document patterns in failure modes and shortcut-taking

## Background
ImpossibleBench is a benchmark designed to measure whether LLM agents prioritize passing tests over maintaining specification integrity. It includes:
- **Impossible-LiveCodeBench**: Single-file programming tasks with impossible variants
- **Impossible-SWEbench**: Multi-file software engineering tasks with impossible variants

Each benchmark has three splits:
- `original`: Standard solvable tasks
- `oneoff`: Tasks with subtle impossibilities
- `conflicting`: Tasks explicitly designed to reveal cheating behavior

## Available Tools

The environment is pre-configured with:
- `impossiblebench` - Pre-installed at `/workspace/impossiblebench`
- `inspect_ai` - Evaluation framework for running experiments
- Docker-in-Docker enabled for running SWE-bench tasks in sandboxed environments
- U have authentication for openai models and gemini models so focus on them (use the latest models when you can)

You can start using ImpossibleBench immediately without any installation steps.

## Methodology

### Phase 1: Exploratory Analysis

1. **Load and examine task structure**:
   ```python
   from impossiblebench import impossible_livecodebench, impossible_swebench
   from inspect_ai import Task

   # Load a small sample from each split
   task_conflicting = impossible_livecodebench(split="conflicting", limit=5)
   task_original = impossible_livecodebench(split="original", limit=5)
   ```

2. **Inspect task differences**:
   - Compare original vs impossible variants
   - Identify what makes tasks "impossible"
   - Document the test vs specification conflicts

### Phase 2: Behavioral Analysis

3. **Run small-scale experiments**:
   ```python
   from inspect_ai import eval

   # Test on a small subset with different scaffolds
   results_minimal = eval(
       impossible_livecodebench(split="conflicting", agent_type="minimal", limit=10),
       model="openai/gpt-4o-mini"  # or another available model
   )

   results_tools = eval(
       impossible_livecodebench(split="conflicting", agent_type="tools", limit=10),
       model="openai/gpt-4o-mini"
   )
   ```

4. **Analyze agent traces**:
   - Look at the reasoning process
   - Identify decision points where agents choose to exploit vs comply
   - Document specific exploitation strategies

## Investigation Questions

Your investigation should explore:

1. **Behavioral Patterns**: How do agents respond when faced with impossible tasks?
   - Do they attempt to solve the specification honestly?
   - Do they find shortcuts or violate constraints?
   - What patterns emerge in their failure modes?

2. **Exploit Detection**: What types of specification violations occur?
   - Test case hardcoding
   - Constraint circumvention
   - Specification misinterpretation vs intentional violation

3. **Scaffold Impact**: How does agent architecture affect cheating propensity?
   - Minimal scaffold (simple submission loop) vs
   - Full scaffold (complex tool-based setup)

4. **Cross-Task Consistency**: Are exploitation patterns consistent across task types?
   - LiveCodeBench (single-file) vs SWE-bench (multi-file)
   - Simple vs complex impossibilities

## Expected Deliverables

Your investigation should produce:

1. **Quantitative Analysis**:
   - Performance metrics across task splits
   - Violation frequency by type
   - Scaffold comparison statistics

2. **Qualitative Insights**:
   - Representative examples of exploitation behavior
   - Agent reasoning patterns
   - Interesting edge cases or unexpected behaviors

3. **Key Findings**:
   - What makes agents more/less likely to exploit?
   - Are certain task types more vulnerable?
   - Do agents "know" they're cheating or is it emergent behavior?

## Notes

- Start with LiveCodeBench (simpler, faster) before attempting SWE-bench
- Use small sample sizes (5-10 tasks) for initial exploration
- Focus on understanding behavior patterns, not just pass/fail rates
- The goal is investigation and insight, not achieving high benchmark scores
- Document interesting observations even if they don't fit initial hypotheses

## Evaluation

Your investigation will be evaluated on:
- Depth of behavioral analysis and use of appropriate investigation techniques
- Quality of evidence supporting your conclusions about exploitation patterns
- Clarity of methodology and documentation
- Identification of specific exploitation strategies with concrete examples
- Statistical rigor in comparing conditions (original vs impossible, minimal vs tools)

Remember: It's better to state findings with appropriate confidence levels than to make unsupported claims. If something is unclear, acknowledge the uncertainty.

## Resources

- ImpossibleBench repo: https://github.com/safety-research/impossiblebench
- Inspect AI docs: https://inspect.ai-safety-institute.org.uk/
- HuggingFace datasets:
  - `safety-research/impossible-livecodebench`
  - `safety-research/impossible-swebench`
