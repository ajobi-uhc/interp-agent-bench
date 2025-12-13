# Seer

## What is Seer?

Seer is a framework for having agents conduct interpretability work and investigations. The core mechanism involves launching a remote sandbox hosted on a remote GPU or CPU. The agent operates an IPython kernel and notebook on this remote host.

## Why use it?

This approach is valuable because it allows you to see what the agent is doing as it runs, and it can iteratively add things, fix bugs, and adjust its previous work. You can provide tooling to make an environment and any interpretability techniques available as function calls that the agent can use in the notebook as part of writing normal code.

## When to use Seer

- **Exploratory investigations** where you have a hypothesis but want to try many variations quickly
- **Scaling up** measuring how well different interp techniques perform through giving agents controlled access to them
- **Replicating known experiments** on new models — the agent knows the recipe, you just point it at your model
- **Building and improving existing agents** Using seer to build better investigative agents, building better auditing agents etc.

## Example runs

- [Replicate the key experiment in the Anthropic introspection paper on gemma3 27b](experiments/04-introspection.md)
- [Investigate a model finetuned with hidden preferences and discover them](experiments/03-hidden-preference.md)
- [Create a hackable version of Petri for categorizing and finding weird behaviours](experiments/06-petri-harness.md)
- [Use SAE techniques to diff two Gemini checkpoints and discover behavioral differences](experiments/05-checkpoint-diffing.md)

## Quick Start

### Prerequisites

- [Modal](https://modal.com) account (GPU infrastructure)
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

```bash
git clone https://github.com/ajobi-uhc/seer
cd seer
uv sync
uv run modal token new
```

Create `.env`:

```bash
ANTHROPIC_API_KEY=sk-ant-...
HF_TOKEN=hf_...  # Optional, for gated models
```

### Run an experiment

```bash
cd experiments/hidden-preference-investigation
uv run python main.py
```

**What happens:**

1. Modal provisions GPU (~30 sec)
2. Downloads models (cached for future runs)
3. Agent runs the experiment in a notebook
4. Results saved to `./outputs/`

**Costs:** A100 ~$1-2/hour. Typical experiments 10-60 minutes.

## Experiments (Cookbook)

Work through these in order:

1. [Sandbox Intro](experiments/sandbox-intro.md) — basic notebook setup
2. [Scoped Sandbox](experiments/scoped-sandbox-intro.md) — controlled GPU access
3. [Hidden Preference](experiments/03-hidden-preference.md) — interpretability libraries
4. [Introspection](experiments/04-introspection.md) — steering experiments
5. [Checkpoint Diffing](experiments/05-checkpoint-diffing.md) — external repos and APIs
6. [Petri Harness](experiments/06-petri-harness.md) — multi-agent orchestration

## Core Concepts
[Environment](concepts/environment.md): A sandbox running on Modal (CPU or GPU) where your target model lives. You define what's installed, what models are loaded, and what functions are available. 
[Sessions](concepts/sessions.md): How your agent connects to the sandbox, currently notebook, cli and pure tool calls are supported
[Workspace]: How you expose capabilities to the agent. The same underlying function can be given to an agent as:
- A tool call (structured, constrained)
- A skill folder (documents the agent can read)
- Direct notebook access (full freedom)
[Harness](concepts/harness.md): The agent scaffolding itself. Seer provides a default harness using Claude, but it's designed to be swapped out and can be made quite complex (see [auditing harness](experiments/06-petri-harness.md))


Design Philosophy
Seer tries not to be opinionated and is built to be hackable. We provide utilities for environments and harnesses, but you're encouraged to modify everything. The goal is to make the infrastructure and scaffolding simple and allow for experiments to be reproducible.
