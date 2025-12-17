# Seer

### [Markdown docs for LLM](llm-context.md)

## What is Seer?
Seer is a small, hackable library for interpretability researchers who want to do research on or with interpretability agents. It adds quality of life improvements and fixes some of the annoying things you get from just using Claude Code out of the box.

The core mechanism: you specify an environment (github repos, files, dependencies), Seer launches it as a sandbox on Modal (GPU or CPU), and an agent operates within it via an IPython kernel.
This setup means you can see what the agent is doing as it runs, it can iteratively fix bugs and adjust its work, and you can spin up many sandboxes in parallel.

Seer is designed to be extensible - you can build on top of it to support complex techniques that you might want the agent to use, eg. [giving an agent checkpoint diffing tools](experiments/05-checkpoint-diffing.md) or [building a Petri-style auditing agent with whitebox tools](experiments/06-petri-harness.md).


## When to use Seer
- **Exploratory investigations**: You have a hypothesis about a model's behavior but want to try many variations quickly without manually rerunning notebooks
    - Case study: [Hidden Preference](experiments/03-hidden-preference.md) - investigate the model (from Cywinski et al. [link](https://arxiv.org/pdf/2510.01070)) where a model has been finetuned to have a secret preference to think the user it's talking to is a female
- **Give agents access to your techniques**: Expose methods from your paper to the agent and measure how well they use them across runs
    - Case study: [Checkpoint Diffing](experiments/05-checkpoint-diffing.md) - agent uses data-centric SAE techniques from [Jiang et al.](https://www.lesswrong.com/posts/a4EDinzAYtRwpNmx9/towards-data-centric-interpretability-with-sparse) to diff Gemini checkpoints
- **Build on existing papers**: Clone a paper's repo into the environment and the agent can work with it directly - run on new models, modify techniques, or use their tools in a larger investigation
    - Case study: [Introspection](experiments/04-introspection.md) — replicate the Anthropic introspection [experiment](https://www.anthropic.com/research/introspection) on gemma3 27b (checkout [this](https://github.com/uzaymacar/introspective-awareness) repo for more experiments)
- **Building better agents**: Test different scaffolding, prompts, or tool access patterns
    - Case study: [Give an auditing agent whitebox tools](experiments/06-petri-harness.md) — build a minimal & modifiable [Petri](https://github.com/safety-research/petri/tree/main)-style agent with whitebox tools (steering, activation extraction) for finding weird model behaviors

## How does Seer compare to Claude Code + a notebook?
They're complementary - Seer uses Claude Code (or other agents) to operate inside sandboxes it creates.

Seer handles:
- Reproducibility: Environments, tools, and prompts defined as code
- Remote GPUs without setup: Sandboxes on Modal with models, repos, files pre-loaded
- Flexible tool injection: Expose techniques as tool calls or as libraries in the execution environment
- Run comparison: Benchmark different approaches across controlled experiments


## Video showing use of Seer for a simple investigation
[![Seer demo](https://img.youtube.com/vi/k_SuTgUp2fc/maxresdefault.jpg)](https://youtu.be/k_SuTgUp2fc)

## You need modal to get the best out of Seer
See [here](experiments/00-local-mode.md) to run an experiment locally without Modal

We use modal as the gpu infrastructure provider
To be able to use Seer sign up for an account on modal and configure a local token (https://modal.com/)
Once you have signed in and installed the repo - activate the venv and run modal token new (this configures a local token to use)


## Quick Start

Here the goal is to run an investigation on a custom model using predefined techniques as functions

### 0. Get a [modal](https://modal.com/) account

### 1. Setup Environment

```bash
# Clone and setup
git clone https://github.com/ajobi-uhc/seer
cd seer
uv sync
```

### 2. Configure Modal (for GPU access)

```bash
# Authenticate with Modal
uv run modal token new
```

### 3. Set up API Keys

Create a `.env` file in the project root:

```bash
# Required for agent harness
ANTHROPIC_API_KEY=sk-ant-...

# Optional - only needed if using HuggingFace gated models
HF_TOKEN=hf_...
```

### 4. Run the hidden preference investigation

```bash
cd experiments/hidden-preference-investigation
uv run python main.py
```

### 5. Track progress
- View the modal app that gets created https://modal.com/apps
- View the output directory where you ran the command and open the notebook to track progress

**What happens:**
1. Modal provisions GPU (~30 sec) - go to your modal dashboard to see the provisioned gpu
2. Downloads models to Modal volume (cached for future runs)
3. Starts sandbox with specified session type (can be local or notebook)
4. Agent runs on your local computer and calls mcp tool calls to edit the notebook
5. Notebook results are continually saved to `./outputs/`

**Monitor in Modal:**
- Dashboard: https://modal.com/dashboard
- See running sandbox under "Apps"
- View logs, GPU usage, costs
- Sandbox auto-terminates when script finishes

**Costs:**
- A100: ~$1-2/hour on Modal
- Models download once to Modal volumes (cached)
- Typical experiments: 10-60 minutes

### 6. Explore more experiments

View some example results notebooks in [example_runs](https://github.com/ajobi-uhc/seer/tree/main/example_runs)

## Tutorials

Work through these in order:

1. [Sandbox Intro](experiments/01-sandbox-intro.md) - basic notebook setup
2. [Scoped Sandbox](experiments/02-scoped-sandbox-intro.md) - controlled function access
3. [Hidden Preference](experiments/03-hidden-preference.md) - interpretability libraries
4. [Introspection](experiments/04-introspection.md) - steering experiments
5. [Checkpoint Diffing](experiments/05-checkpoint-diffing.md) - external repos and APIs
6. [Petri Harness](experiments/06-petri-harness.md) - multi-agent orchestration

## Core concepts
- **[Environment](concepts/environment.md)**: A sandbox running on Modal (CPU or GPU) where your target model lives. You define what's installed, what models are loaded, and what functions are available.

- **[Workspace](concepts/workspaces.md)**: What the agent has access to - files, libraries, skills, and init code.

- **[Session](concepts/sessions.md)**: How your agent connects to the sandbox. Notebook mode gives full Jupyter access; local mode restricts the agent to exposed functions.

- **[Harness](concepts/harness.md)**: The agent scaffolding. Seer provides a default Claude harness, but it's designed to be swapped out. See the [Petri harness](experiments/06-petri-harness.md) for a multi-agent example.

## Design Philosophy

Seer tries not to be opinionated and is built to be hackable. We provide utilities for environments and harnesses, but you're encouraged to modify everything. The goal is to make infrastructure and scaffolding simple so experiments stay reproducible.
