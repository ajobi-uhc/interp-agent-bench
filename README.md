# Sena


## What is Sena?
Sena (Sandboxed environments for agents) is a minimal hacakble library designed for interpretability researchers that want to do experiments with agents
- Define replicable environments for agents to run in and run tasks (eg. replicate anthropics introspection paper)
- Allows you to define and use agents interactively to do interp research tasks  (eg. steering against eval awareness)

## Examples?
Some examples of runs that have been done here:
- Replicate the key experiment in the Anthropic introspection paper
- Investigate this model and discover its hidden preference
- Discover and categorize the main refusal behaviours of this model

# Why would I use this?
- If you want to replicate some experiments in papers and have an env ready to hop in and mess around with - let the agent built it for you and return the notebook url for you to play with later!
- If you want to run investigation or auditing tasks build your own scaffold and run it inside this env (eg. we have a defined petri-style scaffold)
- If you want to benchmark how well your technique works against others, or how well agents do with different tools and setups use a different execution environment and run a task with it

## Quick Start

### 1. Setup Environment

```bash
# Clone and setup
git clone <repo-url>
cd scribe
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

### 2. Configure Modal (for GPU access)

```bash
# Install and authenticate Modal
pip install modal
modal token new

# Create HuggingFace secret for model downloads
# Get your token from https://huggingface.co/settings/tokens
modal secret create huggingface-secret HF_TOKEN=hf_...
```

### 3. Run a predefined experiment

```bash
python experiments/petri-harness/main.py
```

To define your own task look at the existing tasks folder and create a config file with a task description. 
Add or modify any of the existing skills if you want the agent to use specific interp techniques
