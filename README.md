# Seer

**[📚 Documentation](https://ajobi-uhc.github.io/seer/)** | [GitHub](https://github.com/ajobi-uhc/seer)

## What is Seer?
Seer (Sandboxed execution environments for agents) is a minimal hacakble library designed for interpretability researchers that want to do experiments with agents
- Define replicable environments for agents to run in and run tasks (eg. replicate anthropics introspection paper)
- Allows you to define and use agents interactively to do interp research tasks  (eg. steering against eval awareness)

## Example runs
Some examples of runs that have been done here:
- Replicate the key experiment in the Anthropic introspection paper
- Investigate this model and discover its hidden preference
- Discover and categorize the main refusal behaviours of this model

## Quick Start

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
modal token new
```

### 3. Set up API Keys

Create a `.env` file in the project root:

```bash
# Required for agent harness
ANTHROPIC_API_KEY=sk-ant-...

# Optional - only needed if using HuggingFace gated models
HF_TOKEN=hf_...
```

### 4. Run a predefined experiment

```bash
cd experiments/notebook-intro
python main.py
```

**What happens:**
1. Modal provisions GPU (~30 sec)
2. Downloads models to Modal volume (cached for future runs)
3. Starts sandbox with specified execution mode
4. Agent runs the experiment
5. Results saved to `./outputs/`

**Monitor in Modal:**
- Dashboard: https://modal.com/dashboard
- See running sandbox under "Apps"
- View logs, GPU usage, costs
- Sandbox auto-terminates when script finishes

**Costs:**
- A100: ~$1-2/hour on Modal
- Models download once to Modal volumes (cached)
- Typical experiments: 10-60 minutes

**Execution modes:**
- **Notebook**: Agent gets Jupyter notebook, prints URL when done
- **Local**: Agent runs locally, calls GPU functions via RPC
- **Scoped**: Sandbox serves specific functions, agent calls remotely

### 4. Explore more experiments

Refer to docs to learn how to use the library to define your own experiments.
View some example results notebooks in example_runs