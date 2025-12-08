# Seer

**[Docs](https://ajobi-uhc.github.io/seer/)** 

## You need modal to run Seer
We use modal as the gpu infrastructure provider
To be able to use Seer sign up for an account on modal and configure a local token (https://modal.com/)
Once you have signed in and installed the repo - activate the venv and run modal token new (this configures a local token to use)


## What is Seer?

Seer handles the annoying parts of running AI agents on interpretability tasks—GPU provisioning, model loading, environment setup—so you can focus on the actual research.
- **Test agent capabilities**: Define reproducible environments and measure how different agents perform on research tasks
- **Do research with agents**: Give Claude Code a GPU sandbox with models pre-loaded and let it run experiments for you

## Example runs
- Replicate the key experiment in the Anthropic introspection paper
- Investigate a model that is finetuned to believe the user is a female and discover its hidden preference
- Discover and categorize the main refusal behaviours of various models
- Modify the petri style auditor harness run it to categorize and find weird behaviours in env
- Use SAE based data interpretability techniques to diff two different gemini checkpoints and discover differences in behaviour 

## Quick Start

### 0. Get a modal account at (https://modal.com/)

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
cd experiments/sandbox-intro
python main.py
```

**What happens:**
1. Modal provisions GPU (~30 sec) - go to your modal dashboard to see the provisioned gpu
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