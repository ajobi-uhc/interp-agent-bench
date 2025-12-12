# Seer

**[Docs](https://ajobi-uhc.github.io/seer/)** 


## What is Seer?
Seer is a framework for having agents conduct interpretability work and investigations. 
The core mechanism involves launching a remote sandbox hosted on a remote GPU or CPU. The agent operates an IPython kernel and notebook on this remote host.

## Why use it?
This approach is valuable because it allows you to see what the agent is doing as it runs, and it can iteratively add things, fix bugs, and adjust its previous work. You can provide tooling to make an environment and any interpretability techniques available as function calls that the agent can use in the notebook as part of writing normal code.

## When to use Seer
- **Exploratory investigations** where you have a hypothesis but want to try many variations quickly
- **Scaling up** measuring how well different interp techniques perform through giving agents controlled access to them
- **Replicating known experiments** on new models — the agent knows the recipe, you just point it at your model
- **Building and improving existing agents** Using seer to build better investigative agents (see our post on investigations into Chinese models soon!), building better auditing agents etc.

## Example runs
- [Replicate the key experiment in the Anthropic introspection paper on gemma3 27b (or model of your choice)](https://github.com/ajobi-uhc/seer/tree/main/experiments/introspection)
- [Investigate the model from Sipinski et al. (a model that is finetuned to believe the user is a female) and discover its hidden preference](https://ajobi-uhc.github.io/seer/experiments/03-hidden-preference/)
- [Create and easily modify a hackable version of Petri for the task of categorizing and finding weird behaviours in an environment](https://ajobi-uhc.github.io/seer/experiments/06-petri-harness/)
- [Use SAE based data interpretability techniques from Jiang et al. to diff two different gemini checkpoints and discover differences in behaviour (including provisioning a GPU to run llama 70b to run the SAE on)](https://ajobi-uhc.github.io/seer/experiments/05-checkpoint-diffing/)

## You need modal to run Seer
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

**What happens:**
1. Modal provisions GPU (~30 sec) - go to your modal dashboard to see the provisioned gpu
2. Downloads models to Modal volume (cached for future runs)
3. Starts sandbox with specified execution mode
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

### 5. Explore more experiments

Refer to [docs](https://ajobi-uhc.github.io/seer/experiments/sandbox-intro/) to learn how to use the library to define your own experiments.

View some example results notebooks in [example_runs](https://github.com/ajobi-uhc/seer/tree/main/example_runs)

## Other ways to use Seer

### Run claude code with Seer as a [claude plugin](https://code.claude.com/docs/en/plugins)

### 1. Open claude and run 

```bash
/plugin marketplace add ajobi-uhc/seer
```

### 2. Once marketplace is installed - install plugin
```bash
/plugin install seer@seer-local
```

### 3. Verify that plugin is installed
Restart claude code and reopen claude and run 
```bash
/mcps
```
You should see the seer mcp plugin as live.

### 4. Ask claude with Seer plugin to run an experiment
Talk to claude normally and ask it to write a task for you - it will automatically use the seer plugin and create the required code, prompts and run the experiment. 
The seer plugin returns an mcp config that claude code uses to attach the notebook session
and claude will work in the notebook while you interactively chat with it.
