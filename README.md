# Seer
a small hackable library that makes it easier to do interpretability work with agents

#### [Docs](https://ajobi-uhc.github.io/seer/)  
#### [Markdown docs for LLM](https://raw.githubusercontent.com/ajobi-uhc/seer/main/docs/llm-context.md)


## What is Seer?
Seer is a library for interpretability researchers who want to do research on or with agents. It makes use cases like creating environments for agents, equipping an agent with your technique and building on papers easier-and fixes some of the annoying things you get from just using Claude Code out of the box.

The core mechanism: you specify an environment (github repos, files, dependencies), Seer launches it as a sandbox on Modal (GPU or CPU), and an agent operates within it via an IPython kernel. 
This setup means you can see what the agent is doing as it runs, it can iteratively fix bugs and adjust its work, and you can spin up many sandboxes in parallel.

Seer is designed to be extensible - you can build on top of it to support complex techniques that you might want the agent to use, eg. [giving an agent SAE tools to diff two Gemini checkpoints](https://ajobi-uhc.github.io/seer/experiments/05-checkpoint-diffing/) or [building a Petri-style auditing agent with whitebox tools](https://ajobi-uhc.github.io/seer/experiments/06-petri-harness/).

### Modal is sponsoring $250 in GPU credits for Seer users
Receiving a code might take some time. In the meantime you can [signup to Modal now](https://modal.com/) and use the default $30 in free credits to start playing around with Seer.

#### [Request your code here](https://docs.google.com/forms/d/e/1FAIpQLSeueUKsH5kPJtDn5gz1Bm5xFPosx3PjyzC49lfV1o_CUm1UJw/viewform?usp=dialog)



## When to use Seer
- **Exploratory investigations**: You have a hypothesis about a model's behavior but want to try many variations quickly without manually rerunning notebooks
    - Case study: [Hidden Preference](https://ajobi-uhc.github.io/seer/experiments/03-hidden-preference/) - investigate the model (from Cywinski et al. [link](https://arxiv.org/pdf/2510.01070)) where a model has been finetuned to have a secret preference to think the user it's talking to is a female
- **Give agents access to your techniques**: Expose methods from your paper to the agent and measure how well they use them across runs
    - Case study: [Checkpoint Diffing](https://ajobi-uhc.github.io/seer/experiments/05-checkpoint-diffing/) - agent uses data-centric SAE techniques from [Jiang et al.](https://www.lesswrong.com/posts/a4EDinzAYtRwpNmx9 towards-data-centric-interpretability-with-sparse) to diff Gemini 2.0 and Gemini 2.5 checkpoints
- **Build on existing papers**: Clone a paper's repo into the environment and the agent can work with it directly - run on new models, modify techniques, or use their tools in a larger investigation
    - Case study: [Introspection](https://ajobi-uhc.github.io/seer/experiments/04-introspection/) — replicate the Anthropic introspection [experiment](https://www.anthropic.com/research/introspection) on gemma3 27b (checkout [this](https://github.com/uzaymacar/introspective-awareness) repo for more experiments)
- **Building better agents**: Test different scaffolding, prompts, or tool access patterns
    - Case study: [Give an auditing agent whitebox tools](https://ajobi-uhc.github.io/seer/experiments/06-petri-harness/) — build a minimal & modifiable [Petri](https://github.com/safety-research/petri/tree/main)-style agent with whitebox tools (steering, activation extraction) for finding weird model behaviors

## How does Seer compare to Claude Code + a notebook?
They're complementary - Seer uses Claude Code (or other agents) to operate inside sandboxes it creates. 

Seer handles:
- Reproducibility: Complex environments, tools, and prompts defined as code
- Remote GPUs without setup: Sandboxes on Modal with models, repos, files pre-loaded
- Flexible tool injection: Expose techniques as tool calls or as libraries in the execution environment
- Run many experiments in parallel: Since its on a remote sandbox you can launch as many experiments in parallel as you want and benchmark different approaches across runs.


## Video showing me use Seer for a simple investigation
[![Seer demo](https://img.youtube.com/vi/k_SuTgUp2fc/maxresdefault.jpg)](https://youtu.be/k_SuTgUp2fc)


## You need modal to get the best out of Seer 
Modal has kindly offered to sponsor up to $250 in GPU credits for you to get started!
Fill in this [form](https://docs.google.com/forms/d/e/1FAIpQLSeueUKsH5kPJtDn5gz1Bm5xFPosx3PjyzC49lfV1o_CUm1UJw/viewform?usp=dialog) to be able to recieve a code

See [here](https://ajobi-uhc.github.io/seer/experiments/00-local-mode) to run an experiment locally without Modal

We use modal as the gpu infrastructure provider
To be able to use Seer sign up for an account on modal and configure a local token (https://modal.com/)




## Quick Start

Here the goal is to run an investigation on a custom model using predefined techniques as functions

### 0. Get a [modal](https://modal.com/) account
By default new accounts come with $30 USD in credits

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

### 5. Explore more experiments

Refer to [docs](https://ajobi-uhc.github.io/seer) to learn how to use the library to define your own experiments.

View some example results notebooks in [example_runs](https://github.com/ajobi-uhc/seer/tree/main/example_runs)

## Acknowledgements

This project builds on excellent work from:
- [Goodfire](https://goodfire.ai/) for their work on [Scribe](https://github.com/goodfire-ai/scribe), which powers the notebook execution layer
- [Modal](https://modal.com/) for providing the GPU sandbox infrastructure that makes remote experiments possible - and the credits!
- The many MATS scholars and friends who helped with feedback
