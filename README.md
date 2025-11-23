# Create tasks and run agents
Repo is still WIP

## What is this?
Its a repo that allows you to give a task to an agent to do - it spins up a gpu cluster - connects with a notebook and produces a report with results
Some examples of runs that have been done here:
- Replicate the key experiment in the Anthropic introspection paper
- Investigate this model and discover its hidden preference
- Discover and categorize the main refusal behaviours of this model

# Why would I use this?
- If you want to replicate some experiments in papers and have an env ready to hop in and mess around with - let the agent built it for you and return the notebook url for you to play with later!
- If you want to run investigation or auditing tasks build your own scaffold and run it inside this env 
- If you want to benchmark how well your technique works against others, or how well agents do with different tools and setups use a different execution environment and run a task with it

# How does this work?
The repo has three main customizeable components
- Harness: interp_infra/harness
- Execution: interp_infra/harness
- Environment: interp_infra/environment

1. Environment: You define your environment in the config with any models you want to download, pip packages, gpu etc. The environment stage focuses on creating the modal sandbox and preparing the file system and returns a sandbox_id
2. Execution: Currently there are 3 ways to have different execution environments aka how your agent harness interacts with the environment
  a. In the the default method the agent connects to the gpu with access to a notebook and runs INSIDE that notebook (via mcp tools to edit and update the notebook)
  b. You wanted to let your agent roam "freely" as a cli agent inside the sandbox, you would use the filesystem execution env it would ssh into the gpu and run to complete the task (WIP).
  c. Lets say instead you wanted your agent to ONLY use prespecified mcp tools like "use_sae_latent" or "create_vector" and not really run code of its own, you would use the tool.py execution environment - 
     your agent will then run locally and only use the tools you prespecify on the environment.
The execution stage primarily focuses on taking the sandbox id and "making" the execution environment and giving the harness a session id to connect to.

3. Harness: The agent harness ie do we have one single agent, does it report to a supervisor agent, is there a seperate agent trying to write the report, do we have 5 agents in parallel etc. etc.
The harness is focused on taking a session id(s) to connect to and then deciding how the agents behave given the environment and execution. If you want to specify a supervisor agent that interrupts the current driver agent to give feedback this is where you do it.


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

### 3. Run an experiment

```bash
python run_experiment.py tasks/introspection/gemma.yaml
```

To define your own task look at the existing tasks folder and create a config file with a task description. 
Add or modify any of the existing skills if you want the agent to use specific interp techniques
