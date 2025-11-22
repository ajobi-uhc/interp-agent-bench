# Create tasks and run agents

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


Repo is still WIP