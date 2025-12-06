# Sena - Sandboxed Environments for Interpretability Research

## Overview

Sena provides GPU-accelerated sandboxed environments for running interpretability experiments. You can set up remote environments with models pre-loaded, connect to Jupyter notebooks, and run experiments interactively.

## When to Use This Skill

Use this skill when the user wants to:
- Run interpretability experiments on models (steering, activation analysis, etc.)
- Work with models that need GPU acceleration (A100, H100, etc.)
- Set up reproducible experiment environments
- Run code in isolated sandboxes with specific dependencies

## Key Concepts

### 1. Sandboxes
Sandboxes are isolated execution environments running on GPU clusters (via Modal). They can:
- Load specific models (e.g., "google/gemma-2-9b-it")
- Install Python packages
- Run Jupyter notebooks or CLI sessions

### 2. Sessions
Sessions are execution contexts within sandboxes:
- **Notebook sessions**: Jupyter notebook with pre-loaded models
- **CLI sessions**: Command-line execution environment
- **Local sessions**: Run locally without GPU

### 3. Workspaces
Workspaces organize libraries and tools:
- **Libraries**: Python code files made available in the environment
- **Skills**: Reusable interpretability techniques

## How to Use Sena

### Step 1: Set Up a Sandbox

Write a Python script to create the sandbox:

```python
from src.environment import Sandbox, SandboxConfig, ExecutionMode, ModelConfig
from src.workspace import Workspace, Library
from src.execution import create_notebook_session
import json

# Configure the sandbox
config = SandboxConfig(
    execution_mode=ExecutionMode.NOTEBOOK,
    gpu="A100",  # or "H100", "T4", etc.
    models=[
        ModelConfig(name="google/gemma-2-9b-it")
    ],
    python_packages=["torch", "transformers", "accelerate"],
)

# Start the sandbox (this provisions GPU resources)
sandbox = Sandbox(config).start()

# Optional: Create a workspace with libraries
workspace = Workspace(
    libraries=[
        # Library.from_file("path/to/helper.py")
    ]
)

# Create a notebook session
session = create_notebook_session(sandbox, workspace)

# Output connection info
print(json.dumps({
    "session_id": session.session_id,
    "jupyter_url": session.jupyter_url,
}))
```

### Step 2: Run the Setup Script

Save the script to a file and run it:

```bash
uv run python setup_sandbox.py
```

This will:
1. Provision GPU resources on Modal
2. Load the specified models
3. Start a Jupyter server
4. Return connection info as JSON

### Step 3: Connect to the Session

Once you have the `session_id` and `jupyter_url`, use the scribe MCP server to connect:

```python
# Call the attach_to_session tool (available via scribe MCP)
attach_to_session(
    session_id="<session_id from output>",
    jupyter_url="<jupyter_url from output>"
)
```

After attaching, you can use these tools:
- `execute_code(session_id, code)` - Run Python in the notebook
- `add_markdown(session_id, content)` - Add documentation
- `edit_cell(session_id, code, cell_index)` - Edit existing cells
- `get_notebook_outputs()` - View execution results

### Step 4: Run Your Experiment

Now you can execute interpretability code in the sandbox:

```python
# Example: Load the model and extract activations
execute_code(session_id, """
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Model is already loaded by sandbox at:
import os
model_path = os.environ['MODEL_GOOGLE_GEMMA_2_9B_IT_PATH']

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

print(f"Model loaded: {model_path}")
print(f"Device: {model.device}")
""")
```

## Advanced Usage

### Using Libraries

If you have helper code, include it in the workspace:

```python
from src.workspace import Library

workspace = Workspace(
    libraries=[
        Library.from_file("experiments/libraries/steering_hook.py"),
        Library.from_file("experiments/libraries/extract_activations.py"),
    ]
)
```

These libraries will be importable in your notebook:

```python
execute_code(session_id, """
from steering_hook import add_steering_vector
# Use the library functions...
""")
```

### Multiple Models

Load multiple models in the same sandbox:

```python
config = SandboxConfig(
    gpu="A100",
    models=[
        ModelConfig(name="google/gemma-2-9b-it"),
        ModelConfig(name="meta-llama/Llama-2-7b-hf"),
    ],
)
```

Each model is available via environment variables:
- `MODEL_GOOGLE_GEMMA_2_9B_IT_PATH`
- `MODEL_META_LLAMA_LLAMA_2_7B_HF_PATH`

### Using Secrets

Pass API keys and secrets securely:

```python
config = SandboxConfig(
    gpu="A100",
    secrets=["HUGGINGFACE_TOKEN", "OPENAI_API_KEY"],
)
```

These are available as environment variables in the sandbox.

## Common Patterns

### Pattern 1: Quick Experiment

```python
# 1. Write setup script inline
setup_code = '''
from src.environment import Sandbox, SandboxConfig, ExecutionMode, ModelConfig
from src.execution import create_notebook_session
import json

config = SandboxConfig(
    execution_mode=ExecutionMode.NOTEBOOK,
    gpu="A100",
    models=[ModelConfig(name="google/gemma-2-9b-it")],
    python_packages=["torch", "transformers"],
)

sandbox = Sandbox(config).start()
session = create_notebook_session(sandbox, Workspace())

print(json.dumps({
    "session_id": session.session_id,
    "jupyter_url": session.jupyter_url,
}))
'''

# 2. Run it
# (save to file and execute with bash)

# 3. Attach and work
# attach_to_session(...) then execute_code(...)
```

### Pattern 2: Replicate Paper Experiments

For replicating published experiments (e.g., Anthropic's introspection paper):

```python
# 1. Set up sandbox with paper's model
config = SandboxConfig(
    gpu="A100",
    models=[ModelConfig(name="<model from paper>")],
    python_packages=[
        # Dependencies from paper
    ],
)

# 2. Load paper's methodology as libraries
workspace = Workspace(
    libraries=[
        Library.from_file("experiments/libraries/paper_method.py")
    ]
)

# 3. Run experiments in notebook
```

## Troubleshooting

### Models Not Loading
- Ensure you have HuggingFace credentials configured
- Check that model names are exact (e.g., "google/gemma-2-9b-it")
- Verify GPU has enough memory for the model

### Connection Issues
- Check that `jupyter_url` is accessible
- Ensure firewall allows connections to Modal
- Verify `session_id` matches the created session

### Package Installation
- Add packages to `python_packages` in SandboxConfig
- Some packages may need system dependencies - use Modal's container config

## Important Notes

1. **GPU Costs**: Sandboxes use real GPU resources and incur costs. Clean up when done.
2. **Model Loading**: First run takes longer as models download. Subsequent runs are faster.
3. **Session State**: Notebook state persists across `execute_code()` calls - variables are retained.
4. **Cleanup**: Sandboxes remain active until explicitly terminated. Use `sandbox.terminate()`.

## Complete Example

Here's a full workflow:

```python
# save_as: setup_gemma_experiment.py
from src.environment import Sandbox, SandboxConfig, ExecutionMode, ModelConfig
from src.workspace import Workspace, Library
from src.execution import create_notebook_session
import json

config = SandboxConfig(
    execution_mode=ExecutionMode.NOTEBOOK,
    gpu="A100",
    models=[ModelConfig(name="google/gemma-2-9b-it")],
    python_packages=["torch", "transformers", "accelerate", "matplotlib"],
)

sandbox = Sandbox(config).start()

workspace = Workspace(
    libraries=[
        Library.from_file("experiments/libraries/steering_hook.py")
    ]
)

session = create_notebook_session(sandbox, workspace)

print(json.dumps({
    "session_id": session.session_id,
    "jupyter_url": session.jupyter_url,
}))
```

Then run:
```bash
uv run python setup_gemma_experiment.py
```

Parse output and attach:
```python
# From the JSON output:
attach_to_session(
    session_id="<id>",
    jupyter_url="<url>"
)

# Now run your experiment:
execute_code(session_id, """
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = os.environ['MODEL_GOOGLE_GEMMA_2_9B_IT_PATH']
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Your interpretability work here...
""")
```

## API Reference

### SandboxConfig
- `execution_mode`: ExecutionMode.NOTEBOOK or ExecutionMode.CLI
- `gpu`: "A100", "H100", "T4", or None (CPU)
- `models`: List of ModelConfig objects
- `python_packages`: List of package names to install
- `secrets`: List of secret names from Modal
- `env`: Dict of environment variables

### ModelConfig
- `name`: Model identifier (e.g., "google/gemma-2-9b-it")
- `revision`: Optional git revision/branch
- `cache_dir`: Optional custom cache directory

### Workspace
- `libraries`: List of Library objects (Python code)
- `skills`: List of Skill objects (reusable techniques)

### Session Methods
- `session.session_id`: Unique identifier for this session
- `session.jupyter_url`: URL to connect to Jupyter
- `session.mcp_config`: MCP configuration dict

## Tips for Success

1. **Start Small**: Begin with a single model and basic packages
2. **Test Locally**: Use ExecutionMode.CLI or local sessions for quick iteration
3. **Reuse Sessions**: Don't create new sandboxes unnecessarily - attach to existing ones
4. **Document**: Use `add_markdown()` to document your experiment as you go
5. **Check Outputs**: Always examine execution results before proceeding

---

This skill enables powerful interpretability research workflows. Ask questions if you need clarification on any aspect!
