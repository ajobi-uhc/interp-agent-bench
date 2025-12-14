# Seer Documentation

## Overview

Seer is a framework for having agents conduct interpretability work and investigations. The core mechanism involves launching a remote sandbox hosted on a remote GPU or CPU. The agent operates an IPython kernel and notebook on this remote host.

## Why use it?

This approach is valuable because it allows you to see what the agent is doing as it runs, and it can iteratively add things, fix bugs, and adjust its previous work. You can provide tooling to make an environment and any interpretability techniques available as function calls that the agent can use in the notebook as part of writing normal code.

## When to use Seer

- **Exploratory investigations** where you have a hypothesis but want to try many variations quickly
- **Scaling up** measuring how well different interp techniques perform through giving agents controlled access to them
- **Replicating known experiments** on new models — the agent knows the recipe, you just point it at your model
- **Building and improving existing agents** Using seer to build better investigative agents, building better auditing agents etc.

## Example runs

- [Replicate the key experiment in the Anthropic introspection paper on gemma3 27b](https://github.com/ajobi-uhc/seer/blob/main/example_runs/introspection_gemma_27b.ipynb)
- [Investigate a model finetuned with hidden preferences and discover them](https://github.com/ajobi-uhc/seer/blob/main/example_runs/find_hidden_gender_assumption.ipynb)
- [Create a hackable version of Petri for categorizing and finding weird behaviours](https://github.com/ajobi-uhc/seer/tree/main/example_runs/petri-style-transcripts)
- [Use SAE techniques to diff two Gemini checkpoints and discover behavioral differences](https://github.com/ajobi-uhc/seer/blob/main/example_runs/checkpoint_diffing.ipynb)

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

---

# Core Concepts

```
┌──────────────────────────────────────────────────────────────┐
│                         Your Machine                         │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                        Harness                          │ │
│  │  run_agent(prompt, mcp_config, provider="claude")       │ │
│  └───────────────────────────┬─────────────────────────────┘ │
│                              │ MCP                           │
│                              ▼                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                        Session                          │ │
│  │  Notebook: agent works in Jupyter                       │ │
│  │  Local: agent runs locally, calls GPU via RPC           │ │
│  └───────────────────────────┬─────────────────────────────┘ │
└──────────────────────────────┼───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                      Modal (Remote GPU)                      │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                       Sandbox                           │ │
│  │  - GPU (A100, H100, etc.)                               │ │
│  │  - Models (cached on Modal volumes)                     │ │
│  │  - Workspace libraries                                  │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

## Sandbox

GPU environment with models loaded.

```python
sandbox = Sandbox(SandboxConfig(
    gpu="A100",
    models=[ModelConfig(name="google/gemma-2-9b")],
)).start()
```

Two types:
- **Sandbox** — agent has full access
- **ScopedSandbox** — agent can only call functions you expose

## Workspace

Any files/libraries the agent should have in its workspace.

```python
workspace = Workspace(libraries=[
    Library.from_file("steering_hook.py"),
])
```

## Session

How the agent connects to the sandbox.

```python
session = create_notebook_session(sandbox, workspace)  # Access via notebook
# or
session = create_cli_session(workspace, workspace_dir)  # Access via the cli
```

## Harness

Runs the agent.

```python
async for msg in run_agent(prompt, mcp_config=session.mcp_config):
    pass
```

## Putting it together

```python
# 1. Sandbox
config = SandboxConfig(gpu="A100", models=[...])
sandbox = Sandbox(config).start()

# 2. Workspace
workspace = Workspace(libraries=[...])

# 3. Session
session = create_notebook_session(sandbox, workspace)

# 4. Harness
async for msg in run_agent(prompt, mcp_config=session.mcp_config):
    pass

# 5. Cleanup
sandbox.terminate()
```

---

# Environment

An environment is everything your agent needs to do its work: GPU compute, models, packages, files, and tools. Seer environments run on Modal, so you get on-demand GPUs without managing infrastructure.

You define what you need declaratively. Seer handles provisioning, model downloads, and caching.

## Sandbox

The sandbox is the running Modal container where your environment lives. Your agent runs locally and connects to the sandbox to execute code.

```python
config = SandboxConfig(
    gpu="A100",
    models=[ModelConfig(name="google/gemma-2-9b")],
    python_packages=["torch", "transformers"],
)

sandbox = Sandbox(config).start()
# ... agent works ...
sandbox.terminate()
```

## Config options

| Field | What it does |
|-------|--------------|
| `gpu` | GPU type: "A100", "H100", or None for CPU |
| `gpu_count` | Number of GPUs (default: 1) |
| `models` | HuggingFace models to download |
| `python_packages` | pip packages to install |
| `system_packages` | apt packages to install |
| `secrets` | Env vars to pass from local .env |
| `timeout` | Sandbox timeout in seconds (default: 3600) |
| `local_files` | Files to mount: `[("./local.txt", "/sandbox/path.txt")]` |
| `local_dirs` | Directories to mount: `[("./data", "/workspace/data")]` |
| `debug` | Enable VS Code in browser |

## Models

Models are downloaded to Modal volumes and cached across runs:

```python
models=[
    ModelConfig(name="google/gemma-2-9b"),
    ModelConfig(name="my-org/my-adapter", is_peft=True, base_model="meta-llama/Llama-2-7b"),
]
```

| ModelConfig field | What it does |
|-------------------|--------------|
| `name` | HuggingFace model ID |
| `var_name` | Variable name in model info (default: "model") |
| `hidden` | Hide model details from agent |
| `is_peft` | Model is a PEFT/LoRA adapter |
| `base_model` | Base model ID (required if `is_peft=True`) |

## Repos

Clone git repos into the sandbox:

```python
repos=[
    RepoConfig(url="https://github.com/org/repo"),
    RepoConfig(url="org/repo", install="pip install -e ."),
]
```

## Working with a running sandbox

Write files:

```python
sandbox.write_file("/workspace/config.json", '{"key": "value"}')
sandbox.ensure_dir("/workspace/outputs")
```

Run commands:

```python
sandbox.exec("pip install einops")
sandbox.exec_python("print(torch.cuda.is_available())")
```

## Snapshots

Save sandbox state and restore it later:

```python
snapshot = sandbox.snapshot("after setup")

# Later...
new_sandbox = Sandbox.from_snapshot(snapshot, config)
```

Useful for checkpointing long experiments or sharing reproducible starting points.

## Sandbox vs ScopedSandbox

**Sandbox** — agent has full notebook access, can run arbitrary code

**ScopedSandbox** — agent can only call functions you expose via an interface file

```python
# Full access
sandbox = Sandbox(config).start()
session = create_notebook_session(sandbox, workspace)

# Scoped access
scoped = ScopedSandbox(config).start()
model_tools = scoped.serve("interface.py", expose_as="library")
session = create_local_session(workspace, workspace_dir)
```

## Properties

| Property | What it returns |
|----------|-----------------|
| `sandbox.jupyter_url` | Jupyter URL (notebook mode) |
| `sandbox.code_server_url` | VS Code URL (debug mode) |
| `sandbox.model_handles` | Prepared model handles |
| `sandbox.sandbox_id` | Modal sandbox ID |

---

# Scoped Sandbox & RPC

A `ScopedSandbox` serves specific GPU functions via RPC instead of giving the agent full access.

## When to use

- **Sandbox** — agent has full notebook access, good for exploration
- **ScopedSandbox** — agent can only call functions you expose, good for controlled experiments

## Writing interface files

An interface file defines what GPU functions the agent can call.

```python
# interface.py
from transformers import AutoModel, AutoTokenizer
import torch

model_path = get_model_path("google/gemma-2-9b")  # injected
model = AutoModel.from_pretrained(model_path, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)

@expose
def get_embedding(text: str) -> dict:
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    embedding = outputs.hidden_states[-1].mean(dim=1).squeeze()
    return {"embedding": embedding.tolist()}
```

Rules:
- `@expose` marks functions the agent can call
- Must return JSON-serializable types (use `.tolist()` for tensors)
- `get_model_path()` is injected — returns cached model path
- Load models at module level, not inside functions

## Serving the interface

```python
scoped = ScopedSandbox(SandboxConfig(
    gpu="A100",
    models=[ModelConfig(name="google/gemma-2-9b")],
)).start()

model_tools = scoped.serve(
    "interface.py",
    expose_as="library",  # or "mcp"
    name="model_tools"
)
```

`expose_as` options:
- `"library"` — agent imports it: `import model_tools`
- `"mcp"` — agent sees functions as MCP tools

## Using with local session

```python
workspace = Workspace(libraries=[model_tools])
session = create_local_session(workspace, workspace_dir)

async for msg in run_agent(prompt, mcp_config={}):
    pass
```

The agent runs locally. When it calls `model_tools.*`, the call goes to the GPU via RPC.

---

# Sessions

Sessions define how the agent connects to the sandbox.

| Sandbox type | Session type | Agent experience |
|-------------|--------------|------------------|
| `Sandbox` | Notebook | Full Jupyter access on GPU |
| `ScopedSandbox` | Local | Runs locally, calls exposed functions via RPC |

## Notebook session

Agent gets a Jupyter notebook running on the sandbox.

```python
session = create_notebook_session(sandbox, workspace)
```

Returns:
- `session.mcp_config` — pass to `run_agent`
- `session.jupyter_url` — view notebook in browser
- `session.model_info_text` — model details for agent prompt

Use when: exploratory research, iterative probing, visualization.

## Local session

Agent runs on your machine. GPU access is through the functions you exposed.

```python
session = create_local_session(workspace, workspace_dir, name)
```

Returns the same `mcp_config` interface, but execution happens locally.

Use when: controlled experiments, benchmarking specific functions, reproducibility.

Requires `ScopedSandbox` with interface file.

---

# Harness

The harness runs the agent and connects it to a session. Seer provides a default harness, but it's designed to be swapped out. The session provides an mcp config for any harness/agent to connect to.

## Basic usage

```python
async for msg in run_agent(
    prompt=task,
    mcp_config=session.mcp_config,
):
    print(msg)
```

The harness:
1. Connects the agent to the session via MCP
2. Sends the prompt
3. Streams messages back
4. Handles tool calls automatically

## Providers

```python
provider="claude"   # Claude (default)
```

## Interactive mode

Chat with the agent in your terminal. Press ESC to interrupt mid-response.

```python
await run_agent_interactive(
    prompt=prompt,
    mcp_config=session.mcp_config,
    user_message="Start by exploring the model's hidden preferences.",
)
```

## Multi-agent

For multi-agent setups, run multiple agents with different (or the same!) configs:

```python
auditor = run_agent(auditor_prompt, mcp_config=auditor_tools)
investigator = run_agent(investigator_prompt, mcp_config=investigator_tools)
judge = run_agent(judge_prompt, mcp_config={})
```

## Custom harnesses

The harness is just scaffolding around the agent. You can:

- Swap models (`model="claude-sonnet-4-5-20250929"`)
- Add custom logging or callbacks
- Build supervisor/worker patterns
- Implement retries or error handling

The session's `mcp_config` works with any agent framework that supports MCP.

---

# Workspaces

A workspace defines everything the agent has access to: files, libraries, skills, and initialization code.

```python
workspace = Workspace(
    local_dirs=[("./data", "/workspace/data")],
    libraries=[Library.from_file("helpers.py")],
    skill_dirs=["./skills/research"],
    custom_init_code="model = load_my_model()",
)
```

## What you can configure

| Field | What it does |
|-------|--------------|
| `local_dirs` | Mount local directories into the workspace |
| `local_files` | Mount individual files |
| `libraries` | Python modules the agent can import |
| `skill_dirs` | Skill folders for agent discovery |
| `custom_init_code` | Python code to run at startup |
| `preload_models` | Whether to load models before agent starts (default: true) |
| `hidden_model_loading` | Hide model loading output from agent (default: true) |

## Libraries

Make Python files importable by the agent:

```python
workspace = Workspace(libraries=[
    Library.from_file("utils.py"),
    Library.from_skill_dir("skills/steering"),
])
```

When using `ScopedSandbox`, RPC handles are also libraries:

```python
model_tools = scoped.serve("interface.py", expose_as="library")
workspace = Workspace(libraries=[model_tools])
```

Either way, the agent just imports:

```python
from utils import my_helper
import model_tools
```

## Skills

Skill directories contain documentation and tools the agent can discover. Useful for giving the agent reference material or predefined procedures.

```python
workspace = Workspace(skill_dirs=["./skills/activation_patching"])
```

## Custom init code

Run arbitrary Python before the agent starts:

```python
workspace = Workspace(
    custom_init_code="""
from transformers import AutoModel
model = AutoModel.from_pretrained("google/gemma-2-9b")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")
"""
)
```

Variables defined here are available in the agent's namespace.

## Seer toolkit

Common interpretability utilities live in `experiments/toolkit/`:

- `extract_activations.py` — layer activation extraction
- `steering_hook.py` — activation steering via hooks
- `generate_response.py` — text generation helper

```python
toolkit = Path("experiments/toolkit")
workspace = Workspace(libraries=[
    Library.from_file(toolkit / "steering_hook.py"),
    Library.from_file(toolkit / "extract_activations.py"),
])
```

These are meant to be copied and modified.

---

# Experiments

## Experiment 1: Sandbox Intro

Spin up a GPU with a model and let an agent explore it in a Jupyter notebook.

### 1. Configure the sandbox

```python
from src.environment import Sandbox, SandboxConfig, ExecutionMode, ModelConfig

config = SandboxConfig(
    gpu="A100",
    execution_mode=ExecutionMode.NOTEBOOK,
    models=[ModelConfig(name="google/gemma-2-2b-it")],
    python_packages=["torch", "transformers", "accelerate"],
)
```

- `gpu` — A100 has 40GB VRAM, fits models up to ~30B params
- `execution_mode` — NOTEBOOK means agent works in Jupyter on the GPU
- `models` — HuggingFace model IDs to download and load
- `python_packages` — installed in the sandbox

### 2. Start the sandbox

```python
sandbox = Sandbox(config).start()
```

Provisions the GPU on Modal. First run downloads the model (~2 min), subsequent runs use cache.

### 3. Create a workspace

```python
from src.workspace import Workspace

workspace = Workspace(libraries=[])
```

Workspace defines custom code the agent can import. Empty for now — later examples add interpretability tools here.

### 4. Create a session

```python
from src.execution import create_notebook_session

session = create_notebook_session(sandbox, workspace)
```

Returns:
- `session.mcp_config` — config for agent to connect to the notebook
- `session.jupyter_url` — open this to watch the agent work
- `session.model_info_text` — model details to include in agent prompt

### 5. Run the agent

```python
from src.harness import run_agent

task = (example_dir / "task.md").read_text()
prompt = f"{session.model_info_text}\n\n{task}"

async for msg in run_agent(
    prompt=prompt,
    mcp_config=session.mcp_config,
    provider="claude"
):
    pass

sandbox.terminate()
```

The notebook saves to `./outputs/` as the agent works.

### Full example

```bash
cd experiments/sandbox-intro && python main.py
```

---

## Experiment 2: Scoped Sandbox

Give the agent access to specific GPU functions instead of a full notebook.

### When to use this

- **Full sandbox** (previous example) — agent has a notebook, can run arbitrary code, good for exploration
- **Scoped sandbox** — agent can only call functions you define, good when you want explicit control

### 1. Configure the scoped sandbox

```python
from src.environment import ScopedSandbox, SandboxConfig, ModelConfig

scoped = ScopedSandbox(SandboxConfig(
    gpu="A100",
    models=[ModelConfig(name="google/gemma-2-9b")],
    python_packages=["torch", "transformers", "accelerate"],
))

scoped.start()
```

No `execution_mode` — the agent doesn't run in the sandbox. Instead, you serve specific functions from it.

### 2. Define GPU functions

Create an interface file with functions that run on the GPU:

```python
# interface.py
from transformers import AutoModel, AutoTokenizer
import torch

model_path = get_model_path("google/gemma-2-9b")  # injected by RPC server
model = AutoModel.from_pretrained(model_path, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)

@expose
def get_model_info() -> dict:
    """Get basic model information."""
    return {
        "num_layers": model.config.num_hidden_layers,
        "hidden_size": model.config.hidden_size,
        "vocab_size": model.config.vocab_size,
    }

@expose
def get_embedding(text: str) -> dict:
    """Get text embedding from model."""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    embedding = outputs.hidden_states[-1].mean(dim=1).squeeze()
    return {"embedding": embedding.tolist()}
```

- `@expose` marks functions the agent can call — everything else is hidden
- Functions must return JSON-serializable types (use `.tolist()` for tensors)
- `get_model_path()` is injected — returns the cached model path

### 3. Serve the interface

```python
model_tools = scoped.serve(
    str(example_dir / "interface.py"),
    expose_as="library",
    name="model_tools"
)
```

Loads `interface.py` on the GPU and creates an RPC server.

`expose_as` options:
- `"library"` — agent imports it: `import model_tools; model_tools.get_embedding("hello")`
- `"mcp"` — agent sees them as MCP tools

### Full example

```bash
cd experiments/scoped-sandbox-intro && python main.py
```

---

## Experiment 3: Hidden Preference Investigation

Investigate a fine-tuned model for hidden biases using interpretability tools.

This builds on Sandbox Intro by adding interpretability libraries to the workspace.

### 1. Configure with PEFT model

```python
from src.environment import Sandbox, SandboxConfig, ExecutionMode, ModelConfig

config = SandboxConfig(
    gpu="A100",
    execution_mode=ExecutionMode.NOTEBOOK,
    models=[ModelConfig(
        name="bcywinski/gemma-2-9b-it-user-female",
        base_model="google/gemma-2-9b-it",
        is_peft=True,
        hidden=True
    )],
    python_packages=["torch", "transformers", "accelerate", "datasets", "peft"],
    secrets=["huggingface-secret"],
)
```

New `ModelConfig` parameters:
- `base_model` — base model to load first
- `is_peft=True` — this is a PEFT adapter (LoRA, etc.), not a full model
- `hidden=True` — hides model name from agent to prevent bias in investigation

### 2. Add interpretability libraries

```python
from src.workspace import Workspace, Library

toolkit = Path(__file__).parent.parent / "toolkit"

workspace = Workspace(libraries=[
    Library.from_file(toolkit / "steering_hook.py"),
    Library.from_file(toolkit / "extract_activations.py"),
])
```

These are in `experiments/toolkit/`:

- `extract_activations.py` — extract activations at any layer/position
- `steering_hook.py` — inject vectors during generation

The agent can then:

```python
from extract_activations import extract_activation
from steering_hook import create_steering_hook

# Extract activations for two inputs
act1 = extract_activation(model, tokenizer, "neutral text", layer_idx=15)
act2 = extract_activation(model, tokenizer, "biased text", layer_idx=15)

# Compute steering vector
steering_vec = act2 - act1

# Test if it causally affects behavior
with create_steering_hook(model, layer_idx=15, vector=steering_vec, strength=2.0):
    output = model.generate(...)
```

### Full example

```bash
cd experiments/hidden-preference-investigation && python main.py
```

---

## Experiment 4: Introspection

Replicate the Anthropic introspection experiment: can a model detect which concept is being injected into its activations?

This uses the same setup as Hidden Preference — notebook mode with steering libraries.

### The experiment

1. Extract concept vectors (e.g., "Lightning", "Oceans", "Happiness") by computing `activation(concept) - mean(activation(baselines))`
2. Inject these vectors during generation while asking the model "Do you detect an injected thought? What is it about?"
3. Score whether the model correctly identifies the injected concept
4. Compare against control trials (no injection) to establish baseline

### Setup

```python
config = SandboxConfig(
    gpu="H100",  # Larger model needs more VRAM
    execution_mode=ExecutionMode.NOTEBOOK,
    models=[ModelConfig(name="google/gemma-3-27b-it")],
    python_packages=["torch", "transformers", "accelerate", "pandas", "matplotlib", "numpy"],
)
sandbox = Sandbox(config).start()

workspace = Workspace(libraries=[
    Library.from_file(shared_libs / "steering_hook.py"),
    Library.from_file(shared_libs / "extract_activations.py"),
])

session = create_notebook_session(sandbox, workspace)
```

### What the agent does

The task prompt guides the agent through:

1. Extracting concept vectors at ~70% model depth
2. Verifying steering works on neutral prompts
3. Running injection trials with the introspection prompt
4. Running control trials without injection
5. Computing identification rates and comparing against baseline

### Full example

```bash
cd experiments/introspection && python main.py
```

---

## Experiment 5: Checkpoint Diffing

Compare two model checkpoints (Gemini 2.0 vs 2.5 Flash) using SAE-based analysis to find behavioral differences.

This introduces new config options: cloning external repos and accessing external APIs.

### New concepts

#### Cloning external repos

```python
from src.environment import RepoConfig

config = SandboxConfig(
    repos=[RepoConfig(url="nickjiang2378/interp_embed")],
    # ...
)
```

The repo is cloned to `/workspace/interp_embed` in the sandbox. The agent can import from it.

#### External API access

```python
config = SandboxConfig(
    secrets=["GEMINI_API_KEY", "OPENAI_KEY", "OPENROUTER_API_KEY", "HF_TOKEN"],
    # ...
)
```

Secrets are Modal secrets you've configured. They're available as environment variables in the sandbox.

#### Longer timeout

```python
config = SandboxConfig(
    timeout=7200,  # 2 hours (default is 1 hour)
    # ...
)
```

SAE encoding is slow — this experiment can take 1-2 hours.

### What the agent does

1. Generate prompts designed to reveal behavioral differences
2. Collect responses from both Gemini versions via OpenRouter
3. Encode responses using SAE (Llama 3.1 8B SAE with 65k features)
4. Diff feature activations to find what changed between versions
5. Analyze top differentiating features with examples

### Full example

```bash
cd experiments/checkpoint-diffing && python main.py
```

---

## Experiment 6: Petri-Style Harness

A hackable version of Petri for categorizing and finding weird behaviors in models.

This shows how to build multi-agent auditing pipelines with Seer.

### Architecture

```
Phase 1: Audit
┌──────────┐      MCP tools      ┌─────────────────────┐
│ Auditor  │ ──────────────────► │   Scoped Sandbox    │
│ (Claude) │                     │                     │
│          │ ◄────responses───── │  Target (via API)   │
└──────────┘                     └─────────────────────┘

Phase 2: Judge
┌──────────┐
│  Judge   │ ◄── transcript retrieved from sandbox
│ (Claude) │
└──────────┘
     │
     ▼
   scores
```

1. **Auditor** probes the Target via MCP tools exposed from the sandbox
2. **Transcript** is retrieved after the audit completes
3. **Judge** scores the transcript on multiple dimensions

### New concepts

#### Scoped sandbox exposing MCP tools

Use `expose_as="mcp"` so the agent gets tools instead of importable functions:

```python
scoped = ScopedSandbox(SandboxConfig(
    gpu=None,  # No GPU — using OpenRouter API
    python_packages=["openai"],
    secrets=["OPENROUTER_API_KEY"],
))
scoped.start()

mcp_config = scoped.serve(
    "conversation_interface.py",
    expose_as="mcp",
    name="petri_tools"
)
```

The Auditor sees tools like `send_message()`, `get_transcript()` in its tool list.

#### No GPU

The Target model runs via API, so no GPU needed:

```python
SandboxConfig(gpu=None, ...)
```

#### Sequential agents

```python
# Phase 1: Auditor uses MCP tools to probe Target
async for msg in run_agent(auditor_prompt, mcp_config=mcp_config):
    pass

# Phase 2: Retrieve transcript
transcript = scoped.exec("cat /tmp/petri_transcript.txt")

# Phase 3: Judge scores (simple API call, no tools)
judge_response = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    messages=[{"role": "user", "content": build_judge_prompt(transcript)}],
)
```

### Conversation interface

`conversation_interface.py` exposes these MCP tools:

- `set_system_prompt(prompt)` — configure Target's system prompt
- `send_message(content)` — send user message to Target
- `get_response()` — get Target's last response
- `get_transcript()` — save and return full conversation
- `reset_conversation()` — start over

### Full example

```bash
cd experiments/petri-style-harness && python main.py
```

---

# Additional Resources

- GitHub: https://github.com/ajobi-uhc/seer
- Example Notebooks: https://github.com/ajobi-uhc/seer/tree/main/example_runs
- Modal: https://modal.com
- Documentation: https://ajobi-uhc.github.io/seer/
