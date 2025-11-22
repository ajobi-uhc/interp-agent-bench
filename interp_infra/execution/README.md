# Execution Stage

**Stage 2 of 3-stage infrastructure design**

The Execution stage determines **how the agent interacts** with the environment: notebook sessions, filesystem access, or constrained MCP tools.

## Purpose

Creates the execution interface for the agent. This stage **loads models into memory** and configures the interaction mode.

## Execution Modes

### 1. Notebook Mode (✅ Implemented)
- **What**: Full Jupyter notebook access via Scribe MCP server
- **Use case**: Unrestricted experimentation, iterative development
- **Agent tools**: Execute code, edit cells, view outputs
- **Models**: Loaded into kernel memory before agent starts (warm start)

### 2. Filesystem Mode (📝 Placeholder)
- **What**: Direct file system access, no notebook
- **Use case**: File-based experiments, reading/writing scripts
- **Agent tools**: Read/write files, run Python scripts
- **Models**: Would be loaded on-demand

### 3. MCP Mode (📝 Placeholder)
- **What**: Constrained tool set via custom MCP server
- **Use case**: Limited, controlled interactions
- **Agent tools**: Predefined functions only
- **Models**: Exposed via function calls

## Notebook Mode (Current Implementation)

### Inputs

```python
env_handle: EnvironmentHandle  # From Stage 1
config: ExperimentConfig       # Original config
```

### What It Does

1. **Connects to Jupyter** - Uses `jupyter_url` from environment
2. **Creates Kernel** - Starts new Python kernel
3. **Loads Models** - Runs `kernel_setup.py` to load models into GPU memory
4. **Loads Skills** - Injects interpretability technique functions
5. **Creates Session** - Returns session ID for agent to attach

### Model Loading (Kernel Warmup)

**Critical**: Models are loaded **before** the agent starts, so the agent attaches to a pre-warmed session:

```python
# This runs INSIDE the kernel (on the Modal GPU)
init_code = """
import sys
sys.path.insert(0, "/root")
from interp_infra.config.schema import ExperimentConfig
from interp_infra.execution.kernel_setup import create_namespace

# Load models and skills
namespace = create_namespace(config)
globals().update(namespace)
# Now `model`, `tokenizer`, and skill functions are in global scope
"""
```

### Outputs

Returns `NotebookHandle`:

```python
@dataclass
class NotebookHandle:
    session_id: str           # Jupyter session ID (UUID)
    jupyter_url: str          # Same as env_handle.jupyter_url
    kernel_id: str           # Kernel ID
    status: str              # "ready"
```

### What the Agent Needs

**The agent only needs the `session_id`** to attach:

```python
# Agent connects via MCP
await client.query(f"""
Please attach to the existing Jupyter session using:

attach_to_session(session_id='{exec_handle.session_id}')

The model and tokenizer are already loaded in the session.
""")
```

## Usage Example

```python
from interp_infra.environment import setup_environment
from interp_infra.execution import setup_execution
from interp_infra.config.parser import load_config

# Load config
config = load_config("tasks/my-experiment/config.yaml")

# Stage 1: Infrastructure
env_handle = setup_environment(config)

# Stage 2: Execution interface
exec_handle = setup_execution(env_handle, config)

print(f"Session ID: {exec_handle.session_id}")
# → Models are loaded, agent can now attach
```

## Execution Config

Control which mode via config:

```yaml
execution:
  type: "notebook"  # or "filesystem", "mcp"
```

Defaults to `"notebook"` if not specified.

## Implementation Files

- `notebook.py` - Notebook session setup and kernel initialization
- `kernel_setup.py` - Model/skill loading (runs inside kernel)
- `filesystem.py` - Placeholder for filesystem mode
- `mcp.py` - Placeholder for MCP-only mode
- `__init__.py` - Public API (`setup_execution()`)

## Kernel Setup Details

The `kernel_setup.py` module runs **inside the Jupyter kernel** (not the parent process):

1. **Load Models**: Downloads → GPU memory (via transformers)
2. **Load Skills**: Parse markdown → extract functions → inject into namespace
3. **Validate**: Check that models are on correct device
4. **Return Namespace**: `{model, tokenizer, skill_functions, workspace_path, ...}`

### What Gets Loaded

```python
namespace = {
    "model": <AutoModelForCausalLM>,      # On GPU
    "tokenizer": <AutoTokenizer>,
    "workspace": "/workspace",
    "experiment_name": "my-experiment",
    # + any skill functions (e.g., `extract_activations()`)
}
```

## Notes

- **Warm Start**: Models load during Stage 2, not on-demand
- **Session Persistence**: Session stays alive until explicitly shut down
- **Multiple Sessions**: Could create multiple sessions for multi-agent scenarios
- **Skills**: Interpreted from markdown, injected as Python functions

## Next Stage

After execution setup, proceed to **Stage 3: Harness** to orchestrate the agent(s).
