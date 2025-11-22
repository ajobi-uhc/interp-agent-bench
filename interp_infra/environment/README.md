# Environment Stage

**Stage 1 of 3-stage infrastructure design**

The Environment stage handles GPU infrastructure setup: creating sandboxes, downloading models, cloning repos, and starting Jupyter.

## Purpose

Prepares the physical/virtual infrastructure needed for running interpretability experiments. This stage is **infrastructure-only** - it doesn't interact with the agent or load models into memory yet.

## What It Does

1. **Builds Docker Image** - Creates Modal image with dependencies
2. **Creates GPU Sandbox** - Provisions GPU instance with specified hardware
3. **Downloads Models** - Pre-downloads model weights to HF cache or Modal volumes
4. **Clones Repositories** - Checks out any GitHub repos needed for the experiment
5. **Starts Jupyter** - Launches Jupyter server with public HTTPS tunnel

## Inputs

### Required
- `config: ExperimentConfig` - Complete experiment configuration

### Config Fields Used
```python
config.name: str                    # Experiment name
config.gpu: GPUConfig              # GPU type, count, volume settings
config.image: ImageConfig          # Docker image build config
config.models: List[ModelConfig]   # Models to pre-download (not loaded yet!)
config.github_repos: List[str]     # Repos to clone
```

### Example Config
```yaml
name: "my-experiment"
gpu:
  gpu_type: "A100-80GB"
  gpu_count: 2
  use_model_volumes: true
models:
  - name: "google/gemma-2-9b-it"
    device: "auto"
    dtype: "bfloat16"
github_repos:
  - "myorg/my-research-code"
```

## Outputs

Returns `EnvironmentHandle` with connection info:

```python
@dataclass
class EnvironmentHandle:
    sandbox_id: str           # Modal sandbox ID
    jupyter_url: str          # Public HTTPS URL (e.g., https://xxx.modal-proxy.com)
    jupyter_port: int         # Port (typically 8888)
    jupyter_token: str        # Jupyter authentication token
    status: str               # Sandbox status
    sandbox: modal.Sandbox    # Raw Modal sandbox object
    experiment_config: ExperimentConfig  # Original config
    _client: ModalEnvironment # Internal client reference
```

### What the Agent Needs

**The agent only needs the `jupyter_url`** to connect via MCP:

```python
mcp_servers = {
    "notebooks": {
        "type": "stdio",
        "command": "python",
        "args": ["-m", "scribe.notebook.notebook_mcp_server"],
        "env": {
            "SCRIBE_URL": env_handle.jupyter_url,  # ← This is all it needs!
        }
    }
}
```

## Usage

```python
from interp_infra.environment import setup_environment
from interp_infra.config.parser import load_config

# Load config
config = load_config("tasks/my-experiment/config.yaml")

# Stage 1: Setup infrastructure
env_handle = setup_environment(config)

print(f"Sandbox ID: {env_handle.sandbox_id}")
print(f"Jupyter URL: {env_handle.jupyter_url}")
# → Now ready for Stage 2 (Execution)
```

## Implementation Files

- `modal.py` - Modal sandbox creation and management
- `image.py` - Docker image building
- `__init__.py` - Public API (`setup_environment()`)

## Notes

- **Models are downloaded but NOT loaded** - They sit in cache/volumes
- **Jupyter is running but empty** - No kernel, no session yet
- **This stage is stateless** - Just provisions infrastructure
- **Cleanup**: Call `env_handle._client.terminate_sandbox(sandbox_id)` to tear down

## Next Stage

After environment setup, proceed to **Stage 2: Execution** to create interaction interface (notebook session, filesystem access, or MCP tools).
