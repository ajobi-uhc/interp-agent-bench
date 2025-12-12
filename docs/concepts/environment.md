# Environment

The sandbox is a Modal container with GPU, models, and packages.

## Config

```python
config = SandboxConfig(
    gpu="A100",                                      # GPU type
    execution_mode=ExecutionMode.NOTEBOOK,          # NOTEBOOK or CLI
    models=[ModelConfig(name="google/gemma-2-9b")], # HuggingFace model IDs
    python_packages=["torch", "transformers"],      # pip packages
    secrets=["huggingface-secret"],                 # Modal secrets
)
```

## Sandbox vs ScopedSandbox

**Sandbox** — agent has full access, use with notebook mode:

```python
sandbox = Sandbox(config).start()
session = create_notebook_session(sandbox, workspace)
```

**ScopedSandbox** — agent can only call exposed functions, use with local mode:

```python
scoped = ScopedSandbox(config).start()
model_tools = scoped.serve("interface.py", expose_as="library")
session = create_local_session(workspace, workspace_dir)
```

See [Scoped Sandbox](scoped-sandbox.md) for writing interface files.

## Lifecycle

```python
sandbox = Sandbox(config).start()  # Provisions GPU, downloads models
# ... run experiments ...
sandbox.terminate()                 # Shuts down container
```

First run downloads models (~2 min). Subsequent runs use cached models on Modal volumes.
