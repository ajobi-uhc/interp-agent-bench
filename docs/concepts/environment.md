# Environment

The sandbox is a Modal container. You specify what goes in it.

## Config

```python
config = SandboxConfig(
    gpu="A100",                                      # GPU type
    execution_mode=ExecutionMode.NOTEBOOK,          # How agent connects
    models=[ModelConfig(name="google/gemma-2-9b")], # Models to download
    python_packages=["torch", "transformers"],      # Pip installs
    secrets=["huggingface-secret"]                  # API keys
)
```

## What Happens

1. Modal spins up container with specified GPU
2. Models download from HuggingFace (cached on Modal volumes)
3. Python packages install
4. Sandbox starts, returns `sandbox` object
5. You can now create sessions from it

## Sandbox vs ScopedSandbox

### Sandbox (Full Access)

Agent has full access to the GPU environment. Use with Notebook or CLI mode.

```python
sandbox = Sandbox(config).start()
session = create_notebook_session(sandbox, workspace)
# Agent can run arbitrary Python on GPU
```

### ScopedSandbox (RPC Only)

Agent runs locally, GPU only exposes specific functions via RPC. Use with Local mode.

```python
scoped = ScopedSandbox(config).start()

# Serve interface.py - functions marked with @expose become RPC-callable
model_tools = scoped.serve(
    "interface.py",
    expose_as="library",
    name="model_tools"
)

# Agent runs locally, calls GPU functions remotely
session = create_local_session(workspace, workspace_dir)
```

The interface file defines what GPU functions are available:

```python
# interface.py
model = AutoModel.from_pretrained(get_model_path("google/gemma-2-9b"))

@expose
def get_embedding(text: str) -> dict:
    # Runs on GPU, returns JSON to agent
    ...
```

See [Scoped Sandbox Guide](scoped-sandbox.md) for writing interface files.

## Cleanup

```python
sandbox.terminate()  # Shuts down Modal container
```
