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

**Sandbox:** Full environment, agent has broad access
**ScopedSandbox:** Limited environment for RPC, agent calls functions remotely

```python
# Full sandbox
sandbox = Sandbox(config).start()

# Scoped sandbox (RPC pattern)
scoped = ScopedSandbox(config).start()
model_tools = scoped.serve("interface.py", expose_as="library")
```

## Cleanup

```python
sandbox.terminate()  # Shuts down Modal container
```
