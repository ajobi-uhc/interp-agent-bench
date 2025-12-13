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

See [Scoped Sandbox](scoped-sandbox.md) for the scoped pattern.

## Properties

| Property | What it returns |
|----------|-----------------|
| `sandbox.jupyter_url` | Jupyter URL (notebook mode) |
| `sandbox.code_server_url` | VS Code URL (debug mode) |
| `sandbox.model_handles` | Prepared model handles |
| `sandbox.sandbox_id` | Modal sandbox ID |