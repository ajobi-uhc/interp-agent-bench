# Core Concepts

Seer has three main components that compose together: **Sandbox**, **Session**, and **Harness**.

## Sandbox: GPU Environment

- **What**: A GPU container with models and packages loaded
- **Why**: You need GPU compute to run large models, but want the environment to be reproducible

The `Sandbox` handles provisioning compute, downloading models to persistent volumes, and installing Python packages. It abstracts away the infrastructure (Modal, RunPod, etc.) so you can focus on the experiment.

```python
config = SandboxConfig(
    gpu="A100",
    models=[ModelConfig(name="google/gemma-2-2b-it")],
    python_packages=["torch", "transformers", "accelerate"]
)
sandbox = Sandbox(config).start()
```

When you call `.start()`, Modal provisions a GPU container, downloads models (cached for future runs), installs packages, and returns a running environment. The sandbox stays alive until you `.terminate()` it.

For more control, use `ScopedSandbox` to serve specific functions from the GPU environment via RPC.

## Session: How Agents Interact

- **What**: The interface between your agent and the GPU environment
- **Why**: Different experiments need different interaction patterns

There are three session modes:

### Notebook Mode
Agent gets a Jupyter notebook with full GPU access. Best for exploratory research where you want the agent to iteratively probe a model.

```python
session = create_notebook_session(sandbox, workspace)
# Agent can execute cells, load models, run experiments
# Results: notebook saved to disk + Jupyter URL
```

**Use when**: You want the agent to explore freely, try different approaches, and document findings

### Local Mode
Agent runs Python locally but calls GPU functions via RPC. Best for structured experiments with pre-defined model interfaces.

```python
session = create_local_session(workspace, workspace_dir, name)
# Agent has local Python workspace + can call remote GPU functions
# Results: Python files + logs
```

**Use when**: You want tight control over what GPU functions are exposed, and the agent workflow is predictable

### Scoped Mode
(Deprecated in favor of Local mode with ScopedSandbox)

## Harness: Agent Orchestration

- **What**: Runs the agent and connects it to the session
- **Why**: You need to pass tasks to the agent, stream results, and handle the MCP connection

The harness is intentionally thin. It takes a prompt and MCP config, runs the agent, and streams back messages.

```python
async for msg in run_agent(
    prompt=task,
    mcp_config=session.mcp_config,
    provider="claude"  # or "openai"
):
    print(msg)  # Stream agent's progress
```

The harness handles:
- Connecting agent to session via MCP
- Streaming agent messages
- Error handling and retries

## How They Compose

Every experiment follows the same pattern:

```python
# 1. Define environment
config = SandboxConfig(gpu="A100", models=[...], packages=[...])
sandbox = Sandbox(config).start()

# 2. Create workspace and session
workspace = Workspace(libraries=[...])
session = create_notebook_session(sandbox, workspace)

# 3. Run agent with task
task = "Investigate whether this model has a hidden preference..."
async for msg in run_agent(prompt=task, mcp_config=session.mcp_config):
    pass

# 4. Cleanup
sandbox.terminate()
```

That's it. The components are designed to be simple and composable so you can quickly iterate on experiments.
