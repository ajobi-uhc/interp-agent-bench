# Notebook Intro

**Goal**: Learn the basics of working with models in a GPU sandbox through a Jupyter notebook

This is the simplest experiment. It spins up a GPU sandbox, loads a small model (Gemma 2B), and gives an agent a Jupyter notebook to explore the model.

## What This Demonstrates

- How to configure a GPU sandbox with models and packages
- How notebook execution mode works
- How the agent interacts with models through Python code
- Basic model exploration: architecture, generation, tokenization

## The Code

```python
# experiments/notebook-intro/main.py
config = SandboxConfig(
    gpu="A100",
    execution_mode=ExecutionMode.NOTEBOOK,
    models=[ModelConfig(name="google/gemma-2-2b-it")],
    python_packages=["torch", "transformers", "accelerate"],
)
sandbox = Sandbox(config).start()

workspace = Workspace(libraries=[])  # No custom libraries needed
session = create_notebook_session(sandbox, workspace)

task = (example_dir / "task.md").read_text()  # Load the task prompt
prompt = f"{session.model_info_text}\n\n{task}"

async for msg in run_agent(prompt=prompt, mcp_config=session.mcp_config, provider="claude"):
    pass  # Stream agent's progress

print(f"\n✓ Jupyter: {session.jupyter_url}")
```

## The Task

The agent is asked to:

1. **Load and inspect the model**: Explore architecture, parameter count, device placement
2. **Generate text**: Try different prompts and observe behavior
3. **Explore tokenization**: Tokenize strings, check token IDs, try edge cases

See `experiments/notebook-intro/task.md` for the full prompt.

## Running It

```bash
cd experiments/notebook-intro
python main.py
```

**What happens:**

1. Modal provisions an A100 GPU (~30 seconds)
2. Downloads `google/gemma-2-2b-it` to Modal volume (first run only, ~2GB)
3. Installs torch, transformers, accelerate
4. Starts Jupyter server on the GPU
5. Agent connects and runs the exploration task
6. Notebook saved to `./outputs/` and Jupyter URL printed

**Expected output:**

```
→ Starting Modal sandbox...
→ Loading model google/gemma-2-2b-it...
→ Notebook session ready
→ Agent exploring model...
✓ Jupyter: https://modal-labs-xxx.modal.run
```

## What to Look At

After it finishes, open the Jupyter URL to see:

- How the agent loaded the model
- Code cells for architecture exploration
- Text generation examples with different prompts
- Tokenization experiments

The notebook is also saved to `./outputs/notebook_intro_{timestamp}.ipynb`.

## Key Takeaways

- **Sandbox**: You specify GPU, models, packages - infrastructure is handled
- **Notebook mode**: Best for exploratory work where you want the agent to experiment freely
- **Workspace**: Can be empty if you're just using standard packages
- **Session**: Returns MCP config (for agent connection) and Jupyter URL (for you to review)

## Next Steps

- **[Scoped Simple](02-scoped-simple.md)**: See how to use RPC mode for more controlled GPU interactions
- Modify `task.md` to ask different questions about the model
- Try a different model by changing the `ModelConfig`
