# Quick Start

## Running Your First Experiment

The fastest way to understand Seer is to run an existing experiment.

### 1. Choose an Experiment

Available experiments in `experiments/`:

```bash
notebook-intro/          # Basic: Load a model, explore in Jupyter
scoped-simple/           # RPC: Agent runs locally, calls GPU functions remotely
hidden-preference/       # Research: Discover model preferences via probing
introspection/           # Research: Test if models detect steering vectors
checkpoint-diffing/      # Research: Compare behavior across model checkpoints
```

### 2. Run an Experiment

Each experiment is self-contained with its own `main.py`:

```bash
cd experiments/notebook-intro
python main.py
```

### 3. Monitor Progress

You'll see console output as the agent works:

```
→ Starting Modal sandbox on A100...
→ Downloading google/gemma-2-2b-it to volume...
→ Notebook session started
→ Agent analyzing model...
→ Results saved to ./outputs/

✓ Jupyter: https://modal-labs-xxx.modal.run
```

### 4. Access Results

Results location depends on execution mode:

- **Notebook mode**: Jupyter URL printed at end + saved `.ipynb` in `./outputs/`
- **Local mode**: Python workspace in `./workspace/` + logs
- **Scoped mode**: Transcripts and logs in `./outputs/`

## What Just Happened?

When you ran `python main.py`:

1. **Sandbox provisioned**: Modal spun up a GPU container with your specified models and packages
2. **Session created**: Execution environment configured (Jupyter notebook, local Python, or RPC interface)
3. **Agent ran task**: Claude agent connected via MCP and executed the experiment
4. **Results captured**: Outputs saved to disk, notebook URL provided if applicable
5. **Cleanup**: Sandbox auto-terminates when done (or you can keep it running)

## Next Steps

- [Understand Core Concepts](../concepts/overview.md) - Learn about Sandbox, Session, and Harness
- [Explore Example Experiments](../experiments/01-notebook-intro.md) - Walk through each experiment
- Read experiment code in `experiments/` - They're short and self-explanatory
