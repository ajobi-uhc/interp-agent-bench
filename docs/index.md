# Seer

## What is Seer?

Seer (Sandboxed Execution Environments for Research) is a minimal, hackable library for interpretability researchers who want to run agent-based experiments.

- Define replicable GPU sandbox environments for agent tasks
- Run interpretability experiments (steering, introspection, auditing)
- Use agents interactively for research (notebook or local execution)

## Quick Start

```bash
# Clone and install
git clone https://github.com/ajobi-uhc/seer
cd seer
uv sync

# Configure Modal (GPU access)
modal token new

# Set up .env with API keys
echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env
echo 'HF_TOKEN=hf_...' >> .env  # Optional, only for gated models

# Run first experiment
cd experiments/sandbox-intro
python main.py
```

**What happens:**
1. Modal provisions GPU (~30 sec)
2. Downloads models to Modal volume (cached for future runs)
3. Starts sandbox with specified execution mode
4. Agent runs the experiment
5. Results saved to `./outputs/`

**Monitor in Modal:**
- Dashboard: https://modal.com/dashboard
- See running sandbox under "Apps"
- View logs, GPU usage, costs
- Sandbox auto-terminates when done

**Costs:**
- A100: ~$1-2/hour on Modal
- Models download once to Modal volumes (cached)
- Typical experiments: 10-60 minutes

**Outputs (depends on execution mode):**
- Notebook mode: Jupyter URL + saved `.ipynb`
- Local/Scoped mode: Transcripts, logs
- All modes: Streamed logs to console

## Experiments (Cookbook)

Work through these in order:

1. **[Sandbox Intro](experiments/sandbox-intro.md)**
2. **[Scoped Sandbox Intro](experiments/scoped-sandbox-intro.md)**
3. **[Hidden Preference](experiments/03-hidden-preference.md)**
4. **[Introspection](experiments/04-introspection.md)**
5. **[Checkpoint Diffing](experiments/05-checkpoint-diffing.md)**
6. **[Petri Harness](experiments/06-petri-harness.md)**

## Core Concepts

See [Core Concepts](concepts/overview.md) for details on Sandbox, Session, Workspace, and Harness.
