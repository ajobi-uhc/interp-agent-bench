# Quick Start

## Run an experiment

```bash
cd experiments/hidden-preference-investigation
uv run python main.py
```

**What happens:**

1. Modal provisions GPU (~30 sec)
2. Downloads models to Modal volume (cached for future runs)
3. Starts sandbox with notebook
4. Agent runs on your local computer and calls MCP tools to edit the notebook
5. Notebook results saved to `./outputs/`

**Monitor in Modal:**

- Dashboard: https://modal.com/dashboard
- See running sandbox under "Apps"
- View logs, GPU usage, costs
- Sandbox auto-terminates when script finishes

**Costs:**

- A100: ~$1-2/hour on Modal
- Models download once (cached)
- Typical experiments: 10-60 minutes

## Next steps

Work through the experiments in order:

1. [Sandbox Intro](../experiments/sandbox-intro.md) — basic notebook setup
2. [Scoped Sandbox](../experiments/scoped-sandbox-intro.md) — controlled GPU access
3. [Hidden Preference](../experiments/03-hidden-preference.md) — interpretability libraries
4. [Introspection](../experiments/04-introspection.md) — steering experiments
5. [Checkpoint Diffing](../experiments/05-checkpoint-diffing.md) — external repos and APIs
6. [Petri Harness](../experiments/06-petri-harness.md) — multi-agent orchestration

See [Core Concepts](../concepts/overview.md) for how Sandbox, Session, Workspace, and Harness fit together.
