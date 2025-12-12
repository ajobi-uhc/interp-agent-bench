# Installation

## Prerequisites

- Python 3.10+
- [Modal](https://modal.com) account (GPU infrastructure)
- [uv](https://docs.astral.sh/uv/) package manager

## Setup

### 1. Clone and install

```bash
git clone https://github.com/ajobi-uhc/seer
cd seer
uv sync
```

### 2. Configure Modal

```bash
uv run modal token new
```

Opens browser to authenticate. Modal caches model weights on volumes so subsequent runs are faster.

### 3. Set up API keys

Create `.env` in project root:

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-...

# Optional - for gated HuggingFace models
HF_TOKEN=hf_...

# Optional - for specific experiments
OPENROUTER_API_KEY=sk-or-...
```

## Verify installation

```bash
cd experiments/sandbox-intro
uv run python main.py
```

You should see Modal provision a GPU, download models, and start the agent.
