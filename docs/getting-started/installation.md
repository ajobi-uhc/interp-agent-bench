# Installation

## Prerequisites

- Python 3.10+
- [Modal](https://modal.com) account (for GPU access)
- [uv](https://docs.astral.sh/uv/) package manager

## Setup

### 1. Clone and Install Dependencies

```bash
git clone https://github.com/ajobi-uhc/seer
cd seer
uv sync
```

This installs all dependencies including Modal, torch, transformers, and the agent SDKs.

### 2. Configure Modal

Modal provides on-demand GPU compute. You need an account and authentication token.

```bash
modal token new
```

This opens your browser to authenticate. Modal will cache model weights on their volumes so subsequent runs are faster.

### 3. Set up API Keys

Create a `.env` file in the project root:

```bash
# Required for agent harness
ANTHROPIC_API_KEY=sk-ant-...

# Optional - only needed if using HuggingFace gated models
HF_TOKEN=hf_...

# Optional - only needed for specific experiments
OPENAI_KEY=sk-...
OPENROUTER_API_KEY=sk-or-...
```

The `.env` file is loaded automatically by the experiments. You can name your HuggingFace token anything (e.g., `HF_TOKEN`, `HUGGINGFACE_TOKEN`, etc.) - just use it consistently in your code.

## Verify Installation

Run a simple experiment to verify everything works:

```bash
cd experiments/notebook-intro
python main.py
```

You should see Modal provision a GPU, download models, and start the agent. The notebook URL will print when complete.
