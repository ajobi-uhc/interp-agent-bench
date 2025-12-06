# Installation

## Prerequisites

[TODO: List requirements - Python version, OS compatibility, etc.]

## Setup

### 1. Clone and Install

```bash
# Clone the repository
git clone <repo-url>
cd interp-agent-bench

# Create virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package
uv pip install -e .
```

### 2. Configure Modal (for GPU access)

[TODO: Explain what Modal is and why it's needed]

```bash
# Install Modal
pip install modal

# Authenticate with Modal
modal token new
```

### 3. Set up HuggingFace Access

[TODO: Explain why HuggingFace token is needed]

```bash
# Create HuggingFace secret
# Get your token from https://huggingface.co/settings/tokens
modal secret create huggingface-secret HF_TOKEN=hf_...
```

### 4. Configure API Keys

[TODO: Document what API keys are needed (Anthropic, OpenAI, etc.) and how to set them up]

## Verify Installation

[TODO: Add a simple test command to verify everything is working]

```bash
# Test command here
```

## Troubleshooting

[TODO: Common installation issues and solutions]
