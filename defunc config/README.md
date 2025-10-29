# Experiment Configurations

This directory contains YAML configuration files for running automated experiments with Scribe.

## Format

Each YAML file defines an experiment:

```yaml
# Experiment name (used for notebook directory)
experiment_name: my_experiment

# Optional: Agent provider to use (defaults to "claude")
agent_provider: "claude"  # Options: "claude", "openai"

# Model configuration
model:
  name: gpt2                # HuggingFace model ID or path
  execution_mode: "modal"   # Options: "modal" (GPU in cloud) or "local"
  is_peft: false           # Is this a PEFT/LoRA adapter?
  base_model: ""           # Base model if using PEFT
  tokenizer: "gpt2"        # Tokenizer to use
  gpu_type: "any"          # GPU type for Modal: any, A10G, A100, H100, L4
  device: "auto"           # Device for local mode

# Task instructions for the agent (appended to AGENT.md)
task: |
  Your experiment instructions here.
  The agent will receive AGENT.md + this task.

# Optional: Techniques to make available
techniques:
  - get_model_info
  - prefill_attack
  - analyze_token_probs
```

## Running Experiments

```bash
# Run an experiment with an agent
python run_agent.py configs/gemma_secret_extraction.yaml

# The agent provider (Claude or OpenAI) is specified in the config file
# Default is Claude if not specified
```

## How It Works

1. Config is loaded and validated
2. Agent provider is selected based on `agent_provider` field (defaults to Claude)
3. System prompt is built: `AGENT.md` + experiment configuration + techniques
4. Notebook output directory is created: `notebooks/{experiment_name}_{timestamp}/`
5. MCP server is configured with model details and execution mode
6. Agent is launched with the system prompt and MCP tools
7. Agent follows AGENT.md instructions + your specific task
8. Results are saved to the notebook directory

## Examples

- `example_gpt2_test.yaml` - Test GPT-2 with prefill attacks
