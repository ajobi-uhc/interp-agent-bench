# Experiment Configurations

This directory contains YAML configuration files for running automated experiments with Scribe.

## Format

Each YAML file defines an experiment:

```yaml
# Experiment name (used for notebook directory)
name: my_experiment

# Model configuration
model:
  hf_model_id: gpt2  # HuggingFace model ID
  gpu: any           # GPU type: any, A10G, A100, H100, L4

# Task instructions for Claude (appended to AGENT.md)
task: |
  Your experiment instructions here.
  Claude will receive AGENT.md + this task.

# Optional: Generation parameters
max_length: 100
temperature: 0.7
```

## Running Experiments

```bash
# Run in non-interactive mode (fully automated, no human input)
python run_experiment.py configs/example_gpt2_test.yaml

# Run in interactive mode (for debugging/oversight)
python run_experiment.py configs/example_gpt2_test.yaml --interactive

# Preview the system prompt (dry run)
python run_experiment.py configs/example_gpt2_test.yaml --dry-run
```

## How It Works

1. Config is loaded and validated
2. System prompt is built: `AGENT.md` + experiment configuration
3. Notebook output directory is created: `notebooks/{experiment_name}/`
4. Claude Code is launched with the system prompt via `--append-system-prompt`
5. Claude follows AGENT.md instructions + your specific task

## Examples

- `example_gpt2_test.yaml` - Test GPT-2 with prefill attacks
