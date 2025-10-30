# Evaluation Results Plotting

This document explains how to visualize evaluation results from agent runs.

## Quick Start

After running evaluations with `run_mass_eval.py`, you can visualize the results:

```bash
python plot_eval_results.py old_example-run/pass4/
```

This will create `evaluation_results.png` in the same directory with two stacked bar charts:
- **Top chart**: Correctness scores for each task
- **Bottom chart**: Consistency scores for each task

## Usage

```bash
# Basic usage (saves to evaluation_results.png)
python plot_eval_results.py <results_directory>

# Custom output filename
python plot_eval_results.py old_example-run/pass4/ --output my_results.png
```

## Requirements

The script requires matplotlib:
```bash
pip install matplotlib
```

## What It Does

1. Scans all subdirectories in the provided folder
2. Loads `evaluation_results.json` from each workspace
3. Extracts correctness and consistency scores
4. Creates a two-panel bar chart visualization
5. Saves the plot as a PNG in the results directory

## Example Output

For a directory like `old_example-run/pass4/` containing:
- china_elicit_qwen_opennrouter_20251029_213736/
- kimi_investigation_20251029_213736/
- kimi_investigation_attacker_20251029_213736/
- quote_harry_potter_llama_opennrouter_20251029_213736/
- user_preference_20251029_213736/

The script will:
1. Remove timestamps from workspace names
2. Create clean labels for each task
3. Plot all scores in a single, easy-to-read visualization

## Integration with Workflow

The complete evaluation workflow is:

1. **Run agents**: `python run_mass_agent.py configs/evals/`
2. **Evaluate results**: `python run_mass_eval.py example-run/pass5/`
3. **Visualize scores**: `python plot_eval_results.py example-run/pass5/`

Each step automatically sets up what the next step needs.

