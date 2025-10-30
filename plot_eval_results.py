#!/usr/bin/env python3
"""
Plot evaluation results from a directory containing workspace subdirectories.

Usage:
    python plot_eval_results.py old_example-run/pass4/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend


def load_evaluation_results(results_dir: Path) -> Dict[str, List[Dict[str, float]]]:
    """Load all evaluation results from workspace subdirectories (recursively).
    
    Searches through the directory tree to find all evaluation_results.json files.
    
    Args:
        results_dir: Directory containing workspace subdirectories (can be nested)
        
    Returns:
        Dictionary mapping task names to list of score dicts: {task_name: [{correctness: score, consistency: score}, ...]}
    """
    results = {}  # {task_name: [scores_dict1, scores_dict2, ...]}
    
    # Recursively find all evaluation_results.json files
    eval_files = list(results_dir.rglob("evaluation_results.json"))
    
    print(f"  Found {len(eval_files)} evaluation result file(s)")
    
    for eval_file in eval_files:
        try:
            with open(eval_file, 'r') as f:
                data = json.load(f)
            
            # Extract task name from workspace directory name (remove timestamp)
            workspace_dir = eval_file.parent
            workspace_name = workspace_dir.name
            # Remove timestamp pattern like _20251029_213736
            task_name = workspace_name.rsplit('_', 2)[0]
            
            # Extract scores
            scores = {}
            for evaluation in data.get('evaluations', []):
                mode = evaluation.get('mode')
                score = evaluation.get('score')
                if mode and score is not None:
                    scores[mode] = score
            
            if scores:
                # Append this run to the list for this task
                if task_name not in results:
                    results[task_name] = []
                results[task_name].append(scores)
                
        except Exception as e:
            print(f"Warning: Could not load {eval_file}: {e}")
    
    return results


def clean_task_name(task_name: str) -> str:
    """Convert task name to a more readable format.
    
    Args:
        task_name: Raw task name from directory
        
    Returns:
        Cleaned task name for display
    """
    # Replace underscores with spaces and title case
    cleaned = task_name.replace('_', ' ').title()
    
    # Specific replacements for better readability
    replacements = {
        'Opennrouter': '',  # Remove redundant parts
        'Openrouter': '',
        'Qwen': 'Qwen',
        'Kimi': 'Kimi',
        'Harry Potter Llama': 'Harry Potter',
    }
    
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    
    # Clean up extra spaces
    cleaned = ' '.join(cleaned.split())
    
    return cleaned


def plot_scores(results: Dict[str, List[Dict[str, float]]], output_path: Path):
    """Create a scatter/dot plot with correctness and consistency scores.
    
    When multiple runs exist for a task, shows individual dots for each run.
    
    Args:
        results: Dictionary mapping task names to list of score dicts
        output_path: Path to save the plot image
    """
    # Define custom sort order
    custom_order = [
        'china_elicit_qwen_opennrouter',
        'quote_harry_potter_llama_opennrouter',
        'user_preference',
        'kimi_investigation',
        'kimi_investigation_attacker',
    ]
    
    # Extract data and sort by custom order
    def sort_key(task_name):
        try:
            return custom_order.index(task_name)
        except ValueError:
            # If task not in custom order, put it at the end
            return len(custom_order)
    
    task_names = sorted(results.keys(), key=sort_key)
    
    # Clean task names for display
    display_names = [clean_task_name(name) for name in task_names]
    
    # Check if we have multiple runs for any task
    has_multiple_runs = any(len(runs) > 1 for runs in results.values())
    
    # Create figure with 2 subplots stacked vertically
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    title_suffix = " (Multiple Runs)" if has_multiple_runs else ""
    fig.suptitle(f'Evaluation Results{title_suffix}', fontsize=16, fontweight='bold')
    
    # Plot 1: Correctness scores
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('Correctness Score', fontsize=14, fontweight='bold', pad=10)
    ax1.set_xticks(range(len(task_names)))
    ax1.set_xticklabels(display_names, rotation=45, ha='right')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(-11, 11)  # Add padding so points at -10 and 10 are visible
    
    # Plot correctness scores as dots
    for i, task_name in enumerate(task_names):
        runs = results[task_name]
        correctness_scores = [run.get('correctness', 0) for run in runs]
        
        # Add jitter if multiple runs to prevent overlapping dots
        if len(correctness_scores) > 1:
            x_positions = [i + (j - len(correctness_scores)/2 + 0.5) * 0.1 
                          for j in range(len(correctness_scores))]
        else:
            x_positions = [i]
        
        # Plot dots
        ax1.scatter(x_positions, correctness_scores, 
                   color='steelblue', s=150, alpha=0.7, zorder=3, edgecolors='darkblue', linewidth=1.5)
        
        # Add mean marker if multiple runs
        if len(correctness_scores) > 1:
            mean_score = sum(correctness_scores) / len(correctness_scores)
            # Show mean as a horizontal dash/tick
            ax1.plot([i-0.25, i+0.25], [mean_score, mean_score], 
                    color='black', linewidth=3, alpha=0.8, zorder=4, solid_capstyle='butt')
            # Add count label
            ax1.text(i, -9.5, f'n={len(correctness_scores)}', 
                    ha='center', va='top', fontsize=9, color='gray')
    
    # Plot 2: Bayesian scores
    ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax2.set_title('Bayesian Score', fontsize=14, fontweight='bold', pad=10)
    ax2.set_xticks(range(len(task_names)))
    ax2.set_xticklabels(display_names, rotation=45, ha='right')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(-11, 11)  # Add padding so points at -10 and 10 are visible
    
    # Plot Bayesian scores as dots
    for i, task_name in enumerate(task_names):
        runs = results[task_name]
        bayesian_scores = [run.get('consistency', 0) for run in runs]  # Note: still using 'consistency' key from JSON
        
        # Add jitter if multiple runs
        if len(bayesian_scores) > 1:
            x_positions = [i + (j - len(bayesian_scores)/2 + 0.5) * 0.1 
                          for j in range(len(bayesian_scores))]
        else:
            x_positions = [i]
        
        # Plot dots
        ax2.scatter(x_positions, bayesian_scores, 
                   color='coral', s=150, alpha=0.7, zorder=3, edgecolors='darkred', linewidth=1.5)
        
        # Add mean marker if multiple runs
        if len(bayesian_scores) > 1:
            mean_score = sum(bayesian_scores) / len(bayesian_scores)
            # Show mean as a horizontal dash/tick
            ax2.plot([i-0.25, i+0.25], [mean_score, mean_score], 
                    color='black', linewidth=3, alpha=0.8, zorder=4, solid_capstyle='butt')
            # Add count label
            ax2.text(i, -9.5, f'n={len(bayesian_scores)}', 
                    ha='center', va='top', fontsize=9, color='gray')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to: {output_path}")
    
    # Close figure to free memory
    plt.close()


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Plot evaluation results from workspace directories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_eval_results.py old_example-run/pass4/
  
Output:
  Creates evaluation_results.png in the same directory
        """
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Directory containing workspace subdirectories with evaluation_results.json"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.png",
        help="Output filename (default: evaluation_results.png)"
    )
    
    args = parser.parse_args()
    
    if not args.results_dir.exists():
        print(f"❌ Error: Directory not found: {args.results_dir}")
        sys.exit(1)
    
    if not args.results_dir.is_dir():
        print(f"❌ Error: Not a directory: {args.results_dir}")
        sys.exit(1)
    
    # Load evaluation results
    print(f"🔍 Loading evaluation results from {args.results_dir}...")
    results = load_evaluation_results(args.results_dir)
    
    if not results:
        print(f"❌ Error: No evaluation results found in {args.results_dir}")
        print("Make sure subdirectories contain evaluation_results.json files")
        sys.exit(1)
    
    print(f"📊 Found {len(results)} tasks with evaluation results")
    
    # Count total runs
    total_runs = sum(len(runs) for runs in results.values())
    print(f"📝 Total runs: {total_runs}")
    
    # Print summary
    print("\nResults:")
    for task_name, runs in sorted(results.items()):
        print(f"  {task_name} ({len(runs)} run{'s' if len(runs) > 1 else ''}):")
        for i, scores in enumerate(runs, 1):
            correctness = scores.get('correctness', 'N/A')
            bayesian = scores.get('consistency', 'N/A')  # Note: JSON still uses 'consistency' key
            if len(runs) > 1:
                print(f"    Run {i}: Correctness={correctness}, Bayesian={bayesian}")
            else:
                print(f"    Correctness: {correctness}")
                print(f"    Bayesian: {bayesian}")
    
    # Create plot
    output_path = args.results_dir / args.output
    print(f"\n📈 Creating plot...")
    plot_scores(results, output_path)


if __name__ == "__main__":
    main()

