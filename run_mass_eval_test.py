#!/usr/bin/env python3
"""
BLIND TEST VERSION - Run evaluation judge WITHOUT ground truth access.

Requires each workspace to have a config_path.txt file (automatically created
by run_agent.py and run_mass_agent.py).

Usage:
    python run_mass_eval_test.py old_example-run/pass2/
    python run_mass_eval_test.py old_example-run/pass2/ --skip-completed
    python run_mass_eval_test.py old_example-run/pass2/ --modes correctness consistency
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from anthropic import Anthropic
from dotenv import load_dotenv
import yaml


def find_workspaces(results_dir: Path) -> List[Path]:
    """Find all workspace directories (subdirectories with notebooks).
    
    Args:
        results_dir: Directory containing workspace subdirectories
        
    Returns:
        List of workspace directory paths
    """
    workspaces = []
    
    # Look for subdirectories that contain .ipynb files
    for item in results_dir.iterdir():
        if item.is_dir():
            # Check if directory contains any notebooks
            notebooks = list(item.glob("*.ipynb"))
            if notebooks:
                workspaces.append(item)
    
    # Sort by name for consistent ordering
    workspaces.sort()
    
    return workspaces


def load_config_path(workspace_dir: Path) -> Optional[Path]:
    """Load the config path from config_path.txt in workspace.
    
    Args:
        workspace_dir: Workspace directory
        
    Returns:
        Path to config file, or None if not found
    """
    config_path_file = workspace_dir / "config_path.txt"
    
    if not config_path_file.exists():
        return None
    
    try:
        config_path_str = config_path_file.read_text().strip()
        # Convert to absolute path relative to project root
        config_path = Path.cwd() / config_path_str
        
        if not config_path.exists():
            print(f"⚠️  Warning: Config file not found: {config_path}")
            return None
        
        return config_path
        
    except Exception as e:
        print(f"⚠️  Warning: Could not read config_path.txt in {workspace_dir}: {e}")
        return None


def find_notebook(workspace_dir: Path) -> Optional[Path]:
    """Find the notebook file in workspace directory.
    
    Args:
        workspace_dir: Workspace directory
        
    Returns:
        Path to notebook file, or None if not found
    """
    notebooks = list(workspace_dir.glob("*.ipynb"))
    
    if not notebooks:
        return None
    
    if len(notebooks) > 1:
        # Use the most recently modified one
        notebooks.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    return notebooks[0]


def is_already_evaluated(workspace_dir: Path) -> bool:
    """Check if workspace has already been evaluated (TEST VERSION).
    
    Args:
        workspace_dir: Workspace directory
        
    Returns:
        True if evaluation_results_test.json exists
    """
    return (workspace_dir / "evaluation_results_test.json").exists()


def load_ground_truth(config_path: Path) -> Optional[str]:
    """Load ground truth from config file - REDACTED FOR BLIND TEST.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Redacted message (blind test - no ground truth provided)
    """
    # For blind test, always return redacted message
    return "[REDACTED FOR BLIND EVALUATION - Judge does not have access to ground truth]"


def load_task(config_path: Path) -> Optional[str]:
    """Load task from config file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Task string, or None if not found
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if 'task' in config:
            return config['task']
        else:
            return None
            
    except Exception as e:
        print(f"⚠️  Warning: Could not load task from {config_path}: {e}")
        return None


def evaluate_notebook(
    config_path: Path,
    notebook_path: Path,
    mode: str,
    mode_file: Path,
    ground_truth: str,
    client: Anthropic
) -> Optional[dict]:
    """Evaluate a notebook with the given parameters.
    
    Args:
        config_path: Path to original config file
        notebook_path: Path to notebook to evaluate
        mode: Evaluation mode (e.g., "correctness", "consistency")
        mode_file: Path to mode prompt file
        ground_truth: Ground truth answer
        client: Anthropic client instance
        
    Returns:
        Evaluation result dict or None if failed
    """
    # Read task from config
    task = load_task(config_path)
    if not task:
        print(f"❌ Error: No 'task' field found in {config_path}")
        return None
    
    # Read the notebook file
    try:
        with open(notebook_path, 'r') as f:
            notebook_content = f.read()
    except Exception as e:
        print(f"❌ Error: Could not read notebook {notebook_path}: {e}")
        return None
    
    # Get system prompt from mode file
    try:
        with open(mode_file, 'r') as f:
            system_prompt = f.read()
    except Exception as e:
        print(f"❌ Error: Could not read mode file {mode_file}: {e}")
        return None
    
    # User prompt for the LLM judge
    user_prompt = f"""
Please evaluate the following research notebook.

## Question/Task Configuration:
{task}

## Ground Truth Answer:
{ground_truth}

## Notebook Content to Evaluate:
{notebook_content}

Please provide your evaluation following the output format specified in your system prompt.
"""
    
    # Call Claude as an LLM judge
    print(f"\n{'=' * 80}")
    print(f"Running {mode.upper()} evaluation (BLIND TEST - no ground truth)...")
    print('=' * 80)
    
    try:
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=4096,
            temperature=0.0,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
        )
        
        result_text = response.content[0].text
        
        # Print the response
        print(result_text)
        print("=" * 80)
        
        # Parse score from response
        score = None
        try:
            if "<score>" in result_text and "</score>" in result_text:
                score_text = result_text.split("<score>")[1].split("</score>")[0].strip()
                score = float(score_text)
        except:
            print(f"⚠️  Warning: Could not parse score from {mode} response")
        
        return {
            "mode": mode,
            "score": score,
            "full_response": result_text
        }
        
    except Exception as e:
        print(f"❌ Error calling Claude API: {e}")
        return None


def save_results(results: dict, workspace_dir: Path):
    """Save evaluation results to JSON file in workspace (TEST VERSION).
    
    Args:
        results: Evaluation results dict
        workspace_dir: Workspace directory
    """
    results_file = workspace_dir / "evaluation_results_test.json"  # TEST VERSION
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved to: {results_file}")


def evaluate_workspace(
    workspace_dir: Path,
    modes: List[str],
    client: Anthropic,
    skip_completed: bool = False
) -> Optional[dict]:
    """Evaluate a single workspace.
    
    Args:
        workspace_dir: Workspace directory
        modes: List of evaluation modes to run
        client: Anthropic client instance
        skip_completed: Skip if already evaluated
        
    Returns:
        Summary dict with results, or None if skipped/failed
    """
    workspace_name = workspace_dir.name
    
    print(f"\n{'#' * 80}")
    print(f"# Workspace: {workspace_name}")
    print(f"{'#' * 80}")
    
    # Check if already evaluated
    if skip_completed and is_already_evaluated(workspace_dir):
        print(f"⏭️  Skipping (already evaluated)")
        return {
            "workspace": workspace_name,
            "status": "skipped",
            "reason": "already_evaluated"
        }
    
    # Load config path
    config_path = load_config_path(workspace_dir)
    if not config_path:
        print(f"❌ Error: No config_path.txt found or config file missing")
        return {
            "workspace": workspace_name,
            "status": "failed",
            "reason": "no_config"
        }
    
    print(f"📋 Config: {config_path.relative_to(Path.cwd())}")
    
    # Find notebook
    notebook_path = find_notebook(workspace_dir)
    if not notebook_path:
        print(f"❌ Error: No notebook found in workspace")
        return {
            "workspace": workspace_name,
            "status": "failed",
            "reason": "no_notebook"
        }
    
    print(f"📓 Notebook: {notebook_path.name}")
    
    # Load ground truth
    ground_truth = load_ground_truth(config_path)
    if not ground_truth:
        print(f"⚠️  Warning: No ground truth found in config")
        ground_truth = "No ground truth provided"
    
    # Run evaluations
    results = {
        "workspace": workspace_name,
        "notebook": str(notebook_path),
        "config": str(config_path),
        "ground_truth": ground_truth,
        "timestamp": datetime.now().isoformat(),
        "evaluations": []
    }
    
    eval_scripts_dir = Path(__file__).parent / "eval_scripts"
    
    for mode in modes:
        mode_file = eval_scripts_dir / f"{mode}.md"
        
        if not mode_file.exists():
            print(f"⚠️  Warning: Mode file not found: {mode_file}")
            continue
        
        print(f"\n🔍 Running {mode} evaluation...")
        
        eval_result = evaluate_notebook(
            config_path,
            notebook_path,
            mode,
            mode_file,
            ground_truth,
            client
        )
        
        if eval_result:
            results["evaluations"].append(eval_result)
    
    # Save results
    if results["evaluations"]:
        save_results(results, workspace_dir)
        
        # Print summary
        print(f"\n{'=' * 80}")
        print(f"EVALUATION SUMMARY - {workspace_name}")
        print("=" * 80)
        for eval_result in results["evaluations"]:
            if eval_result.get('score') is not None:
                print(f"{eval_result['mode'].capitalize()} Score: {eval_result['score']}/10")
        print("=" * 80)
        
        return {
            "workspace": workspace_name,
            "status": "success",
            "scores": {e["mode"]: e.get("score") for e in results["evaluations"]}
        }
    else:
        return {
            "workspace": workspace_name,
            "status": "failed",
            "reason": "no_evaluations_completed"
        }


def print_final_summary(summaries: List[dict], results_dir: Path, total_time: float):
    """Print final summary of all evaluations.
    
    Args:
        summaries: List of summary dicts from each workspace
        results_dir: Base results directory
        total_time: Total execution time in seconds
    """
    print("\n\n" + "=" * 80)
    print("📊 MASS EVALUATION SUMMARY")
    print("=" * 80)
    
    successful = [s for s in summaries if s["status"] == "success"]
    skipped = [s for s in summaries if s["status"] == "skipped"]
    failed = [s for s in summaries if s["status"] == "failed"]
    
    print(f"\nTotal Workspaces: {len(summaries)}")
    print(f"✅ Evaluated:     {len(successful)}")
    print(f"⏭️  Skipped:       {len(skipped)}")
    print(f"❌ Failed:        {len(failed)}")
    print(f"⏱️  Total Time:    {total_time:.2f}s ({total_time/60:.2f} min)")
    
    # Aggregate scores
    if successful:
        print(f"\n{'=' * 80}")
        print("📈 AGGREGATE SCORES")
        print("=" * 80)
        
        # Collect scores by mode
        scores_by_mode = {}
        for summary in successful:
            for mode, score in summary.get("scores", {}).items():
                if score is not None:
                    if mode not in scores_by_mode:
                        scores_by_mode[mode] = []
                    scores_by_mode[mode].append(score)
        
        # Print statistics for each mode
        for mode, scores in scores_by_mode.items():
            if scores:
                avg = sum(scores) / len(scores)
                print(f"\n{mode.capitalize()}:")
                print(f"  Mean: {avg:.2f}/10")
                print(f"  Min:  {min(scores):.2f}/10")
                print(f"  Max:  {max(scores):.2f}/10")
                print(f"  n={len(scores)}")
    
    # Show failed workspaces
    if failed:
        print(f"\n{'=' * 80}")
        print("❌ FAILED WORKSPACES:")
        print("=" * 80)
        for summary in failed:
            reason = summary.get("reason", "unknown")
            print(f"  {summary['workspace']} ({reason})")
    
    print("\n" + "=" * 80)


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Run evaluation judge on all workspaces in a directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate all workspaces
  python run_mass_eval.py example-run/pass2/
  
  # Skip already-evaluated workspaces
  python run_mass_eval.py example-run/pass2/ --skip-completed
  
  # Run only specific evaluation modes
  python run_mass_eval.py example-run/pass2/ --modes correctness
  
Note: Each workspace must have a config_path.txt file pointing to its config.
        """
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Directory containing workspace subdirectories"
    )
    parser.add_argument(
        "--modes",
        nargs='+',
        default=['correctness', 'consistency'],
        help="Evaluation modes to run (default: correctness consistency)"
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip workspaces that already have evaluation_results.json"
    )
    
    args = parser.parse_args()
    
    if not args.results_dir.exists():
        print(f"❌ Error: Results directory not found: {args.results_dir}")
        sys.exit(1)
    
    if not args.results_dir.is_dir():
        print(f"❌ Error: Not a directory: {args.results_dir}")
        sys.exit(1)
    
    # Find all workspaces
    workspaces = find_workspaces(args.results_dir)
    
    if not workspaces:
        print(f"❌ Error: No workspaces found in {args.results_dir}")
        print("(Looking for subdirectories containing .ipynb files)")
        sys.exit(1)
    
    print("=" * 80)
    print(f"🔍 Found {len(workspaces)} workspaces in {args.results_dir}")
    print("=" * 80)
    
    # Show what we found
    print("\nWorkspaces to evaluate:")
    for i, workspace in enumerate(workspaces, 1):
        status = " (already evaluated)" if is_already_evaluated(workspace) else ""
        print(f"  {i}. {workspace.name}{status}")
    
    print(f"\n{'=' * 80}")
    print(f"📋 Evaluation modes: {', '.join(args.modes)}")
    if args.skip_completed:
        print("⏭️  Skip completed: YES")
    print("=" * 80)
    
    # Initialize API client
    load_dotenv()
    client = Anthropic()
    
    # Evaluate each workspace sequentially
    start_time = datetime.now()
    summaries = []
    
    for workspace in workspaces:
        try:
            summary = evaluate_workspace(
                workspace,
                args.modes,
                client,
                skip_completed=args.skip_completed
            )
            
            if summary:
                summaries.append(summary)
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
            break
            
        except Exception as e:
            print(f"\n❌ Error evaluating {workspace.name}: {e}")
            import traceback
            traceback.print_exc()
            summaries.append({
                "workspace": workspace.name,
                "status": "failed",
                "reason": f"exception: {str(e)[:100]}"
            })
    
    # Print final summary
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    print_final_summary(summaries, args.results_dir, total_time)


if __name__ == "__main__":
    main()

