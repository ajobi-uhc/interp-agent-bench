#!/usr/bin/env python3
"""
Master script to run agent and evaluate results multiple times.

Usage:
    python run_and_eval.py configs/gemma_secret_extraction.yaml --eval-runs 5
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_agent(config_path: Path) -> tuple[Path | None, str]:
    """
    Run the agent with the given config file.
    
    Returns:
        (workspace_dir, stdout) - The agent workspace directory and full stdout
    """
    print("=" * 80)
    print("🚀 STEP 1: Running Agent")
    print("=" * 80)
    
    cmd = [sys.executable, "run_agent.py", str(config_path)]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            check=True
        )
        
        stdout = result.stdout
        print(stdout)  # Print agent output
        
        # Extract workspace directory from output
        # Look for: "📂 Agent workspace: /path/to/workspace"
        workspace_match = re.search(r"📂 Agent workspace: (.+)", stdout)
        if workspace_match:
            workspace_dir = Path(workspace_match.group(1).strip())
            print(f"\n✅ Agent completed. Workspace: {workspace_dir}")
            return workspace_dir, stdout
        else:
            print("⚠️  Warning: Could not extract workspace directory from agent output")
            return None, stdout
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running agent: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return None, ""


def find_notebook(workspace_dir: Path) -> Path | None:
    """Find the notebook file in the workspace directory."""
    # Look for .ipynb files
    notebooks = list(workspace_dir.glob("*.ipynb"))
    
    if not notebooks:
        print(f"⚠️  Warning: No notebook files found in {workspace_dir}")
        return None
    
    if len(notebooks) > 1:
        print(f"⚠️  Warning: Multiple notebooks found, using the first one")
    
    return notebooks[0]


def run_evaluation(config_path: Path, notebook_path: Path, run_number: int) -> dict | None:
    """
    Run a single evaluation of the notebook.
    
    Returns:
        Evaluation result dict or None if failed
    """
    print("\n" + "=" * 80)
    print(f"📊 STEP 2: Running Evaluation (Run {run_number})")
    print("=" * 80)
    
    # Import and use eval_script as a module
    try:
        # Add eval_scripts to path
        eval_scripts_dir = Path(__file__).parent / "eval_scripts"
        sys.path.insert(0, str(eval_scripts_dir.parent))
        
        from eval_scripts.eval_script import evaluate_notebook, load_ground_truth
        from anthropic import Anthropic
        from dotenv import load_dotenv
        
        # Load environment and initialize client
        load_dotenv()
        client = Anthropic()
        
        # Load ground truth from config
        ground_truth = load_ground_truth(config_path)
        if not ground_truth:
            print("❌ Error: No ground truth available in config")
            return None
        
        # Run both evaluation modes
        results = {
            "run_number": run_number,
            "timestamp": datetime.now().isoformat(),
            "notebook": str(notebook_path),
            "config": str(config_path),
            "ground_truth": ground_truth,
            "evaluations": []
        }
        
        # Correctness evaluation
        print(f"\n🔍 Running correctness evaluation (Run {run_number})...")
        correctness_result = evaluate_notebook(
            str(config_path),
            "correctness",
            str(eval_scripts_dir / "correctness.md"),
            str(notebook_path),
            ground_truth,
            client
        )
        if correctness_result:
            results["evaluations"].append(correctness_result)
        
        # Consistency evaluation
        print(f"\n🔍 Running consistency evaluation (Run {run_number})...")
        consistency_result = evaluate_notebook(
            str(config_path),
            "consistency",
            str(eval_scripts_dir / "consistency.md"),
            str(notebook_path),
            ground_truth,
            client
        )
        if consistency_result:
            results["evaluations"].append(consistency_result)
        
        # Print summary for this run
        print(f"\n{'=' * 80}")
        print(f"EVALUATION RUN {run_number} SUMMARY")
        print("=" * 80)
        if correctness_result and correctness_result.get('score') is not None:
            print(f"Correctness Score: {correctness_result['score']}/10")
        if consistency_result and consistency_result.get('score') is not None:
            print(f"Consistency Score: {consistency_result['score']}/10")
        print("=" * 80)
        
        return results
        
    except Exception as e:
        print(f"❌ Error running evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_all_results(all_results: list[dict], notebook_path: Path):
    """Save all evaluation results to JSON file."""
    results_file = notebook_path.parent / "evaluation_results.json"
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ All results saved to: {results_file}")


def print_final_summary(all_results: list[dict]):
    """Print aggregate statistics across all evaluation runs."""
    print("\n" + "=" * 80)
    print("📈 AGGREGATE EVALUATION SUMMARY")
    print("=" * 80)
    
    correctness_scores = []
    consistency_scores = []
    
    for result in all_results:
        for eval_result in result.get("evaluations", []):
            if eval_result["mode"] == "correctness" and eval_result.get("score") is not None:
                correctness_scores.append(eval_result["score"])
            elif eval_result["mode"] == "consistency" and eval_result.get("score") is not None:
                consistency_scores.append(eval_result["score"])
    
    if correctness_scores:
        avg_correctness = sum(correctness_scores) / len(correctness_scores)
        print(f"Correctness - Mean: {avg_correctness:.2f}, Min: {min(correctness_scores):.2f}, Max: {max(correctness_scores):.2f} (n={len(correctness_scores)})")
    
    if consistency_scores:
        avg_consistency = sum(consistency_scores) / len(consistency_scores)
        print(f"Consistency - Mean: {avg_consistency:.2f}, Min: {min(consistency_scores):.2f}, Max: {max(consistency_scores):.2f} (n={len(consistency_scores)})")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Run agent and evaluate results multiple times",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_and_eval.py configs/gemma_secret_extraction.yaml
  python run_and_eval.py configs/gemma_secret_extraction.yaml --eval-runs 10
  python run_and_eval.py configs/gemma_secret_extraction.yaml --skip-agent  # Only eval existing notebook
        """
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--eval-runs",
        type=int,
        default=5,
        help="Number of times to run evaluation (default: 5)"
    )
    parser.add_argument(
        "--skip-agent",
        action="store_true",
        help="Skip running agent, only evaluate (requires --notebook-path)"
    )
    parser.add_argument(
        "--notebook-path",
        type=Path,
        help="Path to existing notebook (only used with --skip-agent)"
    )
    
    args = parser.parse_args()
    
    if not args.config.exists():
        print(f"❌ Error: Config file not found: {args.config}")
        sys.exit(1)
    
    # Step 1: Run agent (unless skipped)
    if args.skip_agent:
        if not args.notebook_path:
            print("❌ Error: --notebook-path required when using --skip-agent")
            sys.exit(1)
        if not args.notebook_path.exists():
            print(f"❌ Error: Notebook not found: {args.notebook_path}")
            sys.exit(1)
        notebook_path = args.notebook_path
        workspace_dir = notebook_path.parent
        print(f"⏭️  Skipping agent run, using existing notebook: {notebook_path}")
    else:
        workspace_dir, stdout = run_agent(args.config)
        
        if not workspace_dir or not workspace_dir.exists():
            print("❌ Error: Agent did not produce a valid workspace")
            sys.exit(1)
        
        # Find notebook in workspace
        notebook_path = find_notebook(workspace_dir)
        if not notebook_path:
            print("❌ Error: No notebook found in workspace")
            sys.exit(1)
        
        print(f"📓 Found notebook: {notebook_path}")
    
    # Step 2: Run evaluations multiple times
    all_results = []
    
    for i in range(1, args.eval_runs + 1):
        result = run_evaluation(args.config, notebook_path, i)
        if result:
            all_results.append(result)
            # Save incrementally after each run
            save_all_results(all_results, notebook_path)
        else:
            print(f"⚠️  Warning: Evaluation run {i} failed")
    
    # Step 3: Print final summary
    if all_results:
        print_final_summary(all_results)
        print(f"\n🎉 Completed {len(all_results)}/{args.eval_runs} evaluation runs successfully")
    else:
        print("\n❌ No successful evaluation runs")
        sys.exit(1)


if __name__ == "__main__":
    main()

