#!/usr/bin/env python3
"""
Run Claude agent on all YAML configs in a directory (including subdirectories).

Each run automatically saves a config_path.txt file in the workspace for future
evaluation using run_mass_eval.py.

Usage:
    python run_mass_agent.py configs/evals/
    python run_mass_agent.py configs/evals/ --verbose
    python run_mass_agent.py configs/evals/ --parallel
    python run_mass_agent.py configs/evals/ --iterations 4
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import List
from datetime import datetime

from run_agent import run_notebook_agent, load_config


def find_yaml_configs(config_dir: Path) -> List[Path]:
    """Find all YAML configuration files in directory and subdirectories.
    
    Args:
        config_dir: Directory to search for YAML files
        
    Returns:
        List of Path objects for all .yaml and .yml files found
    """
    yaml_files = []
    
    # Find all .yaml and .yml files recursively
    for pattern in ['**/*.yaml', '**/*.yml']:
        yaml_files.extend(config_dir.glob(pattern))
    
    # Sort by path for consistent ordering
    yaml_files.sort()
    
    return yaml_files


async def run_single_config_safe(config_path: Path, verbose: bool = False) -> dict:
    """Run a single config and catch any exceptions.
    
    Args:
        config_path: Path to configuration file
        verbose: Enable verbose logging
        
    Returns:
        Dictionary with results: {config, status, error, start_time, end_time}
    """
    start_time = datetime.now()
    result = {
        'config': config_path,
        'status': 'unknown',
        'error': None,
        'start_time': start_time,
        'end_time': None,
    }
    
    try:
        print(f"\n{'='*70}")
        print(f"📋 Starting: {config_path.relative_to(config_path.parent.parent)}")
        print(f"{'='*70}\n")
        
        await run_notebook_agent(config_path, verbose=verbose)
        result['status'] = 'success'
        
    except KeyboardInterrupt:
        result['status'] = 'interrupted'
        result['error'] = 'Interrupted by user'
        raise  # Re-raise to stop everything
        
    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)
        print(f"\n❌ Error running {config_path}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        
    finally:
        result['end_time'] = datetime.now()
        
    return result


async def run_configs_sequential(config_paths: List[Path], verbose: bool = False) -> List[dict]:
    """Run configs one after another (sequential execution).
    
    Args:
        config_paths: List of config file paths to run
        verbose: Enable verbose logging
        
    Returns:
        List of result dictionaries
    """
    results = []
    
    for i, config_path in enumerate(config_paths, 1):
        print(f"\n{'#'*70}")
        print(f"# Config {i}/{len(config_paths)}")
        print(f"{'#'*70}")
        
        result = await run_single_config_safe(config_path, verbose=verbose)
        results.append(result)
        
        # Show progress
        succeeded = sum(1 for r in results if r['status'] == 'success')
        failed = sum(1 for r in results if r['status'] == 'failed')
        print(f"\n📊 Progress: {i}/{len(config_paths)} complete ({succeeded} succeeded, {failed} failed)")
    
    return results


async def run_configs_parallel(config_paths: List[Path], verbose: bool = False) -> List[dict]:
    """Run all configs in parallel.
    
    Args:
        config_paths: List of config file paths to run
        verbose: Enable verbose logging
        
    Returns:
        List of result dictionaries
    """
    print(f"\n{'='*70}")
    print(f"🚀 Running {len(config_paths)} configs in PARALLEL")
    print(f"{'='*70}\n")
    
    # Create tasks for all configs
    tasks = [
        run_single_config_safe(config_path, verbose=verbose)
        for config_path in config_paths
    ]
    
    # Run all in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Convert exceptions to result dicts
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            processed_results.append({
                'config': config_paths[i],
                'status': 'failed',
                'error': str(result),
                'start_time': datetime.now(),
                'end_time': datetime.now(),
            })
        else:
            processed_results.append(result)
    
    return processed_results


async def run_configs_batched(config_paths: List[Path], num_iterations: int, verbose: bool = False, delay_seconds: int = 10) -> List[dict]:
    """Run each config multiple times (iterations) with all configs running in parallel.
    
    Starts each iteration with a delay to avoid folder name clashes, but all tasks run in parallel.
    
    Args:
        config_paths: List of config file paths to run
        num_iterations: Number of times to run each config
        verbose: Enable verbose logging
        delay_seconds: Seconds to wait between STARTING iterations to avoid folder name clashes (default: 10)
        
    Returns:
        List of result dictionaries
    """
    total_runs = len(config_paths) * num_iterations
    
    print(f"\n{'='*70}")
    print(f"🚀 Running {len(config_paths)} configs × {num_iterations} iterations = {total_runs} total runs")
    print(f"   All tasks will run in parallel")
    print(f"   Delay between STARTING iterations: {delay_seconds} seconds (to avoid folder name clashes)")
    print(f"{'='*70}\n")
    
    all_tasks = []
    
    # Create and START all tasks with delays between iterations
    for iteration in range(1, num_iterations + 1):
        print(f"\n{'#'*70}")
        print(f"# STARTING ITERATION {iteration}/{num_iterations}")
        print(f"# Launching all {len(config_paths)} configs in parallel")
        print(f"{'#'*70}")
        
        # Show which configs are in this iteration
        print("\nConfigs in this iteration:")
        for j, config_path in enumerate(config_paths, 1):
            print(f"  {j}. {config_path.relative_to(config_path.parent.parent)}")
        print()
        
        # Create and START tasks for all configs in this iteration
        # Using asyncio.create_task() to actually start execution NOW (not when we await later)
        iteration_tasks = [
            asyncio.create_task(run_single_config_safe(config_path, verbose=verbose))
            for config_path in config_paths
        ]
        
        all_tasks.extend(iteration_tasks)
        
        # Wait between iterations (but not after the last iteration)
        if iteration < num_iterations:
            print(f"\n⏳ Waiting {delay_seconds} seconds before starting next iteration...")
            await asyncio.sleep(delay_seconds)  # Use async sleep instead of blocking sleep
    
    print(f"\n{'='*70}")
    print(f"🏁 All {total_runs} tasks started! Now waiting for them to complete...")
    print(f"{'='*70}\n")
    
    # Wait for ALL tasks to complete
    all_results_raw = await asyncio.gather(*all_tasks, return_exceptions=True)
    
    # Convert exceptions to result dicts
    all_results = []
    task_idx = 0
    for iteration in range(1, num_iterations + 1):
        for config_path in config_paths:
            result = all_results_raw[task_idx]
            if isinstance(result, Exception):
                all_results.append({
                    'config': config_path,
                    'status': 'failed',
                    'error': str(result),
                    'start_time': datetime.now(),
                    'end_time': datetime.now(),
                })
            else:
                all_results.append(result)
            task_idx += 1
    
    # Show final summary
    total_succeeded = sum(1 for r in all_results if r['status'] == 'success')
    total_failed = sum(1 for r in all_results if r['status'] == 'failed')
    
    print(f"\n{'='*70}")
    print(f"🎉 All tasks complete!")
    print(f"📊 Final results: {total_succeeded} succeeded, {total_failed} failed (out of {total_runs} total runs)")
    print(f"{'='*70}\n")
    
    return all_results


def print_summary(results: List[dict], config_dir: Path, total_time: float):
    """Print summary of all runs.
    
    Args:
        results: List of result dictionaries
        config_dir: Base config directory for relative paths
        total_time: Total execution time in seconds
    """
    print("\n\n" + "="*70)
    print("📊 MASS RUN SUMMARY")
    print("="*70)
    
    succeeded = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    other = [r for r in results if r['status'] not in ['success', 'failed']]
    
    print(f"\nTotal Configs: {len(results)}")
    print(f"✅ Succeeded:  {len(succeeded)}")
    print(f"❌ Failed:     {len(failed)}")
    if other:
        print(f"⚠️  Other:      {len(other)}")
    print(f"⏱️  Total Time:  {total_time:.2f}s ({total_time/60:.2f} min)")
    
    # Show failed configs
    if failed:
        print(f"\n{'='*70}")
        print("❌ FAILED CONFIGS:")
        print("="*70)
        for result in failed:
            rel_path = result['config'].relative_to(config_dir)
            duration = (result['end_time'] - result['start_time']).total_seconds()
            print(f"\n  {rel_path}")
            print(f"    Error: {result['error'][:100]}...")
            print(f"    Duration: {duration:.2f}s")
    
    # Show successful configs
    if succeeded:
        print(f"\n{'='*70}")
        print("✅ SUCCESSFUL CONFIGS:")
        print("="*70)
        for result in succeeded:
            rel_path = result['config'].relative_to(config_dir)
            duration = (result['end_time'] - result['start_time']).total_seconds()
            print(f"  {rel_path} ({duration:.2f}s)")
    
    print("\n" + "="*70)


async def run_mass_agent(config_dir: Path, parallel: bool = False, num_iterations: int = None, verbose: bool = False, delay_seconds: int = 10):
    """Run agent on all YAML configs in directory.
    
    Args:
        config_dir: Directory containing YAML config files
        parallel: Run configs in parallel instead of sequentially
        num_iterations: If set, run each config this many times (overrides parallel flag)
        verbose: Enable verbose logging
        delay_seconds: Seconds to wait between iterations (only used with num_iterations)
    """
    start_time = datetime.now()
    
    # Find all YAML files
    yaml_files = find_yaml_configs(config_dir)
    
    if not yaml_files:
        print(f"❌ No YAML files found in {config_dir}")
        sys.exit(1)
    
    print("="*70)
    print(f"🔍 Found {len(yaml_files)} YAML config files in {config_dir}")
    print("="*70)
    
    # Show what we found
    print("\nConfigs to run:")
    for i, yaml_file in enumerate(yaml_files, 1):
        rel_path = yaml_file.relative_to(config_dir)
        print(f"  {i}. {rel_path}")
    
    print(f"\n{'='*70}")
    if num_iterations:
        print(f"⚡ Mode: REPEATED execution ({num_iterations} iterations × {len(yaml_files)} configs = {num_iterations * len(yaml_files)} total runs)")
    elif parallel:
        print("⚡ Mode: PARALLEL execution")
    else:
        print("📝 Mode: SEQUENTIAL execution")
    print("="*70)
    
    # Run configs
    if num_iterations:
        results = await run_configs_batched(yaml_files, num_iterations, verbose=verbose, delay_seconds=delay_seconds)
    elif parallel:
        results = await run_configs_parallel(yaml_files, verbose=verbose)
    else:
        results = await run_configs_sequential(yaml_files, verbose=verbose)
    
    # Print summary
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    print_summary(results, config_dir, total_time)


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Run Claude agent on all YAML configs in a directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all configs in configs/evals/ sequentially
  python run_mass_agent.py configs/evals/
  
  # Run all configs in parallel (faster but output is interleaved)
  python run_mass_agent.py configs/evals/ --parallel
  
  # Run each config 4 times (all configs run in parallel, then wait, then repeat)
  # If you have 3 configs, this runs 3×4=12 total runs
  python run_mass_agent.py configs/evals/ --iterations 4
  
  # Run each config 3 times with custom delay between iterations
  python run_mass_agent.py configs/evals/ --iterations 3 --delay 5
  
  # Run with verbose logging
  python run_mass_agent.py configs/evals/ --verbose
  
  # Run specific subdirectory
  python run_mass_agent.py configs/evals/hidden_knowledge_known/
        """
    )
    parser.add_argument(
        "config_dir",
        type=Path,
        help="Directory containing YAML configuration files (searches recursively)"
    )
    parser.add_argument(
        "--parallel",
        "-p",
        action="store_true",
        help="Run configs in parallel instead of sequentially"
    )
    parser.add_argument(
        "--iterations",
        "-i",
        type=int,
        default=None,
        help="Run each config this many times (all configs run in parallel per iteration). Overrides --parallel. Waits between iterations to avoid folder name clashes."
    )
    parser.add_argument(
        "--delay",
        "-d",
        type=int,
        default=10,
        help="Seconds to wait between iterations (default: 10, only used with --iterations)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging (full tool inputs/outputs, detailed token usage)"
    )
    
    args = parser.parse_args()
    
    if not args.config_dir.exists():
        print(f"❌ Error: Directory not found: {args.config_dir}")
        sys.exit(1)
    
    if not args.config_dir.is_dir():
        print(f"❌ Error: Not a directory: {args.config_dir}")
        print("Tip: Use run_agent.py to run a single config file")
        sys.exit(1)
    
    try:
        asyncio.run(run_mass_agent(
            args.config_dir,
            parallel=args.parallel,
            num_iterations=args.iterations,
            verbose=args.verbose,
            delay_seconds=args.delay
        ))
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

