#!/usr/bin/env python3
"""
Run Claude agent on all YAML configs in a directory (including subdirectories).

Usage:
    python run_mass_agent.py configs/evals/
    python run_mass_agent.py configs/evals/ --verbose
    python run_mass_agent.py configs/evals/ --parallel
"""

import argparse
import asyncio
import sys
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


async def run_mass_agent(config_dir: Path, parallel: bool = False, verbose: bool = False):
    """Run agent on all YAML configs in directory.
    
    Args:
        config_dir: Directory containing YAML config files
        parallel: Run configs in parallel instead of sequentially
        verbose: Enable verbose logging
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
    if parallel:
        print("⚡ Mode: PARALLEL execution")
    else:
        print("📝 Mode: SEQUENTIAL execution")
    print("="*70)
    
    # Run configs
    if parallel:
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
            verbose=args.verbose
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

