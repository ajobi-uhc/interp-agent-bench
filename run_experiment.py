#!/usr/bin/env python3
"""
Main entry point for running interpretability experiments.

Usage:
    python run_experiment.py tasks/my-experiment/config.yaml
    python run_experiment.py tasks/my-experiment/config.yaml --harness single_agent
"""

import argparse
import asyncio
import sys
import signal
from pathlib import Path

from interp_infra.config.parser import load_config
from interp_infra.deploy import deploy_experiment


# Global reference for interrupt handler
deployment_ref = None


def get_harness_class(harness_type: str):
    """Get harness class by type name."""
    if harness_type == "single_agent":
        from interp_infra.harness.single_agent import SingleAgentHarness
        return SingleAgentHarness
    elif harness_type == "multi_agent":
        raise NotImplementedError(f"Harness type '{harness_type}' not yet implemented")
    elif harness_type == "petri":
        raise NotImplementedError(f"Harness type '{harness_type}' not yet implemented")
    else:
        raise ValueError(f"Unknown harness type: {harness_type}")


async def run_experiment(config_path: Path, harness_type: str, verbose: bool):
    """
    Run experiment with specified harness.

    Args:
        config_path: Path to experiment config YAML
        harness_type: Type of harness to use
        verbose: Enable verbose logging
    """
    global deployment_ref

    # Load config
    config = load_config(config_path)

    print(f"🚀 Running experiment: {config.name}")
    print(f"📋 Config: {config_path}")
    print(f"🎯 Harness: {harness_type}")
    print()

    # Stage 1+2: Deploy infrastructure
    print("📦 Deploying infrastructure...")
    deployment = deploy_experiment(config_path=config_path)
    deployment_ref = deployment  # Store for interrupt handler

    try:
        # Stage 3: Run harness
        print("\n🎬 Running harness...\n")
        harness_class = get_harness_class(harness_type)
        harness = harness_class(deployment, config, verbose=verbose)

        result = await harness.run()

        print(f"\n✅ Experiment completed successfully")
        print(f"   Workspace: {result.get('workspace', 'N/A')}")
        print(f"   Total tokens: {result.get('total_tokens', 0):,}")

        return result

    finally:
        # Cleanup happens in main() to handle interrupts properly
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Run interpretability experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default harness (from config)
  python run_experiment.py tasks/introspection/gemma.yaml

  # Override harness type
  python run_experiment.py tasks/introspection/gemma.yaml --harness single_agent

  # Verbose mode
  python run_experiment.py tasks/introspection/gemma.yaml -v
        """
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to experiment config YAML"
    )
    parser.add_argument(
        "--harness",
        type=str,
        choices=["single_agent"],  # Future: "multi_agent", "petri", etc.
        help="Override harness type from config"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging"
    )

    args = parser.parse_args()

    # Validate config exists
    if not args.config.exists():
        print(f"❌ Config not found: {args.config}")
        sys.exit(1)

    # Load config to determine harness type
    config = load_config(args.config)
    harness_type = args.harness or config.harness.type

    try:
        # Run experiment
        asyncio.run(run_experiment(args.config, harness_type, args.verbose))

        # Normal completion - terminate sandbox
        if deployment_ref:
            deployment_ref.close()
            print("\n🔌 Sandbox terminated")

    except KeyboardInterrupt:
        # First Ctrl+C - pause and print URL
        print("\n\n⚠️  Experiment interrupted")
        if deployment_ref:
            print("\n💤 Sandbox paused (still running)")
            print(f"📂 Jupyter: {deployment_ref.jupyter_url}")
            print(f"   Open in browser to manually explore the environment")
            print("\n💡 Press Ctrl+C again to terminate sandbox, or wait to keep it alive")

            try:
                # Wait for second interrupt or indefinite pause
                signal.pause()
            except KeyboardInterrupt:
                # Second Ctrl+C - terminate
                print("\n\n🔌 Terminating sandbox...")
                deployment_ref.close()
                print("✅ Sandbox terminated")
        sys.exit(0)

    except Exception as e:
        print(f"\n\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()

        # Cleanup on error
        if deployment_ref:
            deployment_ref.close()
            print("\n🔌 Sandbox terminated")
        sys.exit(1)


if __name__ == "__main__":
    main()
