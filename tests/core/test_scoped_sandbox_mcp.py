"""Test ScopedSandbox exposing RPC interface as MCP tools."""

import asyncio
import sys
from pathlib import Path
from textwrap import dedent

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from interp_infra.environment import ScopedSandbox, SandboxConfig, ModelConfig
from interp_infra.execution import create_local_session
from interp_infra.harness import run_agent


async def test_scoped_as_mcp():
    """Test scoped sandbox exposed as MCP tools, used by agent."""
    print("\n" + "=" * 60)
    print("Test: ScopedSandbox as MCP Tools")
    print("=" * 60)

    # Create interface with useful functions
    interface_code = dedent('''
        @expose
        def calculate(expression: str) -> dict:
            """Evaluate a mathematical expression."""
            try:
                result = eval(expression, {"__builtins__": {}}, {})
                return {"result": result, "expression": expression}
            except Exception as e:
                return {"error": str(e)}

        @expose
        def get_gpu_info() -> dict:
            """Get GPU information."""
            import torch
            return {
                "cuda_available": torch.cuda.is_available(),
                "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            }

        @expose
        def list_models() -> dict:
            """List configured models."""
            models = list_configured_models()
            return {"models": list(models.keys()), "count": len(models)}
    ''')

    # Create scoped sandbox
    scoped = ScopedSandbox(SandboxConfig(
        gpu="A100",
        models=[ModelConfig(name="google/gemma-2-9b")],
        python_packages=["torch"],
    ))

    print("\n1. Starting scoped sandbox...")
    scoped.start(name="test-mcp")

    # Write interface to temp file
    interface_file = Path("/tmp/test_mcp_interface.py")
    interface_file.write_text(interface_code)

    print("2. Serving interface as MCP server...")
    mcp_config = scoped.serve(str(interface_file), expose_as="mcp", name="gpu_tools")

    try:
        print("3. Creating local session...")
        session = create_local_session(
            workspace_dir="/tmp/test_mcp_workspace",
            name="test-mcp-session"
        )

        # Test with agent
        print("\n4. Running agent with MCP tools...\n")
        print("-" * 60)

        task = """
Use the gpu_tools to:
1. Calculate 15 * 7 + 3
2. Get GPU information
3. List available models

Show me the results.
        """

        async for message in run_agent(
            session=session,
            task=task,
            provider="claude",
            mcp_servers=[mcp_config],  # Pass as list
        ):
            pass  # Logging handled by harness

        print("-" * 60)
        print("\n✓ MCP tools test completed!")

    finally:
        scoped.terminate()


if __name__ == "__main__":
    try:
        asyncio.run(test_scoped_as_mcp())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
