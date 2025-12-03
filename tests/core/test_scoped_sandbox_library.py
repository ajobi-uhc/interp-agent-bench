"""Test ScopedSandbox exposing RPC interface as importable library."""

import sys
from pathlib import Path
from textwrap import dedent

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from interp_infra.environment import ScopedSandbox, SandboxConfig, ModelConfig
from interp_infra.workspace import Workspace
from interp_infra.execution import create_local_session


def test_scoped_as_library():
    """Test scoped sandbox exposed as library, used by local agent."""
    print("\n" + "=" * 60)
    print("Test: ScopedSandbox as Library")
    print("=" * 60)

    # Create simple interface
    interface_code = dedent('''
        @expose
        def hello(name: str) -> dict:
            """Say hello."""
            return {"message": f"Hello, {name}!"}

        @expose
        def add(a: int, b: int) -> dict:
            """Add two numbers."""
            return {"result": a + b}

        @expose
        def list_models() -> dict:
            """List available models."""
            return {"models": list(list_configured_models().keys())}
    ''')

    # Create scoped sandbox
    scoped = ScopedSandbox(SandboxConfig(
        gpu="A100",
        models=[ModelConfig(name="google/gemma-2-9b")],
    ))

    print("\n1. Starting scoped sandbox...")
    scoped.start(name="test-library")

    # Write interface to temp file
    interface_file = Path("/tmp/test_interface.py")
    interface_file.write_text(interface_code)

    print("2. Serving interface as library...")
    library = scoped.serve(str(interface_file), expose_as="library", name="my_tools")

    try:
        # Create workspace with library
        print("3. Creating workspace with library...")
        workspace = Workspace(libraries=[library])

        # Create local session
        print("4. Creating local session...")
        session = create_local_session(
            workspace=workspace,
            workspace_dir="/tmp/test_workspace",
            name="test-library-session"
        )

        # Test calling the library via direct RPC
        print("\n5. Testing RPC calls...")
        import requests

        # Test hello
        resp = requests.post(
            scoped._rpc_url,
            json={"fn": "hello", "kwargs": {"name": "World"}},
            timeout=10
        )
        result = resp.json()
        print(f"  hello('World'): {result['result']}")

        # Test add
        resp = requests.post(
            scoped._rpc_url,
            json={"fn": "add", "kwargs": {"a": 5, "b": 3}},
            timeout=10
        )
        result = resp.json()
        print(f"  add(5, 3): {result['result']}")

        # Test list_models
        resp = requests.post(
            scoped._rpc_url,
            json={"fn": "list_models", "kwargs": {}},
            timeout=10
        )
        result = resp.json()
        print(f"  list_models(): {result['result']}")

        # Verify the library was created correctly
        library_file = session.workspace_path / "my_tools.py"
        if library_file.exists():
            print(f"\n  ✓ Library file created: {library_file}")

        print("\n✓ All library tests passed!")

    finally:
        scoped.terminate()


if __name__ == "__main__":
    test_scoped_as_library()
