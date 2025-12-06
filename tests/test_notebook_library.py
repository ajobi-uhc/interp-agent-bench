"""Test library installation in notebook session."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.environment import Sandbox, SandboxConfig, ExecutionMode
from src.workspace import Workspace, Library
from src.execution import create_notebook_session


async def test_notebook_library():
    """Test that libraries can be imported in notebook session."""
    print("\n" + "=" * 60)
    print("Test: Notebook Session with Library")
    print("=" * 60)

    # Create a simple test library
    test_code = '''"""Test library for notebook import."""

def greet(name: str) -> str:
    """Return a greeting message."""
    return f"Hello, {name}!"

def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b
'''

    test_library = Library(
        name="test_utils",
        files={"test_utils.py": test_code},
        docs="Simple test library with greet() and add() functions."
    )

    print("\n1. Creating test library...")
    print(f"  Name: {test_library.name}")
    print(f"  Files: {list(test_library.files.keys())}")
    print(f"  Is single file: {test_library.is_single_file}")

    # Create workspace with library
    workspace = Workspace(libraries=[test_library])

    # Create sandbox with notebook mode
    print("\n2. Creating sandbox (CPU only)...")
    sandbox = Sandbox(SandboxConfig(
        execution_mode=ExecutionMode.NOTEBOOK,
        python_packages=["numpy"],  # Minimal packages
    ))
    sandbox.start(name="test-notebook-library")
    print(f"  ✓ Sandbox ready")
    print(f"  Jupyter: {sandbox.jupyter_url}")

    try:
        # Create notebook session
        print("\n3. Creating notebook session with library...")
        session = create_notebook_session(
            sandbox=sandbox,
            workspace=workspace,
            name="test-library-session"
        )
        print(f"  ✓ Session ready: {session.session_id}")

        # Check if library file exists
        print("\n4. Checking if library file exists...")
        result = session.exec("""
import os
library_path = '/workspace/test_utils.py'
exists = os.path.exists(library_path)
if exists:
    with open(library_path, 'r') as f:
        size = len(f.read())
    print(f"✓ Library file exists: {library_path} ({size} bytes)")
else:
    print(f"✗ Library file NOT found: {library_path}")
print(f"Files in /workspace: {os.listdir('/workspace')}")
exists
""")

        # Find execute_result output (not stream output)
        file_exists = False
        for output in result.get("outputs", []):
            if output.get("output_type") == "execute_result":
                file_exists = output.get("data", {}).get("text/plain") == "True"
                break

        # Check sys.path
        print("\n5. Checking sys.path...")
        result = session.exec("""
import sys
workspace_in_path = '/workspace' in sys.path
print(f"'/workspace' in sys.path: {workspace_in_path}")
print(f"sys.path: {sys.path[:3]}...")
workspace_in_path
""")

        # Try importing the library
        print("\n6. Testing library import...")
        result = session.exec("""
success = False
try:
    from test_utils import greet, add

    # Test the functions
    greeting = greet("World")
    sum_result = add(2, 3)

    print(f"✓ Successfully imported test_utils")
    print(f"  greet('World') = {greeting}")
    print(f"  add(2, 3) = {sum_result}")

    success = greeting == "Hello, World!" and sum_result == 5
    print(f"✓ Functions work correctly: {success}")
except ImportError as e:
    print(f"✗ Failed to import: {e}")
except Exception as e:
    print(f"✗ Error: {e}")

success
""")

        print(f"  Import result outputs: {result.get('outputs', [])}")

        # Find execute_result output
        import_success = False
        for output in result.get("outputs", []):
            if output.get("output_type") == "execute_result":
                result_value = output.get("data", {}).get("text/plain")
                print(f"  Found execute_result: {result_value}")
                import_success = result_value == "True"
                break
            elif output.get("output_type") == "error":
                print(f"  ✗ Error during import: {output.get('ename')}: {output.get('evalue')}")

        print("\n" + "=" * 60)
        if import_success:
            print("✓ TEST PASSED: Library successfully loaded and used!")
        else:
            print("✗ TEST FAILED: Could not import or use library")
            print(f"File exists: {file_exists}")
        print("=" * 60)

    finally:
        print("\n7. Cleaning up...")
        sandbox.terminate()


if __name__ == "__main__":
    try:
        asyncio.run(test_notebook_library())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
