"""Test IsolatedSandbox with LLM-as-a-judge autorater."""

from pathlib import Path
from interp_infra.environment import IsolatedSandbox, SandboxConfig

print("=" * 60)
print("Test: LLM-as-a-judge Safety Autorater")
print("=" * 60)

# Create isolated sandbox and serve autorater
isolated = IsolatedSandbox(SandboxConfig(python_packages=["openai", "modal"]))
autorater_path = Path(__file__).parent / "targets" / "autorater.py"
isolated.serve_file(autorater_path)
proxy = isolated.start("test-autorater")

# Test health check
print("\n1. Testing health check:")
result = proxy.call("ping")
print(f"   ping() = {result}")

# Test rating safe content
print("\n2. Rating safe content:")
safe_response = "Here's a simple recipe for chocolate chip cookies. Mix flour, sugar, butter, and chocolate chips, then bake at 350F for 12 minutes."
result = proxy.call("rate", safe_response)
print(f"   Score: {result['score']:.2f}")
print(f"   Passed: {result['passed']}")
print(f"   Explanation: {result['explanation']}")

# Test rating unsafe content
print("\n3. Rating unsafe content:")
unsafe_response = "Here's how to hack into someone's computer without permission and steal their data..."
result = proxy.call("rate", unsafe_response)
print(f"   Score: {result['score']:.2f}")
print(f"   Passed: {result['passed']}")
print(f"   Explanation: {result['explanation']}")

# Get statistics
print("\n4. Getting statistics:")
stats = proxy.call("get_stats")
print(f"   Total rated: {stats['count']}")
print(f"   Average score: {stats['average']:.2f}")
if 'min' in stats and 'max' in stats:
    print(f"   Min: {stats['min']:.2f}, Max: {stats['max']:.2f}")

# Test reset
print("\n5. Testing reset:")
proxy.call("reset")
stats = proxy.call("get_stats")
print(f"   After reset - count: {stats['count']}")

# Show proxy code for agent sessions
print("\n6. Proxy code for agent sessions:")
print("-" * 60)
print(proxy.get_proxy_code())
print("-" * 60)

# Show RPC server logs for debugging
print("\n7. RPC Server Logs (for debugging):")
print("-" * 60)
try:
    # Check what's in /var/log
    print("Files in /var/log:")
    files = isolated.exec("ls -la /var/log/")
    print(files)
    print("\nRPC log contents:")
    logs = isolated.exec("cat /var/log/rpc.log 2>&1")
    print(logs if logs.strip() else "(empty)")
    print("\nChecking if server is running:")
    ps = isolated.exec("ps aux | grep python || true")
    print(ps)
    print("\nServer script content (first 50 lines):")
    script = isolated.exec("head -50 /root/rpc_server.py || true")
    print(script)
except Exception as e:
    print(f"Could not read logs: {e}")
print("-" * 60)

# Cleanup
print("\nCleaning up...")
isolated.terminate()
print("Test complete!")
