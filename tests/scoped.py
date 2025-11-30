"""ScopedSandbox serving an interface."""

from pathlib import Path
from interp_infra.environment import ScopedSandbox, SandboxConfig
from conftest import auto_cleanup

scoped = ScopedSandbox(SandboxConfig(python_packages=["openai"]))

autorater = Path("tests/targets/autorater.py")
scoped.serve_file(autorater)

with auto_cleanup(scoped):
    proxy = scoped.start("scoped")

    # Test proxy
    print(proxy.call("ping"))

    result = proxy.call("rate_safety", "Hello, how are you?")
    print(f"Safety score: {result['score']}")

    # Show MCP config
    print(f"\nMCP: {list(proxy.as_mcp_config().keys())}")
