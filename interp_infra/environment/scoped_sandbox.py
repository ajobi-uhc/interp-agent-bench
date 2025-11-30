"""Scoped sandbox for serving interfaces via RPC."""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

from .sandbox import Sandbox, SandboxConfig, ExecutionMode


@dataclass
class Proxy:
    """Proxy to a scoped sandbox's RPC interface."""
    url: str
    functions: list[str]
    timeout: int = 600  # Default timeout for RPC calls

    def call(self, fn: str, *args, **kwargs):
        """Call a function on the scoped sandbox."""
        resp = requests.post(
            self.url,
            json={"fn": fn, "args": args, "kwargs": kwargs},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        
        if not data["ok"]:
            error = data.get("error", "Unknown error")
            if "traceback" in data:
                error += f"\n\nRemote traceback:\n{data['traceback']}"
            raise RuntimeError(error)
        
        return data["result"]
    
    def get_proxy_code(self) -> str:
        """Generate Python code for agent session."""
        lines = [
            "import requests as _proxy_requests",
            f'_PROXY_URL = "{self.url}"',
        ]

        for fn in self.functions:
            lines.append(f'''
def {fn}(*args, **kwargs):
    resp = _proxy_requests.post(_PROXY_URL, json={{"fn": "{fn}", "args": args, "kwargs": kwargs}}, timeout=600)
    data = resp.json()
    if not data["ok"]:
        raise RuntimeError(data.get("error", "Unknown error"))
    return data["result"]
''')

        return "\n".join(lines)

    def as_mcp_config(self, name: str = "interface") -> dict:
        """
        Generate MCP server configuration for this proxy.

        Returns a config that spawns an MCP server (locally) that wraps
        the RPC endpoint. The MCP server translates tool calls into HTTP
        requests to this proxy's RPC endpoint.

        Args:
            name: Name for the MCP server

        Returns:
            MCP configuration dict
        """
        return {
            name: {
                "type": "stdio",
                "command": "python",
                "args": ["-m", "interp_infra.mcps.proxy"],
                "env": {
                    "RPC_URL": self.url,
                    "FUNCTIONS": json.dumps(self.functions)
                }
            }
        }


class ScopedSandbox(Sandbox):
    """
    Sandbox that serves a scoped interface via RPC.

    Usage:
        # Serve a file
        scoped = ScopedSandbox(SandboxConfig())
        scoped.serve_file("./interface.py")
        proxy = scoped.start()

        # Serve a repo
        scoped = ScopedSandbox(SandboxConfig())
        scoped.serve_repo("github.com/org/api", install="pip install -e .", run="python server.py", functions=["chat"])
        proxy = scoped.start()

        # Bare mode
        scoped = ScopedSandbox(SandboxConfig())
        scoped.start()
        scoped.exec("python my_server.py &")
        proxy = scoped.create_proxy(port=8080, functions=["chat"])
    """

    def _prepare_models(self):
        """Override to make idempotent - skip if already prepared."""
        if self._models_prepared:
            return
        super()._prepare_models()
        self._models_prepared = True
    
    def __init__(self, config: SandboxConfig):
        config.execution_mode = None  # ScopedSandbox manages its own RPC server
        super().__init__(config)

        self._rpc_port: int = config.rpc_port
        self._proxy: Optional[Proxy] = None
        self._specified_functions: Optional[list[str]] = None
        self._serve_file_code: Optional[str] = None
        self._models_prepared: bool = False
    
    def serve_file(self, path: str | Path, port: int = None, functions: list[str] = None) -> "ScopedSandbox":
        """Serve a Python file via RPC. Top-level functions become the interface."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        self._serve_file_code = path.read_text()
        if port is not None:
            self._rpc_port = port
        self._specified_functions = functions
        return self

    def serve_code(self, code: str, port: int = None, functions: list[str] = None) -> "ScopedSandbox":
        """Serve inline Python code via RPC."""
        self._serve_file_code = code
        if port is not None:
            self._rpc_port = port
        self._specified_functions = functions
        return self

    def start(self, name: str = "scoped") -> "Proxy | ScopedSandbox":
        """Start sandbox. Returns Proxy if serving, self if bare mode."""
        is_serving = self._serve_file_code

        # Add RPC port to config before starting
        if is_serving:
            if not hasattr(self.config, 'encrypted_ports'):
                self.config.encrypted_ports = []
            self.config.encrypted_ports = [self._rpc_port]

        # Prepare models early so we can inject env vars before sandbox creation
        self._prepare_models()

        # Inject model paths as env var (must be BEFORE super().start() which creates sandbox)
        if self._model_handles:
            self.config.env["PREPARED_MODELS"] = json.dumps({h.name: h.volume_path for h in self._model_handles})

        super().start(name)
        
        if not is_serving:
            return self

        # Start RPC server
        self._start_rpc_server()
        self._proxy = self.create_proxy(self._rpc_port, self._specified_functions)
        print(f"  Proxy: {self._proxy.functions}")
        return self._proxy
    
    def _start_rpc_server(self):
        """Start RPC server with user code."""
        print("  Starting RPC server...")

        # Copy RPC server code to sandbox
        from pathlib import Path
        rpc_server_path = Path(__file__).parent / "rpc_server.py"
        rpc_server_code = rpc_server_path.read_text()

        with self._sandbox.open("/root/rpc_server.py", "w") as f:
            f.write(rpc_server_code)

        # Write user code
        with self._sandbox.open("/root/user_code.py", "w") as f:
            f.write(self._serve_file_code)

        # Run RPC server directly (not as module)
        self.exec(
            f"nohup python /root/rpc_server.py {self._rpc_port} /root/user_code.py > /var/log/rpc.log 2>&1 &"
        )

        # Give it a moment to start
        time.sleep(2)

        # Check for immediate errors
        try:
            initial_logs = self.exec('head -50 /var/log/rpc.log 2>/dev/null || echo "No logs yet"')
            if initial_logs and initial_logs != "No logs yet":
                print(f"  Initial RPC output:\n{initial_logs}")
        except:
            pass

        self._wait_for_port(self._rpc_port)

    def create_proxy(self, port: int, functions: list[str] = None) -> Proxy:
        """Create proxy to HTTP endpoint."""
        if not self._sandbox:
            raise RuntimeError("Sandbox not started")

        tunnels = self._sandbox.tunnels()
        if port not in tunnels:
            raise RuntimeError(f"Port {port} not exposed")

        url = tunnels[port].url

        if functions is None:
            resp = requests.get(url, timeout=10)
            functions = resp.json().get("functions", [])

        return Proxy(url=url, functions=functions, timeout=self.config.rpc_timeout)
    
    def _wait_for_port(self, port: int, max_retries: int = None):
        """Wait for server on port."""
        if max_retries is None:
            max_retries = self.config.wait_timeout // 5  # 5 second intervals

        print(f"  Waiting for server on port {port}...")
        for i in range(max_retries):
            try:
                url = self._sandbox.tunnels()[port].url
                resp = requests.get(url, timeout=5)
                if resp.ok:
                    print(f"  Server ready!")
                    return
            except Exception as e:
                if i % 3 == 0:  # Print every 15 seconds
                    print(f"  Still waiting... (attempt {i+1}/{max_retries})")
                    # Check logs periodically
                    try:
                        logs = self.exec('tail -20 /var/log/rpc.log 2>/dev/null || echo "No logs yet"')
                        if "error" in logs.lower() or "traceback" in logs.lower():
                            print(f"  Error detected in logs:\n{logs}")
                    except:
                        pass
                time.sleep(5)

        # Failed - get full logs
        print(f"  Server failed to start. Fetching logs...")
        logs = self.exec('cat /var/log/rpc.log 2>/dev/null || echo "No RPC logs found"')
        raise RuntimeError(f"Server on port {port} failed to start within {max_retries * 5}s.\n\nRPC Logs:\n{logs}")
    
    @property
    def proxy(self) -> Optional[Proxy]:
        return self._proxy