"""Scoped sandbox for serving interfaces via RPC."""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

from .sandbox import Sandbox, SandboxConfig


@dataclass
class Proxy:
    """Proxy to a scoped sandbox's RPC interface."""
    url: str
    functions: list[str]
    timeout: int = 600

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

    def as_mcp_config(self, name: str = "interface") -> dict:
        """Generate MCP server configuration for this proxy."""
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
    """Sandbox that serves a scoped interface via RPC."""

    def __init__(self, config: SandboxConfig):
        config.execution_mode = None
        super().__init__(config)
        self._rpc_port: int = config.rpc_port
        self._proxy: Optional[Proxy] = None
        self._specified_functions: Optional[list[str]] = None
        self._serve_file_code: Optional[str] = None

    def _prepare_models(self):
        """Prepare models - idempotent."""
        if self._model_handles:
            return
        super()._prepare_models()

    def serve_file(self, path: str | Path, port: int = None, functions: list[str] = None) -> "ScopedSandbox":
        """Serve a Python file via RPC."""
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

        if is_serving:
            if not hasattr(self.config, 'encrypted_ports'):
                self.config.encrypted_ports = []
            self.config.encrypted_ports = [self._rpc_port]

        # Prepare models early to inject env vars before sandbox creation
        self._prepare_models()

        if self._model_handles:
            handles_data = {
                h.name: {
                    "volume_path": h.volume_path,
                    "is_peft": h.is_peft,
                    "base_model_path": h.base_model_path,
                    "hidden": h.hidden,
                    "var_name": h.var_name
                }
                for h in self._model_handles
            }
            self.config.env["PREPARED_MODELS"] = json.dumps(handles_data)

        super().start(name)

        if not is_serving:
            return self

        self._start_rpc_server()
        self._proxy = self.create_proxy(self._rpc_port, self._specified_functions)
        print(f"Proxy functions: {self._proxy.functions}")
        return self._proxy

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

    @property
    def proxy(self) -> Optional[Proxy]:
        return self._proxy

    def _start_rpc_server(self):
        """Start RPC server with user code."""
        from pathlib import Path
        rpc_server_path = Path(__file__).parent / "rpc_server.py"
        rpc_server_code = rpc_server_path.read_text()

        with self._sandbox.open("/root/rpc_server.py", "w") as f:
            f.write(rpc_server_code)

        with self._sandbox.open("/root/user_code.py", "w") as f:
            f.write(self._serve_file_code)

        self.exec(f"python /root/rpc_server.py {self._rpc_port} /root/user_code.py > /var/log/rpc.log 2>&1 &")
        time.sleep(2)

        self._wait_for_port(self._rpc_port)

    def _wait_for_port(self, port: int, max_retries: int = 60):
        """Wait for server on port."""
        for i in range(max_retries):
            try:
                url = self._sandbox.tunnels()[port].url
                if requests.get(url, timeout=5).ok:
                    return
            except Exception:
                if i % 5 == 0:
                    print(f"Waiting for server... ({i+1}/{max_retries})")
                time.sleep(5)

        logs = self.exec('cat /var/log/rpc.log 2>/dev/null || echo "No logs"')
        raise RuntimeError(f"Server on port {port} failed to start.\n\nLogs:\n{logs}")
