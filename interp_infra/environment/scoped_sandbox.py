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
    
    def __init__(self, config: SandboxConfig):
        config.execution_mode = None  # ScopedSandbox manages its own RPC server
        super().__init__(config)

        self._rpc_port: int = config.rpc_port
        self._proxy: Optional[Proxy] = None
        self._specified_functions: Optional[list[str]] = None
        self._serve_file_code: Optional[str] = None
        self._serve_repo_config: Optional[dict] = None
    
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

    def serve_repo(
        self,
        url: str,
        install: str = None,
        run: str = None,
        interface: str = None,
        port: int = None,
        functions: list[str] = None,
    ) -> "ScopedSandbox":
        """Clone a repo and serve via RPC."""
        if not run and not interface:
            raise ValueError("Provide run= or interface=")
        if run and not functions:
            raise ValueError("run= requires functions=")

        self._serve_repo_config = {"url": url, "install": install, "run": run, "interface": interface}
        if port is not None:
            self._rpc_port = port
        self._specified_functions = functions
        return self
    
    def start(self, name: str = "scoped") -> "Proxy | ScopedSandbox":
        """Start sandbox. Returns Proxy if serving, self if bare mode."""
        is_serving = self._serve_file_code or self._serve_repo_config
        
        # Add RPC port to config before starting
        if is_serving:
            if not hasattr(self.config, 'encrypted_ports'):
                self.config.encrypted_ports = []
            self.config.encrypted_ports = [self._rpc_port]
        
        # Inject model paths as env var
        if self._model_handles:
            self.config.env["PREPARED_MODELS"] = json.dumps({h.name: h.volume_path for h in self._model_handles})
        
        super().start(name)
        
        if not is_serving:
            return self
        
        # Start server
        if self._serve_file_code:
            self._start_rpc_server()
        else:
            self._start_repo_server()
        
        self._proxy = self.create_proxy(self._rpc_port, self._specified_functions)
        print(f"  Proxy: {self._proxy.functions}")
        return self._proxy
    
    def _start_rpc_server(self):
        """Start RPC server with user code."""
        print("  Starting RPC server...")

        # Write user code
        with self._sandbox.open("/root/user_code.py", "w") as f:
            f.write(self._serve_file_code)

        # Use the proper RPC server module
        self.exec(
            f"nohup python -m interp_infra.environment.rpc_server {self._rpc_port} /root/user_code.py > /var/log/rpc.log 2>&1 &"
        )
        self._wait_for_port(self._rpc_port)
    
    def _start_repo_server(self):
        """Clone repo and start server."""
        cfg = self._serve_repo_config
        url = cfg["url"] if cfg["url"].startswith("http") else f"https://github.com/{cfg['url']}"
        repo_name = url.split("/")[-1].replace(".git", "")
        repo_path = f"/workspace/{repo_name}"

        print(f"  Cloning {repo_name}...")
        self.exec(f"git clone {url} {repo_path}")

        if cfg["install"]:
            print(f"  Installing...")
            self.exec(f"cd {repo_path} && {cfg['install']}")

        if cfg["run"]:
            print(f"  Running: {cfg['run']}")
            self.exec(f"cd {repo_path} && nohup {cfg['run']} > /var/log/server.log 2>&1 &")
        else:
            # Create user code that imports from repo
            code = f'import sys; sys.path.insert(0, "{repo_path}"); exec(open("{repo_path}/{cfg["interface"]}").read())'
            with self._sandbox.open("/root/user_code.py", "w") as f:
                f.write(code)
            self.exec(
                f"nohup python -m interp_infra.environment.rpc_server {self._rpc_port} /root/user_code.py > /var/log/rpc.log 2>&1 &"
            )

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

        for _ in range(max_retries):
            try:
                url = self._sandbox.tunnels()[port].url
                if requests.get(url, timeout=5).ok:
                    return
            except:
                pass
            time.sleep(5)
        raise RuntimeError(f"Server on {port} failed. Logs:\n{self.exec('cat /var/log/rpc.log /var/log/server.log 2>/dev/null || true')}")
    
    @property
    def proxy(self) -> Optional[Proxy]:
        return self._proxy