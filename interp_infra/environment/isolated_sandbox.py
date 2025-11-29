"""Isolated sandbox for serving interfaces via RPC."""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

from .sandbox import Sandbox, SandboxConfig, ExecutionMode


@dataclass
class Proxy:
    """Proxy to an isolated sandbox's RPC interface."""
    url: str
    functions: list[str]
    
    def call(self, fn: str, *args, **kwargs):
        """Call a function on the isolated sandbox."""
        resp = requests.post(
            self.url,
            json={"fn": fn, "args": args, "kwargs": kwargs},
            timeout=600,
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


class IsolatedSandbox(Sandbox):
    """
    Sandbox that serves an isolated interface via RPC.
    
    Usage:
        # Serve a file
        isolated = IsolatedSandbox(SandboxConfig())
        isolated.serve_file("./interface.py")
        proxy = isolated.start()
        
        # Serve a repo
        isolated = IsolatedSandbox(SandboxConfig())
        isolated.serve_repo("github.com/org/api", install="pip install -e .", run="python server.py", functions=["chat"])
        proxy = isolated.start()
        
        # Bare mode
        isolated = IsolatedSandbox(SandboxConfig())
        isolated.start()
        isolated.exec("python my_server.py &")
        proxy = isolated.create_proxy(port=8080, functions=["chat"])
    """
    
    def __init__(self, config: SandboxConfig):
        config.execution_mode = ExecutionMode.CLI
        super().__init__(config)
        
        self._rpc_port: int = 8080
        self._proxy: Optional[Proxy] = None
        self._specified_functions: Optional[list[str]] = None
        self._serve_file_code: Optional[str] = None
        self._serve_repo_config: Optional[dict] = None
    
    def serve_file(self, path: str | Path, port: int = 8080, functions: list[str] = None) -> "IsolatedSandbox":
        """Serve a Python file via RPC. Top-level functions become the interface."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        self._serve_file_code = path.read_text()
        self._rpc_port = port
        self._specified_functions = functions
        return self
    
    def serve_code(self, code: str, port: int = 8080, functions: list[str] = None) -> "IsolatedSandbox":
        """Serve inline Python code via RPC."""
        self._serve_file_code = code
        self._rpc_port = port
        self._specified_functions = functions
        return self
    
    def serve_repo(
        self,
        url: str,
        install: str = None,
        run: str = None,
        interface: str = None,
        port: int = 8080,
        functions: list[str] = None,
    ) -> "IsolatedSandbox":
        """Clone a repo and serve via RPC."""
        if not run and not interface:
            raise ValueError("Provide run= or interface=")
        if run and not functions:
            raise ValueError("run= requires functions=")
        
        self._serve_repo_config = {"url": url, "install": install, "run": run, "interface": interface}
        self._rpc_port = port
        self._specified_functions = functions
        return self
    
    def start(self, name: str = "isolated") -> "Proxy | IsolatedSandbox":
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
        server_code = self._build_rpc_server(self._serve_file_code)
        
        with self._sandbox.open("/root/rpc_server.py", "w") as f:
            f.write(server_code)
        
        self.exec("nohup python /root/rpc_server.py > /var/log/rpc.log 2>&1 &")
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
            code = f'import sys; sys.path.insert(0, "{repo_path}"); exec(open("{repo_path}/{cfg["interface"]}").read())'
            server_code = self._build_rpc_server(code)
            with self._sandbox.open("/root/rpc_server.py", "w") as f:
                f.write(server_code)
            self.exec("nohup python /root/rpc_server.py > /var/log/rpc.log 2>&1 &")
        
        self._wait_for_port(self._rpc_port)
    
    def _build_rpc_server(self, user_code: str) -> str:
        """Build RPC server script."""
        return f'''
import json, traceback
from http.server import HTTPServer, BaseHTTPRequestHandler

{user_code}

_functions = {{n: o for n, o in globals().items() if callable(o) and not n.startswith("_") and not isinstance(o, type) and getattr(o, "__module__", None) == "__main__"}}
print(f"Functions: {{list(_functions.keys())}}")

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({{"functions": list(_functions.keys())}}).encode())
    
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        try:
            result = _functions[body["fn"]](*body.get("args", []), **body.get("kwargs", {{}}))
            resp = {{"ok": True, "result": result}}
        except Exception as e:
            resp = {{"ok": False, "error": str(e), "traceback": traceback.format_exc()}}
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(resp).encode())
    
    def log_message(self, *a): pass

print(f"RPC on {self._rpc_port}")
HTTPServer(("0.0.0.0", {self._rpc_port}), Handler).serve_forever()
'''
    
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
        
        return Proxy(url=url, functions=functions)
    
    def _wait_for_port(self, port: int, max_retries: int = 60):
        """Wait for server on port."""
        for _ in range(max_retries):
            try:
                url = self._sandbox.tunnels()[port].url
                if requests.get(url, timeout=5).ok:
                    return
            except:
                pass
            time.sleep(1)
        raise RuntimeError(f"Server on {port} failed. Logs:\n{self.exec('cat /var/log/rpc.log /var/log/server.log 2>/dev/null || true')}")
    
    @property
    def proxy(self) -> Optional[Proxy]:
        return self._proxy