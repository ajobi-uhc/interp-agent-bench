"""RPC server for serving Python functions over HTTP."""

import json
import sys
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Callable, Any


class RPCHandler(BaseHTTPRequestHandler):
    """HTTP handler for RPC requests."""

    functions: Dict[str, Callable] = {}

    def do_GET(self):
        """List available functions."""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"functions": list(self.functions.keys())}).encode())

    def do_POST(self):
        """Execute a function call."""
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))

        try:
            fn = body["fn"]
            args = body.get("args", [])
            kwargs = body.get("kwargs", {})

            if fn not in self.functions:
                raise ValueError(f"Function '{fn}' not found")

            result = self.functions[fn](*args, **kwargs)
            resp = {"ok": True, "result": result}
        except Exception as e:
            resp = {"ok": False, "error": str(e), "traceback": traceback.format_exc()}

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(resp).encode())

    def log_message(self, *args):
        """Silence request logging."""
        pass


def extract_functions(namespace: Dict[str, Any]) -> Dict[str, Callable]:
    """
    Extract functions marked with @expose decorator.

    Args:
        namespace: Global namespace dict

    Returns:
        Dict of function name to function object
    """
    from interp_infra.environment.interface import get_exposed_functions
    functions = get_exposed_functions()

    if not functions:
        raise RuntimeError("No functions exposed with @expose decorator")

    print(f"Exposed functions: {list(functions.keys())}", file=sys.stderr)
    return functions


def serve(port: int, user_code: str):
    """
    Start RPC server with user code.

    Args:
        port: Port to listen on
        user_code: Python code to execute (defines functions to serve)
    """
    # Execute user code in a clean namespace
    namespace = {"__name__": "__main__"}
    exec(user_code, namespace)

    # Extract functions
    RPCHandler.functions = extract_functions(namespace)
    print(f"Functions: {list(RPCHandler.functions.keys())}", file=sys.stderr)

    # Start server
    print(f"RPC on {port}", file=sys.stderr)
    HTTPServer(("0.0.0.0", port), RPCHandler).serve_forever()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python rpc_server.py <port> <code_file>", file=sys.stderr)
        sys.exit(1)

    port = int(sys.argv[1])
    code_file = sys.argv[2]

    with open(code_file) as f:
        user_code = f.read()

    serve(port, user_code)
