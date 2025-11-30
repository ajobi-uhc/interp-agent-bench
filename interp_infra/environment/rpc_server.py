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


def serve(port: int, user_code: str):
    """
    Start RPC server with user code.

    Args:
        port: Port to listen on
        user_code: Python code to execute (defines functions to serve)
    """
    try:
        print(f"[RPC] Starting server on port {port}", file=sys.stderr)

        # Registry for exposed functions
        _exposed_functions = {}

        # Define the @expose decorator
        def expose(func):
            """Decorator to mark functions for RPC exposure."""
            _exposed_functions[func.__name__] = func
            return func

        print(f"[RPC] Executing user code...", file=sys.stderr)

        # Execute user code with decorator available
        namespace = {
            "__name__": "__main__",
            "expose": expose,  # Inject decorator
        }
        exec(user_code, namespace)

        # Use exposed functions
        RPCHandler.functions = _exposed_functions

        if not _exposed_functions:
            raise RuntimeError("No functions exposed with @expose decorator")

        print(f"[RPC] Exposed functions: {list(_exposed_functions.keys())}", file=sys.stderr)

        # Start server
        print(f"[RPC] Server listening on 0.0.0.0:{port}", file=sys.stderr)
        HTTPServer(("0.0.0.0", port), RPCHandler).serve_forever()
    except Exception as e:
        print(f"[RPC] ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python rpc_server.py <port> <code_file>", file=sys.stderr)
        sys.exit(1)

    port = int(sys.argv[1])
    code_file = sys.argv[2]

    with open(code_file) as f:
        user_code = f.read()

    serve(port, user_code)
