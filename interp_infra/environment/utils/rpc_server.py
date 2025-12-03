"""RPC server for serving Python functions over HTTP."""

import json
import os
import sys
import traceback
import inspect
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Callable, get_type_hints, get_origin


class RPCHandler(BaseHTTPRequestHandler):
    """HTTP handler for RPC requests."""

    functions: Dict[str, Callable] = {}
    schemas: Dict[str, dict] = {}

    def do_GET(self):
        """List available functions with schemas."""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({
            "functions": list(self.functions.keys()),
            "schemas": self.schemas
        }).encode())

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


TYPE_MAP = {
    int: "integer",
    float: "number",
    str: "string",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def python_type_to_json_type(hint) -> str:
    """Convert Python type hint to JSON schema type."""
    origin = get_origin(hint)
    check_type = origin if origin else hint

    for py_type, json_type in TYPE_MAP.items():
        if check_type is py_type:
            return json_type

    return "string"


def build_function_schema(func: Callable) -> dict:
    """Build JSON schema for a function."""
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or f"Call {func.__name__}()"

    properties = {}
    required = []

    try:
        hints = get_type_hints(func)
    except (NameError, AttributeError):
        hints = {}

    for param_name, param in sig.parameters.items():
        if param_name in ('self', 'cls'):
            continue

        param_type = python_type_to_json_type(hints.get(param_name, str))
        properties[param_name] = {"type": param_type}

        if param.default == inspect.Parameter.empty:
            required.append(param_name)

    return {
        "description": doc,
        "inputSchema": {
            "type": "object",
            "properties": properties,
            "required": required
        }
    }


def get_model_path(model_name: str) -> str:
    """Get path to a configured model from environment."""
    env_key = f"MODEL_{model_name.replace('/', '_').replace('-', '_').upper()}_PATH"
    path = os.environ.get(env_key)
    if not path:
        available = [k for k in os.environ.keys() if k.startswith("MODEL_") and k.endswith("_PATH")]
        raise ValueError(
            f"Model '{model_name}' not configured.\n"
            f"Expected env var: {env_key}\n"
            f"Available: {available}"
        )
    return path


def list_configured_models() -> dict:
    """List all configured models."""
    models = {}
    for key, value in os.environ.items():
        if key.startswith("MODEL_") and key.endswith("_PATH"):
            name = key[6:-5].lower().replace("_", "-")
            models[name] = value
    return models


def serve(port: int, user_code: str):
    """Start RPC server with user code."""
    try:
        print(f"Starting RPC server on port {port}", file=sys.stderr)
        sys.stderr.flush()

        _exposed_functions = {}

        def expose(func):
            """Decorator to mark functions for RPC exposure."""
            _exposed_functions[func.__name__] = func
            return func

        # Execute user code with minimal namespace
        # Inject: expose decorator + helper functions
        namespace = {
            "__name__": "__main__",
            "__builtins__": __builtins__,
            "expose": expose,
            "get_model_path": get_model_path,
            "list_configured_models": list_configured_models,
        }

        print("Executing user code...", file=sys.stderr)
        sys.stderr.flush()

        try:
            exec(user_code, namespace)
        except Exception as e:
            error_msg = f"ERROR executing user code: {e}\n{traceback.format_exc()}"
            print(error_msg, file=sys.stderr)
            sys.stderr.flush()

            # Also write to file so we can definitely read it
            try:
                with open("/tmp/rpc_error.log", "w") as f:
                    f.write(error_msg)
            except:
                pass

            raise

        print(f"User code executed successfully", file=sys.stderr)
        sys.stderr.flush()

        if not _exposed_functions:
            raise RuntimeError("No functions exposed with @expose decorator")

    except Exception as e:
        # Catch ALL errors and write to file
        error_msg = f"FATAL ERROR in serve(): {e}\n{traceback.format_exc()}"
        try:
            with open("/tmp/rpc_error.log", "w") as f:
                f.write(error_msg)
        except:
            pass
        print(error_msg, file=sys.stderr)
        sys.stderr.flush()
        sys.exit(1)

    # Build schemas
    schemas = {}
    for name, func in _exposed_functions.items():
        try:
            schemas[name] = build_function_schema(func)
        except Exception:
            schemas[name] = {
                "description": f"Call {name}()",
                "inputSchema": {"type": "object", "properties": {}}
            }

    RPCHandler.functions = _exposed_functions
    RPCHandler.schemas = schemas

    print(f"Exposed functions: {list(_exposed_functions.keys())}", file=sys.stderr)
    print(f"Listening on 0.0.0.0:{port}", file=sys.stderr)
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
