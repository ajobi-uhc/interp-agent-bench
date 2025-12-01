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


def load_model(model_name: str, model_info: dict):
    """Load a model and return (model, tokenizer)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = model_info["volume_path"]
    is_peft = model_info.get("is_peft", False)

    if is_peft:
        from peft import AutoPeftModelForCausalLM
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    return model, tokenizer


def serve(port: int, user_code: str):
    """Start RPC server with user code."""
    print(f"Starting RPC server on port {port}", file=sys.stderr)

    _exposed_functions = {}

    def expose(func):
        _exposed_functions[func.__name__] = func
        return func

    # Load models from env var
    models, tokenizers = {}, {}
    prepared = os.environ.get("PREPARED_MODELS", "{}")

    if prepared != "{}":
        print("Loading models...", file=sys.stderr)
        for name, info in json.loads(prepared).items():
            try:
                model, tokenizer = load_model(name, info)
                models[name] = model
                tokenizers[name] = tokenizer
                print(f"  Loaded: {name}", file=sys.stderr)
            except Exception as e:
                print(f"  Failed to load {name}: {e}", file=sys.stderr)
                models[name] = tokenizers[name] = None

    # Execute user code
    namespace = {
        "__name__": "__main__",
        "expose": expose,
        "models": models,
        "tokenizers": tokenizers
    }
    exec(user_code, namespace)

    if not _exposed_functions:
        raise RuntimeError("No functions exposed with @expose decorator")

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
