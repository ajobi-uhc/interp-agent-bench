"""Scoped sandbox for serving code via RPC."""

import json
import time
from pathlib import Path
from typing import Optional, Literal

import requests

from .sandbox import Sandbox, SandboxConfig
from .utils.codegen import generate_rpc_client, generate_rpc_prompt
from ..workspace import Library, Skill, Workspace
from ..harness.logging import get_logger

logger = get_logger("scoped_sandbox")


ExposeMode = Literal["mcp", "library", "prompt", "skill"]


class ScopedSandbox(Sandbox):
    """
    Sandbox that serves code via RPC.

    Setup identical to regular Sandbox, but can serve code via RPC
    and expose it as MCP tools, importable library, or prompt docs.

    Example:
        # 1. Create scoped sandbox with its workspace
        sandbox = ScopedSandbox(SandboxConfig(gpu="A100", models=[...]))

        sandbox_workspace = Workspace(
            libraries=[Library.from_file("sae_utils.py")]  # Helpers for RPC code
        )
        sandbox.start(workspace=sandbox_workspace)

        # 2. Serve code
        interp_lib = sandbox.serve("interp_tools.py", expose_as="library")

        # 3. Use in agent's workspace (separate!)
        agent_workspace = Workspace(libraries=[interp_lib])
        session = create_local_session(workspace=agent_workspace)
    """

    def __init__(self, config: SandboxConfig):
        # ScopedSandbox doesn't use execution modes (no jupyter/cli)
        # RPC server is started separately
        if config.execution_mode is not None:
            config.execution_mode = None

        super().__init__(config)
        self._rpc_port: int = config.rpc_port
        self._rpc_url: Optional[str] = None
        self._rpc_process = None

    def _sanitize_name(self, name: str) -> str:
        """Convert model name to valid env var name."""
        # google/gemma-2-9b → GOOGLE_GEMMA_2_9B
        return name.replace("/", "_").replace("-", "_").upper()

    def _setup_models(self):
        """Setup models and set env vars for RPC code access."""
        # Only setup once
        if self.model_handles:
            return

        super()._setup_models()

        # Set environment variables for model paths
        # RPC code can access these via os.environ
        for handle in self.model_handles:
            env_key = f"MODEL_{self._sanitize_name(handle.name)}_PATH"
            self.config.env[env_key] = handle.volume_path

            # If PEFT, also set base model path
            if handle.is_peft and handle.base_model_path:
                base_env_key = f"MODEL_{self._sanitize_name(handle.base_model)}_PATH"
                self.config.env[base_env_key] = handle.base_model_path

    def start(
        self,
        workspace: Optional[Workspace] = None,
        name: str = "scoped",
    ) -> "ScopedSandbox":
        """
        Start scoped sandbox with its workspace.

        This is the workspace for the RPC code (libraries, data it needs),
        NOT the agent's workspace.

        Args:
            workspace: Workspace for scoped sandbox (what RPC code needs)
            name: Sandbox name

        Returns:
            Self for chaining
        """
        # Configure RPC port
        if not hasattr(self.config, 'encrypted_ports'):
            self.config.encrypted_ports = []
        self.config.encrypted_ports = [self._rpc_port]

        # Start sandbox (will call _setup_models which sets env vars)
        super().start(name=name)

        # Apply workspace configuration
        if workspace:
            from ..execution import CLISession
            temp_session = CLISession(sandbox=self, session_id="setup", workspace_path=Path("/workspace"))
            temp_session.setup(workspace)

        return self

    def serve(
        self,
        code_file: str | Path,
        expose_as: ExposeMode = "mcp",
        name: str = "interface",
    ):
        """
        Serve code via RPC.

        Sandbox must be started first with start().

        Args:
            code_file: Python file with functions (use @expose decorator)
            expose_as: How to expose:
                - "mcp": Returns MCP server config dict
                - "library": Returns Library object (wraps RPC calls)
                - "prompt": Returns prompt string describing interface
                - "skill": Returns Skill object (generates SKILL.md)
            name: Interface/library/skill name

        Returns:
            dict | Library | str | Skill depending on expose_as

        Example:
            # As MCP tools
            mcp_config = sandbox.serve("tools.py", expose_as="mcp")
            run_agent(mcp_servers=[mcp_config], ...)

            # As importable library
            library = sandbox.serve("tools.py", expose_as="library")
            agent_workspace = Workspace(libraries=[library])
        """
        if not self._sandbox:
            raise RuntimeError("Sandbox not started. Call start() first.")

        # Read code
        code_path = Path(code_file)
        if not code_path.exists():
            raise FileNotFoundError(f"File not found: {code_file}")

        code = code_path.read_text()

        # Start RPC server
        self._start_rpc_server(code)

        # Return based on expose_as
        if expose_as == "mcp":
            return self._as_mcp(name, code)
        elif expose_as == "library":
            return self._as_library(name, code)
        elif expose_as == "prompt":
            return self._as_prompt(name, code)
        elif expose_as == "skill":
            return self._as_skill(name, code)
        else:
            raise ValueError(f"Invalid expose_as: {expose_as}. Must be 'mcp', 'library', 'prompt', or 'skill'")

    def _as_mcp(self, name: str, source_code: str) -> dict:
        """Return MCP server config."""
        from .utils.codegen import parse_exposed_functions

        # Parse function names from source
        functions = parse_exposed_functions(source_code)
        function_names = [f.name for f in functions]

        return {
            name: {
                "type": "stdio",
                "command": "python",
                "args": ["-m", "src.mcps.proxy"],
                "env": {
                    "RPC_URL": self._rpc_url,
                    "FUNCTIONS": json.dumps(function_names),
                }
            }
        }

    def _as_library(self, name: str, source_code: str) -> Library:
        """Return Library that wraps RPC calls."""
        client_code = generate_rpc_client(
            name=name,
            source_code=source_code,
            rpc_url=self._rpc_url,
        )

        # Generate documentation for the library functions
        docs = generate_rpc_prompt(
            name=name,
            source_code=source_code,
            rpc_url=self._rpc_url,
        )

        return Library(
            name=name,
            files={f"{name}.py": client_code},
            docs=docs,
        )

    def _as_prompt(self, name: str, source_code: str) -> str:
        """Return prompt describing the interface."""
        return generate_rpc_prompt(
            name=name,
            source_code=source_code,
            rpc_url=self._rpc_url,
        )

    def _as_skill(self, name: str, source_code: str) -> Skill:
        """Return Skill that generates SKILL.md for Claude discovery."""
        description = f"GPU-backed tools from {name}. Provides RPC access to functions running on GPU."

        # TODO: Extract description from source code docstring if present
        return Skill.from_source(
            name=name,
            description=description,
            source_code=source_code,
        )

    def _start_rpc_server(self, code: str):
        """Start RPC server with user code."""
        import threading

        rpc_server_path = Path(__file__).parent / "utils" / "rpc_server.py"
        rpc_server_code = rpc_server_path.read_text()

        with self._sandbox.open("/root/rpc_server.py", "w") as f:
            f.write(rpc_server_code)

        with self._sandbox.open("/root/user_code.py", "w") as f:
            f.write(code)

        logger.info("Starting RPC server...")

        # Start RPC server process (2>&1 merges stderr into stdout for logging)
        self._rpc_process = self._sandbox.exec(
            "bash", "-c", f"python -u /root/rpc_server.py {self._rpc_port} /root/user_code.py 2>&1",
        )

        # Stream logs in background thread
        def stream_logs():
            try:
                for line in self._rpc_process.stdout:
                    print(f"[RPC] {line}", end="", flush=True)
            except:
                pass

        self._log_thread = threading.Thread(target=stream_logs, daemon=True)
        self._log_thread.start()

        time.sleep(2)
        self._wait_for_port(self._rpc_port)

    def _wait_for_port(self, port: int, max_retries: int = 100):
        """Wait for RPC server to be ready."""
        logger.info(f"Waiting for RPC server on port {port}...")

        for i in range(max_retries):
            try:
                url = self._sandbox.tunnels()[port].url
                response = requests.get(url, timeout=3)
                if response.ok:
                    self._rpc_url = url
                    logger.info(f"✓ RPC server ready: {url}")
                    return
            except requests.exceptions.RequestException:
                pass
            except Exception:
                pass

            time.sleep(3)

        raise RuntimeError(f"RPC server failed to start within {max_retries * 3}s")

