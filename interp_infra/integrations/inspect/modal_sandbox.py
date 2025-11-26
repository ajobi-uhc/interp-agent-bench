"""Modal Sandbox provider for Inspect AI.

This allows Inspect evaluations to use Modal Sandbox instead of Docker
for code execution isolation.
"""

import base64
from pathlib import Path

import modal
import yaml
from inspect_ai.util import ExecResult, sandboxenv, SandboxEnvironment


@sandboxenv(name="modal")
class ModalSandboxEnvironment(SandboxEnvironment):
    """Sandbox environment that uses Modal Sandbox for isolation."""

    def __init__(
        self,
        task_name: str,
        config: str | None = None,
        **kwargs
    ):
        """
        Initialize Modal sandbox environment.

        Args:
            task_name: Name of the task
            config: Path to Dockerfile, compose.yaml, or directory containing Dockerfile
            **kwargs: Additional arguments
        """
        super().__init__()

        self.task_name = task_name
        self.config = config

        # Find and build image from Dockerfile
        dockerfile_path = self._find_dockerfile(config)
        self._image = modal.Image.from_dockerfile(
            str(dockerfile_path),
            context_dir=str(dockerfile_path.parent)
        )

        # Create Modal app for this task
        self._app = modal.App.lookup(
            f"inspect-sandbox-{task_name}",
            create_if_missing=True
        )

        # Sandbox instance (created on first use)
        self._sandbox: modal.Sandbox | None = None
        self._sandbox_started = False

    @staticmethod
    def _find_dockerfile(config: str | None) -> Path:
        """
        Find Dockerfile from config path.

        Args:
            config: Path to Dockerfile, compose.yaml, or directory.
                   If None, looks for Dockerfile in current working directory.

        Returns:
            Absolute path to Dockerfile

        Raises:
            ValueError: If config is invalid
            FileNotFoundError: If Dockerfile cannot be found
        """
        if not config:
            # Fallback: look for Dockerfile in current working directory
            cwd = Path.cwd()
            dockerfile = cwd / "Dockerfile"
            if dockerfile.exists():
                return dockerfile
            raise FileNotFoundError(
                f"No config provided and no Dockerfile found in current directory: {cwd}"
            )

        config_path = Path(config).resolve()

        if not config_path.exists():
            raise FileNotFoundError(f"Config path does not exist: {config_path}")

        # Case 1: compose.yaml - parse to find Dockerfile
        if config_path.name in ("compose.yaml", "compose.yml", "docker-compose.yaml", "docker-compose.yml"):
            return ModalSandboxEnvironment._parse_compose_for_dockerfile(config_path)

        # Case 2: Direct Dockerfile path
        elif config_path.name == "Dockerfile" or config_path.name.startswith("Dockerfile."):
            return config_path

        # Case 3: Directory - look for Dockerfile inside
        elif config_path.is_dir():
            dockerfile = config_path / "Dockerfile"
            if not dockerfile.exists():
                raise FileNotFoundError(f"Dockerfile not found in directory: {config_path}")
            return dockerfile

        else:
            raise ValueError(f"Invalid config: {config_path} (expected Dockerfile, compose.yaml, or directory)")

    @staticmethod
    def _parse_compose_for_dockerfile(compose_path: Path) -> Path:
        """
        Parse compose.yaml to find Dockerfile path.

        Args:
            compose_path: Path to compose.yaml file

        Returns:
            Absolute path to Dockerfile

        Raises:
            FileNotFoundError: If no Dockerfile found in compose config
        """
        with open(compose_path) as f:
            compose = yaml.safe_load(f)

        # Find the first service with a build context
        services = compose.get("services", {})

        for service_name, service_config in services.items():
            build_config = service_config.get("build")

            if not build_config:
                continue

            # Handle string format: build: ./path/to/context
            if isinstance(build_config, str):
                context_dir = compose_path.parent / build_config
                dockerfile = context_dir / "Dockerfile"

            # Handle dict format: build: { context: ./path, dockerfile: Dockerfile }
            elif isinstance(build_config, dict):
                context = build_config.get("context", ".")
                dockerfile_name = build_config.get("dockerfile", "Dockerfile")
                context_dir = compose_path.parent / context
                dockerfile = context_dir / dockerfile_name

            else:
                continue

            if dockerfile.exists():
                return dockerfile.resolve()

        raise FileNotFoundError(
            f"No Dockerfile found in compose.yaml at {compose_path}. "
            f"Ensure at least one service has a 'build' configuration."
        )

    async def _ensure_sandbox(self) -> modal.Sandbox:
        """Create sandbox on first use."""
        if not self._sandbox_started:
            # Create long-running sandbox with sleep infinity
            self._sandbox = modal.Sandbox.create(
                "sleep", "infinity",
                image=self._image,
                app=self._app,
                timeout=3600 * 2,  # 2 hour timeout
            )
            self._sandbox_started = True

        return self._sandbox

    async def exec(
        self,
        cmd: list[str],
        input: str | bytes | None = None,
        cwd: str | None = None,
        env: dict[str, str] = {},
        user: str | None = None,
        timeout: int | None = None,
        timeout_retry: bool = True,
        concurrency: bool = True,
    ) -> ExecResult[str]:
        """
        Execute command in Modal sandbox.

        Args:
            cmd: Command and arguments to execute
            input: Optional stdin input
            cwd: Working directory (relative paths converted to /sandbox/<path>)
            env: Environment variables
            user: User to run as (not supported, will warn)
            timeout: Execution timeout in seconds
            timeout_retry: Whether to retry on timeout
            concurrency: Whether to allow concurrent execution

        Returns:
            ExecResult with stdout, stderr, return code
        """
        sandbox = await self._ensure_sandbox()

        # Warn if user specified (not supported in Modal)
        if user:
            print(f"Warning: 'user' parameter not supported in Modal sandbox (requested: {user})")

        # Build command with working directory
        if cwd:
            # Ensure absolute path
            if not cwd.startswith('/'):
                cwd = f"/sandbox/{cwd}"

            # Wrap command to cd first
            shell_cmd = f"cd {cwd} && {' '.join(cmd)}"
            full_cmd = ["bash", "-c", shell_cmd]
        else:
            full_cmd = cmd

        # Execute in sandbox
        process = sandbox.exec(
            *full_cmd,
            timeout=timeout or 120,
        )

        # Collect output
        stdout_lines = []
        stderr_lines = []

        for line in process.stdout:
            stdout_lines.append(line)

        for line in process.stderr:
            stderr_lines.append(line)

        process.wait()

        # Return result
        return ExecResult(
            success=process.returncode == 0,
            returncode=process.returncode,
            stdout="".join(stdout_lines),
            stderr="".join(stderr_lines),
        )

    async def write_file(self, file: str, contents: str | bytes) -> None:
        """
        Write file to sandbox.

        Args:
            file: File path (relative or absolute)
            contents: File contents (text or binary)
        """
        sandbox = await self._ensure_sandbox()

        # Use path as-is - relative paths go to sandbox's working directory
        # Ensure parent directory exists if path has directories
        if '/' in file:
            parent_dir = str(Path(file).parent)
            mkdir_process = sandbox.exec("mkdir", "-p", parent_dir)
            mkdir_process.wait()

        # Write file
        if isinstance(contents, str):
            # Text mode - write directly
            contents_bytes = contents.encode('utf-8')
        else:
            # Binary mode
            contents_bytes = contents

        # Base64 encode and write via python
        encoded = base64.b64encode(contents_bytes).decode('ascii')
        write_script = f"""
import base64
with open('{file}', 'wb') as f:
    f.write(base64.b64decode('{encoded}'))
"""

        write_process = sandbox.exec("python3", "-c", write_script)
        write_process.wait()

        if write_process.returncode != 0:
            raise IOError(f"Failed to write file {file}")

    async def read_file(
        self,
        file: str,
        text: bool = True
    ) -> str | bytes:
        """
        Read file from sandbox.

        Args:
            file: File path (relative or absolute)
            text: If True, return str; if False, return bytes

        Returns:
            File contents as str or bytes
        """
        sandbox = await self._ensure_sandbox()

        # Use path as-is - relative paths use sandbox's working directory
        # Read file using base64 encoding for safe transport
        read_script = f"""
import base64
with open('{file}', 'rb') as f:
    print(base64.b64encode(f.read()).decode('ascii'))
"""

        read_process = sandbox.exec("python3", "-c", read_script)

        # Collect output
        output_lines = []
        for line in read_process.stdout:
            output_lines.append(line)

        read_process.wait()

        if read_process.returncode != 0:
            raise FileNotFoundError(f"File not found: {file}")

        # Decode from base64
        encoded = "".join(output_lines).strip()
        contents_bytes = base64.b64decode(encoded)

        if text:
            return contents_bytes.decode('utf-8')
        else:
            return contents_bytes

    @classmethod
    async def sample_init(
        cls,
        task_name: str,
        config: str | None,
        metadata: dict[str, str],
    ) -> dict[str, "SandboxEnvironment"]:
        """
        Initialize sandbox environments for a sample.

        Args:
            task_name: Name of task
            config: Path to Dockerfile, compose.yaml, or directory
            metadata: Sample metadata

        Returns:
            Dictionary mapping environment names to instances
        """
        return {"default": cls(task_name, config)}

    @classmethod
    async def sample_cleanup(
        cls,
        task_name: str,
        config: str | None,
        environments: dict[str, "SandboxEnvironment"],
        interrupted: bool,
    ) -> None:
        """
        Clean up sandbox environments.

        Args:
            task_name: Name of task
            config: Optional config
            environments: Dictionary of environment instances
            interrupted: Whether execution was interrupted
        """
        for env in environments.values():
            if isinstance(env, ModalSandboxEnvironment) and env._sandbox:
                try:
                    env._sandbox.terminate()
                except Exception as e:
                    print(f"Warning: Failed to terminate sandbox: {e}")
