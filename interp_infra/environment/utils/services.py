"""Service management for sandboxes - Jupyter, Docker, code-server."""

import time
import requests
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..sandbox import Sandbox

from ._scripts import DOCKERD_SCRIPT, jupyter_startup_script, code_server_install_script


def start_jupyter(sandbox: "Sandbox", port: int = 8888):
    """Start Jupyter server in sandbox."""
    script = jupyter_startup_script(port).replace('"', '\\"')
    sandbox.exec(f'nohup python -c "{script}" > /var/log/jupyter.log 2>&1 &')


def start_docker_daemon(sandbox: "Sandbox"):
    """Start Docker daemon in sandbox."""
    sandbox._sandbox.open("/start-dockerd.sh", "w").write(DOCKERD_SCRIPT)
    sandbox.exec("chmod +x /start-dockerd.sh && nohup /start-dockerd.sh > /var/log/dockerd.log 2>&1 &")

    for _ in range(30):
        try:
            sandbox.exec("docker info > /dev/null 2>&1")
            return
        except RuntimeError:
            time.sleep(1)

    raise RuntimeError("Docker daemon failed to start")


def start_code_server(sandbox: "Sandbox", port: int = 8080):
    """Start code-server in sandbox."""
    sandbox.exec(f"{code_server_install_script()} > /var/log/code-server-install.log 2>&1")
    sandbox.exec(f'nohup code-server --bind-addr 0.0.0.0:{port} --auth none /workspace > /var/log/code-server.log 2>&1 &')


def wait_for_service(url: str, max_retries: int = 100):
    """Wait for HTTP service to be ready."""
    for _ in range(max_retries):
        try:
            if requests.get(url, timeout=5).status_code == 200:
                return
        except (requests.RequestException, ConnectionError):
            pass
        time.sleep(2)

    raise RuntimeError(f"Service at {url} failed to start")
