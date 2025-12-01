"""Scripts for sandbox services."""

DOCKERD_SCRIPT = '''#!/bin/bash
set -xe -o pipefail

dev=$(ip route show default | awk '/default/ {print $5}')
if [ -z "$dev" ]; then
    echo "Error: No default device found."
    ip route show
    exit 1
fi
addr=$(ip addr show dev "$dev" | grep -w inet | awk '{print $2}' | cut -d/ -f1)
if [ -z "$addr" ]; then
    echo "Error: No IP address found for device $dev."
    exit 1
fi

echo 1 > /proc/sys/net/ipv4/ip_forward
iptables-legacy -t nat -A POSTROUTING -o "$dev" -j SNAT --to-source "$addr" -p tcp
iptables-legacy -t nat -A POSTROUTING -o "$dev" -j SNAT --to-source "$addr" -p udp

update-alternatives --set iptables /usr/sbin/iptables-legacy
update-alternatives --set ip6tables /usr/sbin/ip6tables-legacy

exec /usr/bin/dockerd --iptables=false --ip6tables=false -D
'''


def jupyter_startup_script(port: int) -> str:
    """Generate Jupyter startup script."""
    return f'''
import sys
sys.path.insert(0, "/root")
from scribe.notebook.notebook_server import ScribeServerApp
app = ScribeServerApp()
app.initialize([
    "--ip=0.0.0.0",
    "--port={port}",
    "--ServerApp.token=",
    "--ServerApp.password=",
    "--ServerApp.allow_root=True",
])
app.start()
'''


def code_server_install_script() -> str:
    """Generate code-server install script."""
    return "curl -fsSL https://code-server.dev/install.sh | sh"
