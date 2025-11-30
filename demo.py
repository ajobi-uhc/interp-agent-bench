import os
import tempfile

import modal

# Use the 2025.06 Modal Image Builder which avoids the need to install Modal client
# dependencies into the container image.

os.environ["MODAL_IMAGE_BUILDER_VERSION"] = "2025.06"


# Create an image for the parent Modal container.
# We install various Docker basics and a script to start the Docker daemon.
def create_modal_container_image(start_dockerd_filename: str):
    image = (
        modal.Image.from_registry("ubuntu:22.04")
        .env({"DEBIAN_FRONTEND": "noninteractive"})
        .apt_install(["wget", "ca-certificates", "curl", "net-tools", "iproute2"])
        .run_commands(
            [
                "install -m 0755 -d /etc/apt/keyrings",
                "curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc",
                "chmod a+r /etc/apt/keyrings/docker.asc",
                'echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo \\"${UBUNTU_CODENAME:-$VERSION_CODENAME}\\") stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null',
                "mkdir /build",
            ]
        )
        .apt_install([
            "docker-ce=5:27.5.0-1~ubuntu.22.04~jammy",
            "docker-ce-cli=5:27.5.0-1~ubuntu.22.04~jammy",
            "containerd.io",
            "docker-buildx-plugin",
            "docker-compose-plugin"
        ])
        # Ensure that our runc installation is modern.
        # We need this relatively-recent runc patch to ensure reliable networking in Docker:
        # https://github.com/opencontainers/runc/commit/491326cdeb3762a8b5f926be9bb5ddd36115e31d.
        .run_commands(
            [
                "rm $(which runc)",
                "wget https://github.com/opencontainers/runc/releases/download/v1.3.0/runc.amd64",
                "chmod +x runc.amd64",
                "mv runc.amd64 /usr/local/bin/runc",
            ]
        )
        # gVisor doesn't support nftables yet (https://github.com/google/gvisor/issues/10510).
        # Explicitly ensure that we use iptables-legacy -- the non-nftables version of iptables.
        .run_commands(
            [
                "update-alternatives --set iptables /usr/sbin/iptables-legacy",
                "update-alternatives --set ip6tables /usr/sbin/ip6tables-legacy",
            ]
        )
        .add_local_file(start_dockerd_filename, "/start-dockerd.sh", copy=True)
        .run_commands(["chmod +x /start-dockerd.sh"])
    )
    return image


start_dockerd_sh_content = """#!/bin/bash
set -xe -o pipefail

dev=$(ip route show default | awk '/default/ {print $5}')
if [ -z "$dev" ]; then
    echo "Error: No default device found."
    ip route show
    exit 1
else
    echo "Default device: $dev"
fi
addr=$(ip addr show dev "$dev" | grep -w inet | awk '{print $2}' | cut -d/ -f1)
if [ -z "$addr" ]; then
    echo "Error: No IP address found for device $dev."
    ip addr show dev "$dev"
    exit 1
else
    echo "IP address for $dev: $addr"
fi

echo 1 > /proc/sys/net/ipv4/ip_forward
iptables-legacy -t nat -A POSTROUTING -o "$dev" -j SNAT --to-source "$addr" -p tcp
iptables-legacy -t nat -A POSTROUTING -o "$dev" -j SNAT --to-source "$addr" -p udp

# gVisor doesn't support nftables yet (https://github.com/google/gvisor/issues/10510).
# Explicitly ensure that we use iptables-legacy -- the non-nftables version of iptables.
update-alternatives --set iptables /usr/sbin/iptables-legacy
update-alternatives --set ip6tables /usr/sbin/ip6tables-legacy

exec /usr/bin/dockerd --iptables=false --ip6tables=false -D"""


def main():
    print("Looking up modal.Sandbox app")
    app = modal.App.lookup("docker-test", create_if_missing=True)
    print("Creating sandbox")

    # Write the start-dockerd.sh content to a temporary local file.
    with tempfile.NamedTemporaryFile(mode="w", delete=True, encoding="utf-8") as start_dockerd_sh:
        print(f'Writing the "start dockerd" script to: {start_dockerd_sh.name}')
        start_dockerd_sh.write(start_dockerd_sh_content)
        start_dockerd_sh.flush()
        os.chmod(start_dockerd_sh.name, 0o755)

        with modal.enable_output():
            sb = modal.Sandbox.create(
                "/start-dockerd.sh",
                timeout=60 * 60,
                app=app,
                image=create_modal_container_image(start_dockerd_sh.name),
                experimental_options={"enable_docker": True},
            )

    # A simple Dockerfile that we'll build and run within Modal.
    dockerfile = """
    FROM ubuntu
    RUN apt-get update
    RUN apt-get install -y cowsay curl
    RUN mkdir -p /usr/share/cowsay/cows/
    RUN curl -o /usr/share/cowsay/cows/docker.cow https://raw.githubusercontent.com/docker/whalesay/master/docker.cow
    ENTRYPOINT ["/usr/games/cowsay", "-f", "docker.cow"]
    """
    with sb.open("/build/Dockerfile", "w") as f:
        f.write(dockerfile)

    print("Building docker image")
    p = sb.exec("docker", "build", "--network=host", "-t", "whalesay", "/build")
    for l in p.stdout:
        print(l, end="")
    p.wait()
    print("--------------------------------")
    if p.returncode != 0:
        print(p.stderr.read())
        raise Exception("Docker build failed")

    # Get the Sandbox to run the built image and show this:
    #
    #  ________
    # < Hello! >
    #  --------
    #     \
    #      \
    #       \
    #                     ##         .
    #               ## ## ##        ==
    #            ## ## ## ## ##    ===
    #        /"""""""""""""""""\___/ ===
    #       {                       /  ===-
    #        \______ O           __/
    #          \    \         __/
    #           \____\_______/

    print("Running Docker image")
    # Note we can't use -it here because we're not in a TTY.
    p = sb.exec("docker", "run", "--rm", "whalesay", "Hello!")
    print(p.stdout.read())
    p.wait()
    if p.returncode != 0:
        raise Exception(f"Docker run failed: {p.stderr.read()}")
    sb.terminate()


if __name__ == "__main__":
    main()