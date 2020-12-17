"""
Build docker image, start container, then set its SSH service port to VSO variable "docker_port".

Usage:
    python start_docker.py <nni-version> <container-name>
"""

import random
import socket
import sys

from _common import build_wheel, run_command, set_variable

# find idle port
port = random.randint(10000, 20000)
while True:
    sock = socket.socket()
    if sock.connect_ex(('localhost', port)) != 0:
        break  # failed to connect, so this is idle
    sock.close()
    port = random.randint(10000, 20000)

build_wheel()
run_command(f'docker build --build-arg NNI_RELEASE={sys.argv[1]} -t nni-nightly')
run_command(f'docker run -d -p {port}:22 --name {sys.argv[2]} nni-nightly')
set_variable('docker_port', port)
