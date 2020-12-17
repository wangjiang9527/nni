from pathlib import Path
import os
import subprocess
import sys

nni_root = str(Path(__file__).parents[2])

def build_wheel():
    python = sys.executable
    os_spec = {
        'linux': 'manylinux1_x86_64',
        'darwin': 'macosx_10_9_x86_64',
        'win32': 'win_amd64'
    }[sys.platform]

    run_command(f'{python} setup.py clean --all')
    run_command(f'{python} setup.py build_ts', NNI_RELEASE=sys.argv[1])
    run_command(f'{python} setup.py bdist_wheel -p {os_spec}', NNI_RELEASE=sys.argv[1])

    return f'dist/nni-{version}-py3-none-{os_spec}.whl'

def run_command(command, **extra_environ):
    if isinstance(command, str):
        command = cmd.split()
    subprocess.run(
        command,
        cwd = nni_root,
        check = True,
        env = {**os.environ, **extra_environ}
    )

def set_variable(key, value):
    print(f'##vso[task.setvariable variable={key}]{value}')
