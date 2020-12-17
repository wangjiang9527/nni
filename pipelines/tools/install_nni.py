import sys

import _common import build_wheel, run_command

wheel = build_wheel()
run_command(f'{sys.executable} -m pip install {wheel}')
