"""Launch a command fully detached (new session) so it survives the
parent shell/process-group being killed -- e.g. by a harness restart.

Usage: python scripts/detach_launch.py <logfile> <pidfile> -- <cmd> [args...]
"""
from __future__ import annotations

import subprocess
import sys


def main() -> None:
    argv = sys.argv[1:]
    sep = argv.index("--")
    logfile, pidfile = argv[0], argv[1]
    cmd = argv[sep + 1:]

    with open(logfile, "ab") as log:
        proc = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    with open(pidfile, "a") as f:
        f.write(f"{proc.pid}\n")
    print(f"launched pid={proc.pid} -> {logfile}")


if __name__ == "__main__":
    main()
