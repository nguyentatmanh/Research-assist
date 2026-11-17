"""Stop supervisor-owned processes on Kaggle and during local Windows testing."""

import os
import subprocess
from typing import Any


def stop_owned_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        # Windows virtual environments launch another interpreter below python.exe.
        try:
            stopped = subprocess.run(
                ["taskkill", "/pid", str(process.pid), "/t", "/f"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=False,
            )
            if stopped.returncode == 0:
                process.wait(timeout=5)
                return
        except (OSError, subprocess.TimeoutExpired):
            pass
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
