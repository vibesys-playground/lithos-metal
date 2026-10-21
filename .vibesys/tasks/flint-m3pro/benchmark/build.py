#!/usr/bin/env python3
"""Build this measured checkout's native runtime, without installing another tree.

VibeSys runs this candidate build before the trusted checker/measurement.
CMake emits the extension into monolith/runtime. No GPU work is executed.
"""

from __future__ import annotations

import subprocess
import sys


def main() -> None:
    subprocess.run(
        [
            "cmake",
            "-S",
            ".",
            "-B",
            "build/flint-task",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DPython_EXECUTABLE={sys.executable}",
        ],
        check=True,
        timeout=120,
    )
    subprocess.run(
        [
            "cmake",
            "--build",
            "build/flint-task",
            "--config",
            "Release",
            "--parallel",
            "2",
        ],
        check=True,
        timeout=300,
    )


if __name__ == "__main__":
    main()
