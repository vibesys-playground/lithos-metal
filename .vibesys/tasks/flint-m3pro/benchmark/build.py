#!/usr/bin/env python3
"""Build this measured checkout's native runtime with the locked environment's cmake.

The executable is that environment's cmake, never a PATH lookup. The extension
is written into this checkout. A prebuilt extension from another checkout is
not used.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from task_env import (
    TaskEnvironmentError,
    load_prepared,
    native_build_commands,
    project_root,
    require_current_interpreter,
)


def main() -> None:
    project = project_root()
    try:
        prepared = load_prepared(project, Path.home())
        require_current_interpreter(Path(sys.executable), prepared)
    except TaskEnvironmentError as exc:
        raise SystemExit(str(exc)) from exc
    configure, compile_command = native_build_commands(project, prepared)
    subprocess.run(configure, check=True, timeout=120, cwd=project)
    subprocess.run(compile_command, check=True, timeout=300, cwd=project)


if __name__ == "__main__":
    main()
