"""Engine task environment: one locked uv sync, shared by hash, outside VibeSys.

The candidate's own uv.lock is the image. `prepare` installs it into a cache
directory named by that file's sha256, then returns that environment's
interpreter and cmake. Weights and the shared pack are never placed there.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# `--locked` refuses to rewrite uv.lock and fails when pyproject.toml has
# drifted. `--frozen` installs that stale lock without failing, so it is not
# the setup command. uv 0.8 rejects passing the two flags together.
SETUP_ARGV = (
    "sync",
    "--locked",
    "--extra",
    "serve",
    "--no-install-project",
    "--python",
    "3.12",
)
SETUP_DISPLAY = "uv sync --locked --extra serve --no-install-project --python 3.12"
_ACTIVE = "FLINT_TASK_ENV"
_GPU_LOCK = "/usr/bin/lockf"
_ENV_LOCK = "/tmp/flint-task-env.lock"


class TaskEnvironmentError(RuntimeError):
    """The task environment cannot be used. The message names the next action."""


class MissingToolError(TaskEnvironmentError):
    """A tool is absent from the locked environment."""


class FrozenLockError(TaskEnvironmentError):
    """The candidate lock cannot be installed as it is."""


class HostCheckError(TaskEnvironmentError):
    """check_host.sh reported a missing host prerequisite."""


@dataclass(frozen=True)
class Prepared:
    """Interpreter and cmake from one lockfile-addressed environment."""

    root: Path
    python: Path
    cmake: Path


class CommandRunner:
    """Run one argv. Production and tests share this method, not a mock."""

    def run(self, argv: list[str], env: dict[str, str]) -> int:
        command = list(argv)
        if len(command) >= 2 and command[1] == "sync":
            command = [_GPU_LOCK, "-k", _ENV_LOCK, *command]
        return subprocess.run(command, env=env).returncode


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def task_directory(project: Path) -> Path:
    return project / ".vibesys" / "tasks" / "flint-m3pro"


def environment_root(home: Path, lock_bytes: bytes) -> Path:
    """Cache directory for these exact lock bytes. A new lock gets a new directory."""
    digest = hashlib.sha256(lock_bytes).hexdigest()
    return Path(home) / ".cache" / "lithos-metal" / "task-env" / digest


def frozen_failure_message(lockfile: Path) -> str:
    return (
        f"uv sync --locked failed for {lockfile}. "
        "The task environment is installed only from this candidate's uv.lock, "
        "into the cache directory keyed by that file's sha256. "
        "If you added or changed a dependency, run `uv lock` in the candidate "
        "checkout and commit the updated uv.lock. This sync does not update "
        "the lock or resolve new dependencies."
    )


def missing_tool_message(name: str, path: Path) -> str:
    return (
        f"Missing tool {name!r} at {path}. Create the engine task environment with "
        f"{SETUP_DISPLAY}, with UV_PROJECT_ENVIRONMENT set to "
        "$HOME/.cache/lithos-metal/task-env/<sha256 of the candidate uv.lock>. "
        "The task reads this tool from that environment and does not search PATH."
    )


def tool_path(environment: Path, name: str) -> Path:
    """Return environment/bin/name, or raise. Never consults PATH."""
    path = Path(environment) / "bin" / name
    if not path.is_file() or not os.access(path, os.X_OK):
        raise MissingToolError(missing_tool_message(name, path))
    return path.resolve()


def uv_search_paths(home: Path, environ: dict[str, str]) -> tuple[Path, ...]:
    """Absolute uv locations. PATH is intentionally absent."""
    candidates: list[Path] = []
    override = environ.get("FLINT_UV", "")
    if override and Path(override).is_absolute():
        candidates.append(Path(override))
    candidates.extend(
        (
            Path(home) / ".local" / "bin" / "uv",
            Path("/opt/homebrew/bin/uv"),
            Path("/usr/local/bin/uv"),
        )
    )
    return tuple(candidates)


def locate_uv(
    home: Path,
    environ: dict[str, str],
    candidates: tuple[Path, ...] | None = None,
) -> Path:
    """Find uv at an absolute location. A PATH entry is not a candidate."""
    search = uv_search_paths(home, environ) if candidates is None else candidates
    for candidate in search:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    raise MissingToolError(
        "Missing tool 'uv'. The task environment is created by "
        f"{SETUP_DISPLAY}, with UV_PROJECT_ENVIRONMENT set to "
        "$HOME/.cache/lithos-metal/task-env/<sha256 of uv.lock>. "
        "Install uv at ~/.local/bin/uv (https://docs.astral.sh/uv/) or set "
        "FLINT_UV to an absolute path. The task does not search PATH."
    )


def check_host_argv(project: Path) -> list[str]:
    return ["/bin/sh", str(task_directory(project) / "check_host.sh")]


def sync_argv(uv: Path, project: Path) -> list[str]:
    return [str(uv), *SETUP_ARGV, "--project", str(project)]


def sync_environment(base: dict[str, str], root: Path) -> dict[str, str]:
    """Environment for uv sync. Drops the caller's virtualenv and PATH."""
    skipped = {"VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME", "UV_PROJECT_ENVIRONMENT"}
    env = {key: value for key, value in base.items() if key not in skipped}
    env["PATH"] = "/usr/bin:/bin:/opt/homebrew/bin:/usr/local/bin"
    env["UV_PROJECT_ENVIRONMENT"] = str(root)
    return env


def load_prepared(project: Path, home: Path) -> Prepared:
    """Tools already installed for this lock. Does not sync and does not search PATH."""
    lock_path = project / "uv.lock"
    if not lock_path.is_file():
        raise FrozenLockError(frozen_failure_message(lock_path))
    root = environment_root(home, lock_path.read_bytes())
    return Prepared(root, tool_path(root, "python"), tool_path(root, "cmake"))


def require_current_interpreter(current: Path, prepared: Prepared) -> None:
    if Path(current).resolve() != prepared.python.resolve():
        raise TaskEnvironmentError(
            f"Refusing to build with {current}. The task interpreter is {prepared.python}. "
            "Run .vibesys/tasks/flint-m3pro/setup_env.sh and use the interpreter it prints. "
            "The task does not use python3 or cmake from PATH."
        )


def native_build_commands(project: Path, prepared: Prepared) -> tuple[list[str], list[str]]:
    """CMake argv for this checkout. The executable is the environment's cmake."""
    build = project / "build" / "flint-task"
    configure = [
        str(prepared.cmake),
        "-S",
        str(project),
        "-B",
        str(build),
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DPython_EXECUTABLE={prepared.python}",
    ]
    compile_command = [
        str(prepared.cmake),
        "--build",
        str(build),
        "--config",
        "Release",
        "--parallel",
        "2",
    ]
    return configure, compile_command


def _reject_layout(project: Path, home: Path, root: Path) -> None:
    pack = Path(home) / ".cache" / "lithos-metal" / "flint-int4"
    weights = Path(home) / ".cache" / "huggingface"
    if root == project or project in root.parents or root in project.parents:
        raise TaskEnvironmentError(
            "The task environment must stay outside the candidate checkout."
        )
    if root in pack.parents or pack in root.parents or root == pack:
        raise TaskEnvironmentError(
            "The task environment must not contain the shared pack."
        )
    if root in weights.parents or weights in root.parents or root == weights:
        raise TaskEnvironmentError(
            "The task environment must not contain the weight snapshot."
        )


def prepare(
    project: Path,
    home: Path,
    runner: CommandRunner,
    base_env: dict[str, str],
) -> Prepared:
    """Run check_host.sh, then one locked sync of this candidate's lock."""
    project = Path(project).resolve()
    home = Path(home)
    check_code = runner.run(check_host_argv(project), dict(base_env))
    if check_code != 0:
        raise HostCheckError(
            "Host check failed (.vibesys/tasks/flint-m3pro/check_host.sh). "
            "Fix every missing item it printed, then retry."
        )
    lock_path = project / "uv.lock"
    if not lock_path.is_file():
        raise FrozenLockError(frozen_failure_message(lock_path))
    root = environment_root(home, lock_path.read_bytes())
    _reject_layout(project, home, root)
    uv = locate_uv(home, base_env)
    code = runner.run(sync_argv(uv, project), sync_environment(base_env, root))
    if code != 0:
        raise FrozenLockError(frozen_failure_message(lock_path))
    return Prepared(root, tool_path(root, "python"), tool_path(root, "cmake"))


def enter_task_environment() -> None:
    """Re-exec this process with the locked interpreter. No-op once entered."""
    if os.environ.get(_ACTIVE) == "1":
        return
    try:
        prepared = prepare(project_root(), Path.home(), CommandRunner(), dict(os.environ))
    except TaskEnvironmentError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc
    if Path(sys.executable).resolve() == prepared.python.resolve():
        os.environ[_ACTIVE] = "1"
        return
    os.environ[_ACTIVE] = "1"
    os.environ["PATH"] = f"{prepared.root / 'bin'}:/usr/bin:/bin"
    os.environ["VIRTUAL_ENV"] = str(prepared.root)
    os.environ.pop("PYTHONPATH", None)
    os.environ.pop("PYTHONHOME", None)
    python = str(prepared.python)
    os.execv(python, [python, *sys.argv])


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    command = args[0] if args else "setup"
    if command not in {"setup", "accuracy"}:
        print("usage: task_env.py [setup|accuracy]", file=sys.stderr)
        return 2
    try:
        prepared = prepare(project_root(), Path.home(), CommandRunner(), dict(os.environ))
    except TaskEnvironmentError as exc:
        print(exc, file=sys.stderr)
        return 1
    if command == "setup":
        print(prepared.python)
        return 0
    checker = task_directory(project_root()) / "accuracy_checker" / "checker.py"
    os.execv(
        _GPU_LOCK,
        [_GPU_LOCK, "-k", "/tmp/flint-gpu.lock", str(prepared.python), str(checker)],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
