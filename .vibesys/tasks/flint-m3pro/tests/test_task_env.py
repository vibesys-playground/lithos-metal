"""Tool resolution and frozen-lock failures. No GPU, sleep, or live uv."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
from task_env import (
    FrozenLockError,
    HostCheckError,
    MissingToolError,
    Prepared,
    check_host_argv,
    environment_root,
    frozen_failure_message,
    load_prepared,
    locate_uv,
    missing_tool_message,
    native_build_commands,
    prepare,
    require_current_interpreter,
    sync_argv,
    sync_environment,
    tool_path,
)


class FakeRunner:
    """Records argv and, on a successful sync, materializes python and cmake."""

    def __init__(self, sync_code: int = 0, host_code: int = 0):
        self.sync_code = sync_code
        self.host_code = host_code
        self.calls: list[tuple[list[str], dict[str, str]]] = []

    def run(self, argv, env):
        self.calls.append((list(argv), dict(env)))
        if argv[:1] == ["/bin/sh"]:
            return self.host_code
        if self.sync_code != 0:
            return self.sync_code
        bindir = Path(env["UV_PROJECT_ENVIRONMENT"]) / "bin"
        bindir.mkdir(parents=True, exist_ok=True)
        for name in ("python", "cmake"):
            path = bindir / name
            path.write_text("#!/bin/sh\n")
            path.chmod(0o755)
        return 0


class TaskEnvTests(unittest.TestCase):
    def project(self, root: Path, lock: bytes = b"lock-v1\n") -> Path:
        project = root / "candidate"
        project.mkdir(parents=True)
        (project / "uv.lock").write_bytes(lock)
        return project

    def test_environment_directory_is_the_lock_hash_and_excludes_weights(self):
        home = Path("/cache-home")
        first = environment_root(home, b"same-lock")
        again = environment_root(home, b"same-lock")
        changed = environment_root(home, b"same-lock\n")
        self.assertEqual(first, again)
        self.assertNotEqual(first, changed)
        self.assertEqual(len(first.name), 64)
        self.assertNotIn("flint-int4", first.parts)
        self.assertNotIn("huggingface", first.parts)
        pack = home / ".cache/lithos-metal/flint-int4"
        weights = home / ".cache/huggingface"
        self.assertFalse(str(first).startswith(str(pack)))
        self.assertFalse(str(first).startswith(str(weights)))
        self.assertNotIn(first, pack.parents)
        self.assertNotIn(pack, first.parents)

    def test_every_missing_tool_names_itself_and_the_frozen_setup(self):
        environment = Path("/env/does-not-exist")
        for name in ("cmake", "python", "ninja"):
            with self.assertRaises(MissingToolError) as caught:
                tool_path(environment, name)
            message = str(caught.exception)
            self.assertIn(repr(name), message)
            self.assertIn("uv sync --locked", message)
            self.assertIn("does not search PATH", message)
            self.assertIn(str(environment / "bin" / name), message)

    def test_tool_path_uses_the_environment_not_a_path_decoy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            decoy_dir = root / "path-decoy"
            decoy_dir.mkdir()
            decoy = decoy_dir / "cmake"
            decoy.write_text("decoy")
            decoy.chmod(0o755)
            environment = root / "task-env"
            binary = environment / "bin" / "cmake"
            binary.parent.mkdir(parents=True)
            binary.write_text("real")
            binary.chmod(0o755)
            previous = os.environ.get("PATH")
            os.environ["PATH"] = str(decoy_dir)
            try:
                resolved = tool_path(environment, "cmake")
            finally:
                if previous is None:
                    os.environ.pop("PATH", None)
                else:
                    os.environ["PATH"] = previous
            self.assertEqual(resolved, binary.resolve())
            self.assertNotEqual(resolved, decoy.resolve())

    def test_uv_is_not_taken_from_path(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            decoy_dir = root / "bin"
            decoy_dir.mkdir()
            decoy = decoy_dir / "uv"
            decoy.write_text("uv")
            decoy.chmod(0o755)
            with self.assertRaises(MissingToolError) as caught:
                locate_uv(root / "home", {"PATH": str(decoy_dir)}, candidates=())
            self.assertIn("'uv'", str(caught.exception))
            self.assertIn("does not search PATH", str(caught.exception))
            explicit = locate_uv(root / "home", {"FLINT_UV": str(decoy)}, candidates=(decoy,))
            self.assertEqual(explicit, decoy.resolve())

    def test_prepare_checks_the_host_then_syncs_the_candidate_lock(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            home = root / "home"
            project = self.project(root, b"candidate-lock\n")
            (home / ".local" / "bin").mkdir(parents=True)
            uv = home / ".local" / "bin" / "uv"
            uv.write_text("uv")
            uv.chmod(0o755)
            runner = FakeRunner()
            caller_env = {
                "PATH": "/vibesys/.venv/bin:/usr/bin",
                "VIRTUAL_ENV": "/vibesys/.venv",
                "HOME": str(home),
            }
            prepared = prepare(project, home, runner, caller_env)
            self.assertEqual(runner.calls[0][0], check_host_argv(project.resolve()))
            sync_call, sync_env = runner.calls[1]
            self.assertEqual(sync_call, sync_argv(uv.resolve(), project.resolve()))
            self.assertIn("--locked", sync_call)
            self.assertNotIn("--frozen", sync_call)
            self.assertEqual(
                sync_env["UV_PROJECT_ENVIRONMENT"],
                str(environment_root(home, b"candidate-lock\n")),
            )
            self.assertNotIn("/vibesys/.venv", sync_env["PATH"])
            self.assertNotIn("VIRTUAL_ENV", sync_env)
            self.assertEqual(prepared.python, (prepared.root / "bin" / "python").resolve())
            self.assertEqual(prepared.cmake, (prepared.root / "bin" / "cmake").resolve())
            self.assertNotEqual(prepared.root, home / ".cache" / "huggingface")
            configure, compile_command = native_build_commands(project, prepared)
            self.assertEqual(configure[0], str(prepared.cmake))
            self.assertNotEqual(configure[0], "cmake")
            self.assertEqual(compile_command[0], str(prepared.cmake))
            self.assertIn(f"-DPython_EXECUTABLE={prepared.python}", configure)

    def test_host_failure_does_not_sync(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            project = self.project(root)
            runner = FakeRunner(host_code=1)
            with self.assertRaises(HostCheckError) as caught:
                prepare(project, root / "home", runner, {})
            self.assertIn("check_host.sh", str(caught.exception))
            self.assertEqual(len(runner.calls), 1)

    def test_stale_lock_fails_with_a_commit_message(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            home = root / "home"
            project = self.project(root)
            (home / ".local" / "bin").mkdir(parents=True)
            uv = home / ".local" / "bin" / "uv"
            uv.write_text("uv")
            uv.chmod(0o755)
            runner = FakeRunner(sync_code=1)
            with self.assertRaises(FrozenLockError) as caught:
                prepare(project, home, runner, {})
            message = str(caught.exception)
            self.assertEqual(message, frozen_failure_message(project.resolve() / "uv.lock"))
            self.assertIn("uv sync --locked", message)
            self.assertIn("uv.lock", message)
            self.assertIn("commit the updated uv.lock", message)
            self.assertFalse((environment_root(home, b"lock-v1\n") / "bin" / "cmake").exists())

    def test_wrong_interpreter_and_missing_cmake_fail_before_a_build(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            home = root / "home"
            project = self.project(root, b"built\n")
            environment = environment_root(home, b"built\n")
            (environment / "bin").mkdir(parents=True)
            python = environment / "bin" / "python"
            cmake = environment / "bin" / "cmake"
            python.write_text("python")
            cmake.write_text("cmake")
            python.chmod(0o755)
            cmake.chmod(0o755)
            decoy = root / "other" / "cmake"
            decoy.parent.mkdir()
            decoy.write_text("decoy")
            decoy.chmod(0o755)
            prepared = load_prepared(project, home)
            self.assertEqual(prepared.cmake, cmake.resolve())
            self.assertNotEqual(prepared.cmake, decoy.resolve())
            require_current_interpreter(python, prepared)
            other_home_project = self.project(root / "second", b"other-lock")
            partial = environment_root(home, b"other-lock") / "bin"
            partial.mkdir(parents=True)
            partial_python = partial / "python"
            partial_python.write_text("python")
            partial_python.chmod(0o755)
            with self.assertRaises(MissingToolError) as missing:
                load_prepared(other_home_project, home)
            self.assertIn("'cmake'", str(missing.exception))
            self.assertIn("uv sync --locked", str(missing.exception))
            other = Prepared(environment, python, cmake)
            with self.assertRaisesRegex(Exception, "does not use python3 or cmake from PATH"):
                require_current_interpreter(Path("/usr/bin/python3"), other)

    def test_missing_tool_message_is_stable(self):
        text = missing_tool_message("cmake", Path("/env/bin/cmake"))
        self.assertIn("'cmake'", text)
        self.assertIn("/env/bin/cmake", text)
        self.assertIn("uv sync --locked", text)


if __name__ == "__main__":
    unittest.main()
