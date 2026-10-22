"""CPU-only shell entrypoint checks; subprocess bounds only guard hangs."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]


class HostScriptTests(unittest.TestCase):
    def test_directory_failures_stop_at_the_named_task_path(self):
        for name in ("setup_env.sh", "check_host.sh"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                missing = str(Path(temp) / "absent-task" / name)
                result = subprocess.run(
                    ["/bin/sh", "-c", (TASK / name).read_text(), missing],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Cannot enter task directory:", result.stderr)
                self.assertIn(str(Path(missing).parent), result.stderr)
                self.assertNotIn("Traceback", result.stderr)
                self.assertNotIn("Host check", result.stderr)

    def test_host_check_keeps_manifest_and_pack_subprocess_diagnostics(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            task = root / ".vibesys/tasks/flint-m3pro"
            task.mkdir(parents=True)
            script = task / "check_host.sh"
            script.write_text((TASK / script.name).read_text())
            pack = root / "pack"
            pack.mkdir()
            for name in ("manifest.json", "pack-key.json", "weights.pack"):
                (pack / name).write_text("invalid json")
            result = subprocess.run(
                ["/bin/sh", str(script)],
                env={**os.environ, "FLINT_PACK_DIR": str(pack)},
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Host check failed:", result.stderr)
            diagnostic = result.stderr.split("Host check failed:", 1)[1]
            self.assertIn("FileNotFoundError", diagnostic)
            self.assertIn(str(task / "reference/checkpoint.json"), diagnostic)
            self.assertIn("JSONDecodeError", diagnostic)


if __name__ == "__main__":
    unittest.main()
