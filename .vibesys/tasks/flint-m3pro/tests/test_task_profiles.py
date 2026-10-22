"""Every Flint task variant declares its own quick-workload profile capture."""

import tomllib
import unittest
from pathlib import Path


class TaskProfileTests(unittest.TestCase):
    def test_each_variant_declares_the_complete_profile_contract(self):
        tasks = Path(__file__).resolve().parents[2]
        manifests = sorted(tasks.glob("flint-*/vibesys.input.toml"))
        self.assertTrue(manifests)
        for manifest in manifests:
            with self.subTest(task=manifest.parent.name):
                data = tomllib.loads(manifest.read_text())
                self.assertEqual(
                    data.get("profile"),
                    {
                        "entrypoint": "request-factory-adapter",
                        "args": [
                            "${PROJECT_ROOT}/.vibesys/tasks/flint-m3pro/benchmark/run.py",
                            "--profile",
                            "--workload",
                            "quick",
                        ],
                        "timeout_seconds": 2500,
                        "fields": ["prefill_timing", "decode_timing"],
                    },
                )


if __name__ == "__main__":
    unittest.main()
