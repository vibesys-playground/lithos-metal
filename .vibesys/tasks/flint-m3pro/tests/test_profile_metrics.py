"""Synthetic native phases: no model, server, GPU lock, or real clock."""

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
from profile_metrics import (
    PHASES,
    PREFLIGHT_GENERATIONS,
    ProfileWriter,
    profile_values,
    synthetic_profile,
)


class ProfileTests(unittest.TestCase):
    planned = [dict(request_id=str(n), output_len="32") for n in range(2)]

    def records(self):
        return [
            {"outcome": dict(request_id=str(n), output_len_actual=32, status="SUCCESS", error=None)}
            for n in range(2)
        ]

    def logs(self, **changes):
        return [
            "\n".join(
                "INFO:monolith.serve:Generation: "
                + repr(
                    {
                        **{name: value for name in PHASES.values()},
                        **changes,
                    }
                )
                for value in [1000.0] * PREFLIGHT_GENERATIONS + [1.0, 2.0]
            )
        ]

    def test_native_phases_are_per_request_and_summed_without_scoring_directions(self):
        values = profile_values(self.planned, self.records(), self.logs(), 12.0)
        self.assertEqual(values["startup_duration_s"], 12.0)
        for field in PHASES:
            self.assertEqual(values[f"request_0.{field}"], 1.0)
            self.assertEqual(values[f"request_1.{field}"], 2.0)
            self.assertEqual(values[field], 3.0)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "profile.jsonl"
            writer = ProfileWriter(path)
            writer.result(values)
            writer.close()
            hello, result = map(json.loads, path.read_text().splitlines())
        self.assertEqual(hello["kind"], "hello")
        self.assertEqual(hello["protocol"], 2)
        self.assertEqual(result, {"kind": "result", "values": values})
        self.assertFalse(any("direction" in spec for spec in hello["metrics"].values()))

    def test_missing_invalid_or_negative_phase_fails_instead_of_inventing_zero(self):
        for value in (None, -1.0, True, "2", math.inf):
            with self.subTest(value=value), self.assertRaises((ValueError, SyntaxError)):
                profile_values(self.planned, self.records(), self.logs(prefill_wall_ms=value), 1.0)

    def test_missing_extra_or_failed_requests_and_generations_fail(self):
        records = self.records()
        failed = self.records()
        failed[0]["outcome"]["status"] = "ERROR"
        for observed, logs in (
            (records[:1], self.logs()),
            (records[::-1], self.logs()),
            (failed, self.logs()),
            (records, []),
            (records, self.logs() * 2),
        ):
            with self.subTest(records=observed, logs=logs), self.assertRaises(ValueError):
                profile_values(self.planned, observed, logs, 1.0)

    def test_synthetic_engine_uses_the_same_phase_decoder(self):
        values = synthetic_profile(self.planned)
        self.assertEqual(values["prefill_timing"], 3.0)
        self.assertEqual(values["decode_timing"], 3.0)


if __name__ == "__main__":
    unittest.main()
