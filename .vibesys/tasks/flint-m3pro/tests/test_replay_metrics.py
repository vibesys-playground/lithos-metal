"""RF deadline policy uses fixed planned denominators and actual wire timings."""

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
from replay_metrics import WORST_MS, summarize


class ReplayMetricsTests(unittest.TestCase):
    planned = [dict(request_id=str(n), output_len="3") for n in range(3)]

    def record(self, n, **changes):
        return {
            "outcome": dict(
                request_id=str(n),
                output_len_actual=3,
                first_token_id_ms=50,
                token_delivery_tpot_ms=100,
                complete_timestamp=10 + n,
                error=None,
                **changes,
            )
        }

    def test_complete_values_match_rf_summary_means(self):
        values, rows = summarize(self.planned, [self.record(n) for n in range(3)])
        self.assertEqual(values["ttft_ms_avg"], 50)
        self.assertEqual(values["tpot_ms_avg"], 100)
        self.assertEqual(values["decode_tokens_per_second"], 10)
        self.assertEqual(values["completed_requests"], 3)
        self.assertFalse(any(r["censored"] for r in rows))

    def test_every_unfinished_round_is_explicit_and_scores_improve_with_completion(
        self,
    ):
        previous = None
        for completed in range(4):
            values, rows = summarize(
                self.planned,
                [self.record(n) for n in range(completed)],
                timed_out=True,
                deadline_timestamp=20,
            )
            self.assertEqual(values["timed_out_requests"], 3 - completed)
            self.assertEqual(sum(r["censored"] for r in rows), 3 - completed)
            for row in rows[completed:]:
                self.assertEqual(
                    (
                        row["status"],
                        row["ttft_ms"],
                        row["tpot_ms"],
                        row["decode_tokens_per_second"],
                    ),
                    ("timed_out", WORST_MS, WORST_MS, 0),
                )
            if previous:
                self.assertLess(values["ttft_ms_avg"], previous["ttft_ms_avg"])
                self.assertLess(values["tpot_ms_avg"], previous["tpot_ms_avg"])
                self.assertGreater(
                    values["decode_tokens_per_second"],
                    previous["decode_tokens_per_second"],
                )
            previous = values

    def test_shutdown_grace_completion_is_censored(self):
        record = self.record(0)
        record["outcome"]["complete_timestamp"] = 21
        values, _ = summarize(
            self.planned, [record], timed_out=True, deadline_timestamp=20
        )
        self.assertEqual(values["completed_requests"], 0)

    def test_errors_before_deadline_and_short_outputs_never_score(self):
        for field, value in [
            ("error", "HTTP 500"),
            ("output_len_actual", 2),
            ("first_token_id_ms", None),
            ("token_delivery_tpot_ms", math.nan),
        ]:
            record = self.record(0)
            record["outcome"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                summarize(self.planned, [record], timed_out=True, deadline_timestamp=20)

    def test_missing_rows_require_actual_deadline_and_duplicates_fail(self):
        for records in ([], [self.record(0), self.record(0)]):
            with self.assertRaises(ValueError):
                summarize(self.planned, records)
