"""RF deadline policy uses fixed planned denominators and actual wire timings."""

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
from replay_metrics import BUDGET_SECONDS, WORST_MS, replay_window, summarize


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
                status="SUCCESS",
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


class SummaryMappingTests(unittest.TestCase):
    planned = ReplayMetricsTests.planned
    record = ReplayMetricsTests.record

    def summary(self):
        return {
            "replay": {
                "kind": "sessions",
                "common": {
                    "attempted_steps": 3,
                    "success_steps": 3,
                    "failed_steps": 0,
                    "output_mismatch_steps": 0,
                    "ttft_ms_avg": 50,
                    "tpot_ms_avg": 100,
                },
            }
        }

    def test_metrics_are_taken_from_validated_rf_summary(self):
        values, _ = summarize(
            self.planned, [self.record(n) for n in range(3)], rf_summary=self.summary()
        )
        self.assertEqual(values["ttft_ms_avg"], 50)
        self.assertEqual(values["tpot_ms_avg"], 100)
        for field, bad in [
            ("ttft_ms_avg", 51),
            ("tpot_ms_avg", 99),
            ("failed_steps", 1),
            ("output_mismatch_steps", 1),
            ("attempted_steps", 2),
        ]:
            summary = self.summary()
            summary["replay"]["common"][field] = bad
            with self.subTest(field=field), self.assertRaises(ValueError):
                summarize(
                    self.planned, [self.record(n) for n in range(3)], rf_summary=summary
                )

    def test_timeout_requires_deadline_and_rejects_invalid_timestamps(self):
        for options in [
            dict(timed_out=True),
            dict(deadline_timestamp=20),
            dict(timed_out=True, deadline_timestamp=math.nan),
        ]:
            with self.assertRaises(ValueError):
                summarize(self.planned, [], **options)
        for bad in [math.nan, math.inf, True, None]:
            record = self.record(0)
            record["outcome"]["complete_timestamp"] = bad
            with self.assertRaises(ValueError):
                summarize(self.planned, [record], timed_out=True, deadline_timestamp=20)

    def test_only_failure_crossing_deadline_is_censored(self):
        record = self.record(0)
        record["outcome"].update(
            error="stream closed", complete_timestamp=21, status="FAILED"
        )
        values, rows = summarize(
            self.planned, [record], timed_out=True, deadline_timestamp=20
        )
        self.assertEqual(values["timed_out_requests"], 3)
        self.assertEqual(rows[0]["status"], "timed_out")
        record["outcome"]["complete_timestamp"] = 20
        with self.assertRaises(ValueError):
            summarize(self.planned, [record], timed_out=True, deadline_timestamp=20)


class ReplayWindowTests(unittest.TestCase):
    def test_delayed_log_creation_never_extends_scoring_window(self):
        # RF is represented by its real timestamped log contract. No clock or
        # process runs: these outcomes cover either order of writer/request start.
        for writer_start in (9, 10, 11):
            for finished in (609, 610, 611):
                records = [
                    {
                        "outcome": {
                            "submit_timestamp": 10,
                            "complete_timestamp": finished,
                        }
                    }
                ]
                start, deadline, expired = replay_window(
                    records,
                    log_created_timestamp=writer_start,
                    watchdog_timed_out=False,
                )
                self.assertEqual(start, 10)
                self.assertEqual(deadline, 10 + BUDGET_SECONDS)
                self.assertEqual(expired, finished > deadline)

    def test_no_outcome_retains_watchdog_timeout_and_bad_timestamps_fail(self):
        self.assertEqual(
            replay_window([], log_created_timestamp=10, watchdog_timed_out=True),
            (10, 610, True),
        )
        for submitted, completed in [(math.nan, 20), (1, math.inf), (2, 1), (True, 20)]:
            with self.assertRaises(ValueError):
                replay_window(
                    [
                        {
                            "outcome": {
                                "submit_timestamp": submitted,
                                "complete_timestamp": completed,
                            }
                        }
                    ],
                    log_created_timestamp=10,
                    watchdog_timed_out=True,
                )
