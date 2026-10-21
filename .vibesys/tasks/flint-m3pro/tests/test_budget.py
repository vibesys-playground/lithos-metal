"""Deadline expiration unwinds resources using a manually advanced timer."""

import sys
import unittest
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
from budget import deadline_guard


class FakeDeadlineTimer:
    def __init__(self):
        self.remaining = None
        self.expired = None

    def start(self, seconds, expired):
        if self.remaining is not None:
            raise RuntimeError("Timer already active")
        self.remaining, self.expired = seconds, expired

    def advance(self, seconds):
        if self.remaining is not None:
            self.remaining -= seconds
            if self.remaining <= 0:
                self.expired()

    def stop(self):
        self.remaining = None
        self.expired = None


class BudgetTests(unittest.TestCase):
    def test_deadline_unwinds_startup_or_request_and_stops_timer(self):
        for phase in ("startup", "request"):
            timer, active = FakeDeadlineTimer(), []

            @contextmanager
            def server():
                active.append(phase)
                try:
                    yield
                finally:
                    active.clear()

            with (
                self.subTest(phase=phase),
                self.assertRaisesRegex(TimeoutError, "600 seconds"),
            ):
                with deadline_guard(600, timer=timer), server():
                    timer.advance(599)
                    self.assertEqual(active, [phase])
                    timer.advance(1)
            self.assertEqual(active, [])
            self.assertIsNone(timer.remaining)

    def test_normal_and_error_exits_stop_timer(self):
        for failure in (False, True):
            timer = FakeDeadlineTimer()
            try:
                with deadline_guard(600, timer=timer):
                    if failure:
                        raise ValueError("caller failed")
            except ValueError:
                self.assertTrue(failure)
            self.assertIsNone(timer.remaining)

    def test_invalid_deadline_does_not_start_timer(self):
        timer = FakeDeadlineTimer()
        for seconds in (-1, 0):
            with self.assertRaises(ValueError):
                with deadline_guard(seconds, timer=timer):
                    self.fail("Invalid deadline admitted")
        self.assertIsNone(timer.remaining)
