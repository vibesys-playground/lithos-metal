"""Command-wide deadline ownership for the operator/evaluator CLI.

The timer interrupts blocking startup and HTTP I/O in the main thread. The
exception unwinds through the launcher's server cleanup before being reported.
The fake timer used by hermetic tests expires only when explicitly advanced.
"""

import signal
from contextlib import contextmanager
from typing import Callable, Protocol

__all__ = ["deadline_guard", "DeadlineTimer"]


class DeadlineTimer(Protocol):
    def start(self, seconds: float, expired: Callable[[], None]) -> None: ...
    def stop(self) -> None: ...


class SignalDeadlineTimer:
    def __init__(self):
        self.previous = None
        self.active = False

    def start(self, seconds, expired):
        if self.active or signal.getitimer(signal.ITIMER_REAL)[0]:
            raise RuntimeError("Benchmark cannot replace an active command deadline")
        self.previous = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, lambda signum, frame: expired())
        self.active = True
        try:
            signal.setitimer(signal.ITIMER_REAL, seconds)
        except BaseException:
            self.stop()
            raise

    def stop(self):
        if self.active:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, self.previous)
            self.active = False


@contextmanager
def deadline_guard(seconds, *, timer: DeadlineTimer | None = None):
    if seconds <= 0:
        raise ValueError("Command deadline must be positive")
    timer = timer if timer is not None else SignalDeadlineTimer()

    def expired():
        raise TimeoutError(f"Whole benchmark exceeded {seconds:g} seconds")

    timer.start(seconds, expired)
    try:
        yield
    finally:
        timer.stop()
