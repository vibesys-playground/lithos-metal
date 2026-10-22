"""Published minimal protocol-v2 writer; sdk/vs-evaluator/PROTOCOL.md owns schema."""

from __future__ import annotations

import json
import math
from pathlib import Path

__all__ = ["ResultWriter", "METRICS"]
METRICS = {
    "ttft_ms_avg": {"unit": "ms", "direction": "min"},
    "tpot_ms_avg": {"unit": "ms", "direction": "min"},
    "decode_tokens_per_second": {"unit": "tok/s", "direction": "max"},
    "completed_requests": {"unit": "requests", "direction": "max"},
    "timed_out_requests": {"unit": "requests"},
    "startup_duration_s": {"unit": "s"},
    "scored_duration_s": {"unit": "s"},
}


class ResultWriter:
    def __init__(self, path: Path):
        self.file = path.open("w", encoding="utf-8")
        self.finished = False
        self.write({"kind": "hello", "protocol": 2, "metrics": METRICS})

    def write(self, record: dict) -> None:
        self.file.write(json.dumps(record, allow_nan=False) + "\n")
        self.file.flush()

    def result(self, values: dict[str, float]) -> None:
        if self.finished or values.keys() != METRICS.keys():
            raise ValueError("Duplicate result or mismatched metric schema")
        if any(
            type(v) not in (int, float) or not math.isfinite(v) for v in values.values()
        ):
            raise ValueError("Metrics must be finite numbers")
        self.write({"kind": "result", "values": values})
        self.finished = True

    def error(self, message: str) -> None:
        if not self.finished:
            self.write({"kind": "error", "message": message})
            self.finished = True

    def close(self) -> None:
        self.file.close()
