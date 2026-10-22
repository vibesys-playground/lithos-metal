"""Native lithos request phases as evaluator protocol-v2 informational metrics.

The benchmark owns the same server and quick RF replay. These are the server's native
phase wall times, not TTFT/TPOT estimates. A missing generation or phase fails capture.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

from vs_protocol import ResultWriter

# Server warmup also logs generations, before its readiness message. Only
# generations after that boundary belong to RF; do not assume a warmup count.
READY_MARKER = "lithos-metal ready: "

# Pinned RF 89dce4a sends exactly two cache-check completions before opening its
# workload log (src/backend/preflight.rs). Keep these as startup diagnostics and
# require the exact count; never silently assign a probe to a workload request.
PREFLIGHT_GENERATIONS = 2

PHASES = {
    "prefill_timing": "prefill_wall_ms",
    "decode_timing": "decode_wall_ms",
    "setup_timing": "setup_ms",
    "load_timing": "load_ms",
    "prefill_gpu_timing": "prefill_gpu_ms",
    "decode_gpu_timing": "decode_gpu_ms",
    "state_reset_timing": "state_reset_ms",
    "checkpoint_timing": "checkpoint_ms",
    "prefill_encode_timing": "prefill_encode_ms",
    "prefill_commit_timing": "prefill_commit_ms",
    "prefill_wait_timing": "prefill_wait_ms",
}


def _number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"Missing or invalid native phase {name}")
    return float(value)


def profile_values(planned, records, server_logs, startup_seconds):
    """Map every completed RF request in order to exactly one native generation."""
    expected = [row["request_id"] for row in planned]
    observed = [row["outcome"]["request_id"] for row in records]
    if expected != observed or any(
        row["outcome"].get("status") != "SUCCESS"
        or row["outcome"].get("error")
        or row["outcome"].get("output_len_actual") != int(plan["output_len"])
        for plan, row in zip(planned, records)
    ):
        raise ValueError("Profile requires every quick request to complete in trace order")
    generations = []
    ready = False
    for log in server_logs or ():
        for line in log.splitlines():
            if READY_MARKER in line:
                if ready:
                    raise ValueError("Duplicate native server readiness boundary")
                ready = True
            elif ready and "Generation: " in line:
                generations.append(ast.literal_eval(line.split("Generation: ", 1)[1]))
    if not ready:
        raise ValueError("Missing native server readiness boundary")
    if len(generations) != len(expected) + PREFLIGHT_GENERATIONS:
        raise ValueError("Native generation count differs from completed quick requests")
    values = {"startup_duration_s": _number(startup_seconds, "startup_duration_s")}
    for index, phases in enumerate(generations[:PREFLIGHT_GENERATIONS]):
        for field, native in PHASES.items():
            values[f"preflight_{index}.{field}"] = _number(phases.get(native), native)
    for index, phases in enumerate(generations[PREFLIGHT_GENERATIONS:]):
        for field, native in PHASES.items():
            value = _number(phases.get(native), native)
            values[f"request_{index}.{field}"] = value
            values[field] = values.get(field, 0.0) + value
    return values


class ProfileWriter(ResultWriter):
    """One declared metric per phase and request, followed by one protocol result."""

    def __init__(self, path: Path):
        # The schema depends on the quick trace's request count; publish it with the result.
        self.file = path.open("w", encoding="utf-8")
        self.finished = False

    def result(self, values):
        if self.finished or not values:
            raise ValueError("Duplicate or empty profile result")
        checked = {name: _number(value, name) for name, value in values.items()}
        self.write(
            {
                "kind": "hello",
                "protocol": 2,
                "metrics": {
                    name: {"unit": "s" if name == "startup_duration_s" else "ms"}
                    for name in checked
                },
            }
        )
        self.write({"kind": "result", "values": checked})
        self.finished = True


def synthetic_profile(planned):
    """Exercise the real phase decoder and writer with an explicitly synthetic engine."""
    records = [
        {
            "outcome": {
                "request_id": row["request_id"],
                "status": "SUCCESS",
                "error": None,
                "output_len_actual": int(row["output_len"]),
            }
        }
        for row in planned
    ]
    logs = [
        "INFO:monolith.serve:" + READY_MARKER + "synthetic server\n"
        + "\n".join(
            "INFO:monolith.serve:Generation: "
            + repr(
                {
                    native: float(index - PREFLIGHT_GENERATIONS + 1)
                    if index >= PREFLIGHT_GENERATIONS
                    else 0.0
                    for native in PHASES.values()
                }
            )
            for index, _ in enumerate([None] * PREFLIGHT_GENERATIONS + planned)
        )
    ]
    return profile_values(planned, records, logs, 0.0)
