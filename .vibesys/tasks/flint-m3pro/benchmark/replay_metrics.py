"""Fixed-denominator RF measurements, with explicit deadline censoring."""

import math

__all__ = ["BUDGET_SECONDS", "WORST_MS", "replay_window", "summarize"]

BUDGET_SECONDS = 600
WORST_MS = BUDGET_SECONDS * 1000


def replay_window(records, *, log_created_timestamp, watchdog_timed_out):
    """Use RF's first submission as the authoritative start of scored work.

    The async log writer's file creation is a watchdog approximation. Reconcile
    against actual RF timestamps so scheduler delay cannot admit extra work.
    With no outcome the entire planned trace is censored at watchdog expiry.
    """
    if not _finite(log_created_timestamp):
        raise ValueError("Invalid RF log creation timestamp")
    for record in records:
        outcome = record["outcome"]
        if not all(
            _finite(outcome.get(key))
            for key in ("submit_timestamp", "complete_timestamp")
        ):
            raise ValueError("Invalid RF request timestamp")
        if outcome["complete_timestamp"] < outcome["submit_timestamp"]:
            raise ValueError("RF completion precedes submission")
    start = min(
        (r["outcome"]["submit_timestamp"] for r in records),
        default=log_created_timestamp,
    )
    deadline = start + BUDGET_SECONDS
    expired = watchdog_timed_out or any(
        r["outcome"]["complete_timestamp"] > deadline for r in records
    )
    return start, deadline, expired


def summarize(
    planned, records, *, timed_out=False, deadline_timestamp=None, rf_summary=None
):
    """Map RF session measurements to fixed-denominator protocol-v2 metrics.

    Invalid/short/error responses fail closed. Only an actual global deadline
    authorizes missing or late rounds. RF summary values must agree with the
    request log before uncensored means are used as objectives.
    """
    if timed_out != (deadline_timestamp is not None):
        raise ValueError("A timeout requires a finite replay deadline")
    if deadline_timestamp is not None and not _finite(deadline_timestamp):
        raise ValueError("Invalid replay deadline")
    expected = {row["request_id"]: row for row in planned}
    if len(expected) != len(planned) or not expected:
        raise ValueError("Trace request IDs must be unique and nonempty")
    measured = {}
    for record in records:
        outcome = record["outcome"]
        key = outcome["request_id"]
        if key not in expected or key in measured:
            raise ValueError(f"Unexpected or duplicate RF request: {key}")
        if not _finite(outcome.get("complete_timestamp")):
            raise ValueError(f"Invalid RF completion timestamp: {key}")
        measured[key] = outcome
    rows = []
    for key, plan in expected.items():
        outcome = measured.get(key)
        censored = outcome is None or (
            deadline_timestamp is not None
            and outcome["complete_timestamp"] > deadline_timestamp
        )
        if outcome is not None and outcome.get("error") is not None:
            if not timed_out:
                raise ValueError(f"RF request failed: {key}: {outcome['error']}")
            # Only the request crossing the global deadline may fail as a
            # consequence of owned-server shutdown. Earlier errors stay errors.
            if outcome["complete_timestamp"] <= deadline_timestamp:
                raise ValueError(f"RF request failed before deadline: {key}")
            censored = True
        if censored:
            if not timed_out:
                raise ValueError(f"Missing RF completion: {key}")
            row = dict(
                request_id=key,
                status="timed_out",
                censored=True,
                ttft_ms=WORST_MS,
                tpot_ms=WORST_MS,
                decode_tokens_per_second=0.0,
            )
        else:
            if outcome.get("status") != "SUCCESS":
                raise ValueError(f"RF request not successful: {key}")
            if outcome["output_len_actual"] != int(plan["output_len"]):
                raise ValueError(f"RF output length mismatch: {key}")
            ttft = outcome.get("first_token_id_ms")
            if ttft is None:
                raise ValueError(f"RF token-ID TTFT missing: {key}")
            tpot = outcome.get("token_delivery_tpot_ms")
            if any(
                type(v) not in (int, float)
                or not math.isfinite(v)
                or v < 0
                or v > WORST_MS
                for v in (ttft, tpot)
            ):
                raise ValueError(f"Invalid RF timing: {key}")
            # Reject a batch carrying the entire answer at one instant: it has
            # no measurable delivery TPOT and must not create infinite speed.
            if tpot == 0:
                raise ValueError(f"Unmeasurable RF decode timing: {key}")
            row = dict(
                request_id=key,
                status="completed",
                censored=False,
                ttft_ms=ttft,
                tpot_ms=tpot,
                decode_tokens_per_second=1000 / tpot,
                rf_outcome=outcome,
            )
        rows.append(row)
    n = len(rows)
    completed = sum(not row["censored"] for row in rows)
    values = {
        "ttft_ms_avg": sum(row["ttft_ms"] for row in rows) / n,
        "tpot_ms_avg": sum(row["tpot_ms"] for row in rows) / n,
        "decode_tokens_per_second": sum(row["decode_tokens_per_second"] for row in rows)
        / n,
        "completed_requests": completed,
        "timed_out_requests": n - completed,
    }
    if rf_summary is not None:
        replay = rf_summary["replay"]
        if replay["kind"] != "sessions":
            raise ValueError("RF summary must describe sessions")
        common = replay["common"]
        if common["attempted_steps"] != len(records):
            raise ValueError("RF summary attempted_steps differs from request log")
        if not timed_out:
            if common["success_steps"] != n or common["failed_steps"] != 0:
                raise ValueError("RF summary contains failed or missing steps")
            if common["output_mismatch_steps"] != 0:
                raise ValueError("RF summary contains output mismatches")
            for metric in ("ttft_ms_avg", "tpot_ms_avg"):
                measured_mean = common[metric]
                if not _finite(measured_mean) or not math.isclose(
                    values[metric], measured_mean, rel_tol=1e-9, abs_tol=1e-6
                ):
                    raise ValueError(f"RF summary metric disagrees with log: {metric}")
                values[metric] = measured_mean
    return values, rows


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)
