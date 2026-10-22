"""Fixed-denominator RF measurements, with explicit deadline censoring."""

import math

BUDGET_SECONDS = 600
WORST_MS = BUDGET_SECONDS * 1000


def summarize(planned, records, *, timed_out=False, deadline_timestamp=None):
    expected = {row["request_id"]: row for row in planned}
    if len(expected) != len(planned) or not expected:
        raise ValueError("Trace request IDs must be unique and nonempty")
    measured = {}
    for record in records:
        outcome = record["outcome"]
        key = outcome["request_id"]
        if key not in expected or key in measured:
            raise ValueError(f"Unexpected or duplicate RF request: {key}")
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
    return {
        "ttft_ms_avg": sum(row["ttft_ms"] for row in rows) / n,
        "tpot_ms_avg": sum(row["tpot_ms"] for row in rows) / n,
        "decode_tokens_per_second": sum(row["decode_tokens_per_second"] for row in rows)
        / n,
        "completed_requests": completed,
        "timed_out_requests": n - completed,
    }, rows
