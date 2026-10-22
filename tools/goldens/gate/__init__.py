"""Pinned Flint accuracy gate.

Thresholds are an input. The checker computes metrics from traces and fails
closed on incomplete or out-of-contract data. A candidate-supplied qualified
flag is not read.
"""

from tools.goldens.gate.check import (
    Failure,
    GateError,
    ReferenceCase,
    Verdict,
    aggregate_records,
    evaluate_records,
    evaluate_traces,
    expected_tensors,
    judge_continuation,
    load_gate,
    pinned_input_ids,
    suite_limits,
)

__all__ = [
    "Failure",
    "GateError",
    "ReferenceCase",
    "Verdict",
    "aggregate_records",
    "evaluate_records",
    "evaluate_traces",
    "expected_tensors",
    "judge_continuation",
    "load_gate",
    "pinned_input_ids",
    "suite_limits",
]
