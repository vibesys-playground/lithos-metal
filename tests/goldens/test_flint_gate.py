"""Hermetic Flint gate checks. No GPU, clock, or stored trace is required."""

import json

import numpy as np
import pytest

from tools.goldens.gate import (
    Failure,
    GateError,
    ReferenceCase,
    aggregate_records,
    evaluate_records,
    evaluate_traces,
    expected_tensors,
    judge_continuation,
    load_gate,
    pinned_input_ids,
    suite_limits,
)

PROMPTS = [
    {"prompt_ids": [1], "greedy_ids": [2, 3, 4, 5]},
    {"prompt_ids": [8], "greedy_ids": [9, 10, 11, 12]},
    {"prompt_ids": [1], "greedy_ids": [2, 3, 4, 5]},
]
IDENTITIES = {"config_sha256": "config", "revision": "rev"}
ROWS = 4


def _ids():
    return [1, 2, 3, 4]


def _limits(floor):
    built = suite_limits([ReferenceCase("zero", ROWS, _summary(0, 0, 0))])
    return {
        name: {metric: floor for metric in metrics}
        for name, metrics in built["suite"].items()
    }


def _gate(floor=1.0, stable=0.0, **extra):
    gate = {
        "schema_version": 1,
        "suite_limits": _limits(floor),
        "stable_margin": {"threshold": stable},
        "identities": dict(IDENTITIES),
        "prompts": PROMPTS,
    }
    gate.update(extra)
    return gate


def _pair():
    native = {}
    reference = {}
    for name, (kind, width) in expected_tensors().items():
        if kind == "routing":
            ids = np.tile(np.arange(width, dtype=np.int32), (ROWS, 1))
            reference[name] = ids.copy()
            native[name] = ids.copy()
        else:
            values = np.ones((ROWS, width), np.float32)
            reference[name] = values.copy()
            native[name] = values.copy()
    return native, reference


def _evaluate(native, reference, gate=None, token_ids=None):
    return evaluate_traces(
        native,
        reference,
        _gate() if gate is None else gate,
        example=0,
        token_ids=_ids() if token_ids is None else token_ids,
        identities=IDENTITIES,
    )


def _codes(verdict):
    return {(failure.code, failure.path) for failure in verdict.failures}


def test_protected_tensor_set_has_the_contract_width():
    spec = expected_tensors()
    assert len(spec) == 194
    assert spec["logits"] == ("logits", 150144)
    assert spec["layers.0.mlp.h"] == ("numeric", 2048)
    assert "layers.0.mlp.ids" not in spec
    assert spec["layers.2.mlp.ids"] == ("routing", 6)
    assert spec["layers.48.mlp.router.gate.y"] == ("numeric", 256)


def test_identical_traces_pass_and_a_qualified_flag_is_ignored():
    native, reference = _pair()
    verdict = _evaluate(native, reference, _gate(qualified=False))
    assert verdict.passed
    assert verdict.failures == ()


def test_logit_error_fails_when_a_candidate_marks_itself_qualified():
    native, reference = _pair()
    native["logits"] = native["logits"].copy()
    native["logits"][:, 0] += 4
    verdict = _evaluate(native, reference, _gate(floor=1.0, qualified=True))
    assert not verdict.passed
    assert ("metric", "logits") in _codes(verdict)


@pytest.mark.parametrize(
    ("mutate", "code", "path"),
    [
        ("drop", "missing_tensor", "embed_tokens.h"),
        ("extra", "unexpected_tensor", "extra"),
        ("width", "shape", "embed_tokens.h"),
        ("rows", "missing_rows", "embed_tokens.h"),
        ("nan", "non_finite", "embed_tokens.h"),
        ("cardinality", "routing_cardinality", "layers.2.mlp.ids"),
        ("duplicate", "duplicate_expert", "layers.2.mlp.ids"),
        ("range", "expert_range", "layers.2.mlp.ids"),
    ],
)
def test_trace_defects_fail_closed(mutate, code, path):
    native, reference = _pair()
    if mutate == "drop":
        del native["embed_tokens.h"]
    elif mutate == "extra":
        native["extra"] = np.ones((ROWS, 1), np.float32)
    elif mutate == "width":
        native["embed_tokens.h"] = np.ones((ROWS, 7), np.float32)
    elif mutate == "rows":
        native["embed_tokens.h"] = np.ones((ROWS - 1, 2048), np.float32)
    elif mutate == "nan":
        native["embed_tokens.h"] = native["embed_tokens.h"].copy()
        native["embed_tokens.h"][0, 0] = np.nan
    elif mutate == "cardinality":
        native["layers.2.mlp.ids"] = np.tile(np.arange(5, dtype=np.int32), (ROWS, 1))
    elif mutate == "duplicate":
        native["layers.2.mlp.ids"] = native["layers.2.mlp.ids"].copy()
        native["layers.2.mlp.ids"][:, 1] = native["layers.2.mlp.ids"][:, 0]
    else:
        native["layers.2.mlp.ids"] = native["layers.2.mlp.ids"].copy()
        native["layers.2.mlp.ids"][:, 0] = 256
    verdict = _evaluate(native, reference)
    assert not verdict.passed
    assert (code, path) in _codes(verdict)


def test_provenance_mismatch_fails_closed():
    native, reference = _pair()
    wrong_ids = _evaluate(native, reference, token_ids=[1, 2, 3, 9])
    wrong_identity = evaluate_traces(
        native,
        reference,
        _gate(),
        example=0,
        token_ids=_ids(),
        identities={"config_sha256": "other", "revision": "rev"},
    )
    assert ("provenance", "token_ids") in _codes(wrong_ids)
    assert ("provenance", "config_sha256") in _codes(wrong_identity)


def test_stable_top1_requires_a_strictly_larger_reference_margin():
    native, reference = _pair()
    reference["logits"] = np.zeros((ROWS, 150144), np.float32)
    reference["logits"][:, 1] = 1
    reference["logits"][0, 1] = 1
    reference["logits"][0, 0] = 0.5
    native["logits"] = reference["logits"].copy()
    native["logits"][0, 0] = 2
    native["logits"][0, 1] = 0
    at_threshold = _evaluate(native, reference, _gate(floor=100, stable=0.5))
    assert at_threshold.passed
    above = _evaluate(native, reference, _gate(floor=100, stable=0.49))
    assert ("stable_top1", "logits") in _codes(above)


def test_records_recompute_per_position_error_and_ignore_a_stored_maximum():
    records = _records(abs_value=5)
    for record in records:
        if record["name"] == "embed_tokens.h":
            record["max_abs"] = 0
    measurement = aggregate_records(records, rows=ROWS)
    assert measurement.failures == ()
    assert measurement.summaries["embed_tokens.h"]["max_abs"] == 5
    assert measurement.summaries["embed_tokens.h"]["p99_abs"] == pytest.approx(5)
    verdict = evaluate_records(
        records,
        _gate(floor=1.0),
        example=0,
        token_ids=_ids(),
        identities=IDENTITIES,
    )
    assert ("metric", "embed_tokens.h") in _codes(verdict)


def test_record_coverage_gaps_overlaps_and_non_finite_values_fail():
    short = _records(abs_value=0)
    for record in short:
        if record["name"] == "embed_tokens.h":
            record["rows"] = ROWS - 1
            record["per_position_max_abs"] = [0.0] * (ROWS - 1)
            record["per_position_relative_l2"] = [0.0] * (ROWS - 1)
            record["per_position_cosine"] = [1.0] * (ROWS - 1)
    assert aggregate_records(short, rows=ROWS).failures[0].code == "missing_rows"

    overlap = _records(abs_value=0)
    extra = dict(overlap[0])
    overlap.append(extra)
    assert aggregate_records(overlap, rows=ROWS).failures[0].code == "missing_rows"

    poisoned = _records(abs_value=0)
    poisoned[0]["per_position_max_abs"][0] = float("nan")
    assert aggregate_records(poisoned, rows=ROWS).failures[0].code == "non_finite"


def test_omitted_routing_rows_use_the_sibling_chunk_and_a_gap_fails():
    inferred = _records(abs_value=0)
    for record in inferred:
        if record["name"].endswith(".ids"):
            del record["rows"]
    measurement = aggregate_records(inferred, rows=ROWS)
    assert measurement.failures == ()
    assert measurement.summaries["layers.2.mlp.ids"]["set_disagreement"] == 0

    orphaned = _records(abs_value=0)
    orphaned.append(
        {
            "name": "layers.2.mlp.ids",
            "position": 99,
            "set_agreement": 1.0,
            "expert_overlap": 1.0,
        }
    )
    assert aggregate_records(orphaned, rows=ROWS).failures[0].code == "missing_rows"


def test_routing_rates_are_weighted_by_rows_and_p99_uses_linear_interpolation():
    records = _records(abs_value=0)
    records = [record for record in records if record["name"] != "layers.2.mlp.ids"]
    records.append(
        {
            "name": "layers.2.mlp.ids",
            "position": 0,
            "rows": 1,
            "set_agreement": 0.0,
            "expert_overlap": 1.0,
        }
    )
    records.append(
        {
            "name": "layers.2.mlp.ids",
            "position": 1,
            "rows": 3,
            "set_agreement": 1.0,
            "expert_overlap": 1.0,
        }
    )
    for record in records:
        if record["name"] == "embed_tokens.h":
            record["per_position_max_abs"] = [0.0, 10.0, 10.0, 10.0]
    measurement = aggregate_records(records, rows=ROWS)
    assert measurement.failures == ()
    assert measurement.summaries["layers.2.mlp.ids"][
        "set_disagreement"
    ] == pytest.approx(0.25)
    # Two-point linear p99 is not the maximum. This series is four points.
    series = np.array([0.0, 10.0, 10.0, 10.0])
    assert measurement.summaries["embed_tokens.h"]["p99_abs"] == pytest.approx(
        float(np.quantile(series, 0.99, method="linear"))
    )


def test_suite_limits_double_the_largest_reference_deviation_only():
    small = ReferenceCase("short", 4, _summary(max_abs=1, flip=0.1, gap=0.05))
    large = ReferenceCase("long", 8, _summary(max_abs=3, flip=0.0, gap=0.0))
    tied = ReferenceCase("early", 4, _summary(max_abs=3, flip=0.0, gap=0.0))
    limits = suite_limits([small, large], [("greedy-e2-ref1", 0.4)])
    logit = limits["suite"]["logits"]["max_abs"]
    assert logit == {"threshold": 6, "deviation": 3, "source": "long"}
    assert limits["per_length"]["4"]["logits"]["max_abs"]["threshold"] == 2
    assert limits["per_length"]["8"]["logits"]["max_abs"]["threshold"] == 6
    margin = limits["stable_margin"]
    assert margin["threshold"] == pytest.approx(0.1)
    assert margin["winner_change_margin"] == {
        "deviation": 0.4,
        "source": "greedy-e2-ref1",
    }
    assert margin["top2_gap_error"]["source"] == "short"
    tied_limits = suite_limits([tied, large])
    assert tied_limits["suite"]["logits"]["max_abs"]["source"] == "early"


def test_long_context_ids_repeat_example_1_only():
    gate = _gate()
    assert pinned_input_ids(gate, 1, 4) == [8, 9, 10, 11]
    assert pinned_input_ids(gate, 1, 6) == [8, 9, 10, 11, 8, 9]
    with pytest.raises(GateError):
        pinned_input_ids(gate, 0, 6)


def test_continuation_accepts_only_a_margin_at_or_below_the_threshold():
    assert judge_continuation(None, None, 0.2) is None
    assert judge_continuation(4, 0.2, 0.2) is None
    failure = judge_continuation(4, 0.3, 0.2)
    assert isinstance(failure, Failure)
    assert failure.code == "stable_top1"
    assert judge_continuation(4, None, 0.2).code == "non_finite"


def test_load_gate_rejects_a_missing_contract_key(tmp_path):
    path = tmp_path / "gate.json"
    path.write_text(json.dumps({"schema_version": 1}))
    with pytest.raises(GateError):
        load_gate(path)


def _records(abs_value):
    records = []
    for name, (kind, _width) in expected_tensors().items():
        if kind == "routing":
            records.append(
                {
                    "name": name,
                    "position": 0,
                    "rows": ROWS,
                    "set_agreement": 1.0,
                    "expert_overlap": 1.0,
                }
            )
            continue
        record = {
            "name": name,
            "position": 0,
            "rows": ROWS,
            "per_position_max_abs": [abs_value] * ROWS,
            "per_position_relative_l2": [0.0] * ROWS,
            "per_position_cosine": [1.0] * ROWS,
        }
        if kind == "logits":
            record.update(
                reference_margins=[0.0] * ROWS,
                top1_mismatches=[],
                top1_error_max=0.0,
                top2_gap_error_max=0.0,
                mismatch_margin_max=0.0,
            )
        records.append(record)
    return records


def _summary(max_abs, flip, gap):
    summary = {}
    numeric = (
        "max_abs",
        "p99_abs",
        "mean_abs",
        "relative_l2_max",
        "relative_l2_p99",
        "relative_l2_mean",
        "cosine_distance_max",
        "cosine_distance_mean",
    )
    for name, (kind, _width) in expected_tensors().items():
        if kind == "routing":
            summary[name] = {"set_disagreement": 0.0, "expert_overlap_loss": 0.0}
        else:
            summary[name] = {metric: 0.0 for metric in numeric}
            if kind == "logits":
                summary[name]["max_abs"] = max_abs
                summary[name]["top1_error_max"] = 0.0
                summary[name]["top2_gap_error_max"] = gap
                summary[name]["mismatch_margin_max"] = flip
    return summary
