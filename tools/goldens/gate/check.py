"""Compute Flint gate metrics and apply a pinned threshold file.

Reference-only cases enter `suite_limits`. Native traces enter `evaluate_traces`
or `evaluate_records` after the thresholds already exist. Nothing in this
module reads a qualified flag.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType

import numpy as np

HIDDEN = 2048
VOCAB = 150144
EXPERT_COUNT = 256
TOP_K = 6
N_LAYERS = 49
MLP_ONLY_LAYERS = frozenset({0, 1})
NUMERIC_METRICS = (
    "max_abs",
    "p99_abs",
    "mean_abs",
    "relative_l2_max",
    "relative_l2_p99",
    "relative_l2_mean",
    "cosine_distance_max",
    "cosine_distance_mean",
)
LOGIT_METRICS = NUMERIC_METRICS + ("top1_error_max", "top2_gap_error_max")
ROUTING_METRICS = ("set_disagreement", "expert_overlap_loss")
_ABS_TOLERANCE = 1e-12


class GateError(ValueError):
    """The gate file itself is not a usable contract."""


@dataclass(frozen=True)
class Failure:
    """One closed failure. `path` names the tensor, identity, or case."""

    code: str
    path: str
    detail: str = ""


@dataclass(frozen=True)
class Verdict:
    """Pass only when `failures` is empty. There is no qualified override."""

    passed: bool
    failures: tuple[Failure, ...]
    summaries: Mapping[str, Mapping[str, float]]

    def __post_init__(self) -> None:
        if self.passed and self.failures:
            raise ValueError("a passing verdict cannot carry failures")
        if not self.passed and not self.failures:
            raise ValueError("a failing verdict needs a failure")


@dataclass(frozen=True)
class ReferenceCase:
    """One complete reference-to-reference comparison.

    `summary` values are deviations already computed from that pair. Native
    measurements do not belong here.
    """

    case_id: str
    length: int
    summary: Mapping[str, Mapping[str, float]]


@dataclass(frozen=True)
class _Measurement:
    failures: tuple[Failure, ...]
    summaries: Mapping[str, Mapping[str, float]]
    margins: np.ndarray | None
    mismatches: np.ndarray | None


@lru_cache(maxsize=1)
def expected_tensors() -> Mapping[str, tuple[str, int]]:
    """Protected tensor names, kinds, and widths. Kinds are numeric, logits, routing."""

    spec: dict[str, tuple[str, int]] = {
        "embed_tokens.h": ("numeric", HIDDEN),
        "logits": ("logits", VOCAB),
    }
    for layer in range(N_LAYERS):
        spec[f"layers.{layer}.self_attn.h"] = ("numeric", HIDDEN)
        spec[f"layers.{layer}.mlp.h"] = ("numeric", HIDDEN)
        if layer not in MLP_ONLY_LAYERS:
            spec[f"layers.{layer}.mlp.router.gate.y"] = ("numeric", EXPERT_COUNT)
            spec[f"layers.{layer}.mlp.ids"] = ("routing", TOP_K)
    if len(spec) != 194:
        raise GateError(f"protected tensor set has {len(spec)} names, expected 194")
    return MappingProxyType(spec)


def pinned_input_ids(
    gate: Mapping, example: int, length: int | None = None
) -> list[int]:
    """Teacher-forced ids: prompt plus every committed greedy id except the last.

    A length repeats example 1's short ids and truncates. Other examples have
    no long-context form.
    """

    prompts = gate["prompts"]
    if example < 0 or example >= len(prompts):
        raise GateError(f"prompt example {example} is outside the pinned set")
    prompt = prompts[example]
    ids = list(prompt["prompt_ids"]) + list(prompt["greedy_ids"][:-1])
    if length is None or length == len(ids):
        return ids
    if length <= 0:
        raise GateError("pinned length must be positive")
    if example != 1:
        raise GateError("long contexts repeat example 1 only")
    return (ids * ((length + len(ids) - 1) // len(ids)))[:length]


def load_gate(path: Path | str) -> dict:
    """Load a gate file. Missing contract keys fail before any trace is judged."""

    gate = json.loads(Path(path).read_text())
    if not isinstance(gate, dict):
        raise GateError("gate file must be an object")
    required = (
        "schema_version",
        "suite_limits",
        "stable_margin",
        "identities",
        "prompts",
    )
    missing = [key for key in required if key not in gate]
    if missing:
        raise GateError(f"gate file missing {missing}")
    if gate["schema_version"] != 1:
        raise GateError(f"unsupported gate schema {gate['schema_version']}")
    margin = gate["stable_margin"]
    if not isinstance(margin, dict) or "threshold" not in margin:
        raise GateError("stable_margin.threshold is required")
    if not isinstance(gate["prompts"], list) or len(gate["prompts"]) != 3:
        raise GateError("prompts must be the three pinned fixtures")
    return gate


def suite_limits(
    cases: Sequence[ReferenceCase],
    free_run_margins: Sequence[tuple[str, float]] = (),
) -> dict:
    """Twice the largest reference deviation, per tensor and scalar.

    The suite pools every supplied case. Per-length extrema are recorded beside
    it and do not replace it. `free_run_margins` are reference-only first
    divergence gaps. The stable margin is the minimum of twice the largest
    winner-change margin and twice the largest top-two gap error.
    """

    if not cases:
        raise GateError("reference calibration has no cases")
    for label, margin in free_run_margins:
        if not np.isfinite(margin):
            raise GateError(f"{label} free-run margin is non-finite")
    _require_complete_cases(cases)
    suite, suite_flip, suite_gap = _extrema(cases)
    # Free-run labels compete with teacher-forced flips for the winner-change source.
    flip_deviation, flip_source = suite_flip
    for label, margin in free_run_margins:
        if margin > flip_deviation:
            flip_deviation, flip_source = margin, label
    gap_deviation, gap_source = suite_gap
    per_length: dict[str, dict] = {}
    lengths = sorted({case.length for case in cases})
    for length in lengths:
        group = [case for case in cases if case.length == length]
        limits, _, _ = _extrema(group)
        per_length[str(length)] = limits
    return {
        "suite": suite,
        "per_length": per_length,
        "stable_margin": {
            "threshold": min(2 * flip_deviation, 2 * gap_deviation),
            "winner_change_margin": {
                "deviation": flip_deviation,
                "source": flip_source,
            },
            "top2_gap_error": {"deviation": gap_deviation, "source": gap_source},
        },
    }


def aggregate_records(records: Iterable[Mapping], *, rows: int) -> _Measurement:
    """Recompute distribution metrics from per-position record values.

    Stored max/mean fields are ignored. Routing uses the position-weighted
    chunk rates because those records do not carry expert ids; array checks
    own cardinality. Missing coverage, non-finite values, and overlapping rows
    fail closed.
    """

    if rows <= 0:
        return _failed(Failure("missing_rows", "*", "row count must be positive"))
    spec = expected_tensors()
    numeric_names = [name for name, (kind, _) in spec.items() if kind != "routing"]
    routing_names = [name for name, (kind, _) in spec.items() if kind == "routing"]
    pieces: dict[str, list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]] = {
        name: [] for name in numeric_names
    }
    routing: dict[str, list[tuple[int, int, float, float]]] = {
        name: [] for name in routing_names
    }
    logit_parts: list[tuple[int, np.ndarray, np.ndarray, float, float, float]] = []
    failures: list[Failure] = []
    for record in records:
        name = record.get("name")
        if name not in spec:
            failures.append(Failure("unexpected_tensor", str(name)))
            continue
        position = int(record.get("position", 0))
        kind, _width = spec[name]
        if kind == "routing":
            failure = _take_routing(record, name, position, routing)
            if failure:
                failures.append(failure)
            continue
        taken = _take_numeric(record, name, position, pieces)
        if isinstance(taken, Failure):
            failures.append(taken)
            continue
        if kind == "logits":
            logit = _take_logits(record, position, taken)
            if isinstance(logit, Failure):
                failures.append(logit)
            else:
                logit_parts.append(logit)
    if failures:
        return _Measurement(tuple(failures), {}, None, None)
    summaries: dict[str, dict[str, float]] = {}
    for name in numeric_names:
        failure, summary = _finish_numeric(name, pieces[name], rows)
        if failure:
            return _failed(failure)
        summaries[name] = summary
    for name in routing_names:
        failure, summary = _finish_routing(name, routing[name], rows)
        if failure:
            return _failed(failure)
        summaries[name] = summary
    failure, margins, mismatches, extras = _finish_logits(logit_parts, rows)
    if failure:
        return _failed(failure)
    summaries["logits"].update(extras)
    return _Measurement((), summaries, margins, mismatches)


def evaluate_records(
    records: Iterable[Mapping],
    gate: Mapping,
    *,
    example: int,
    token_ids: Sequence[int],
    identities: Mapping,
) -> Verdict:
    """Judge stored per-position measurements against a frozen gate."""

    return _judge_measurement(
        aggregate_records(records, rows=len(token_ids)),
        gate,
        example=example,
        token_ids=token_ids,
        identities=identities,
    )


def evaluate_traces(
    native: Mapping[str, np.ndarray],
    reference: Mapping[str, np.ndarray],
    gate: Mapping,
    *,
    example: int,
    token_ids: Sequence[int],
    identities: Mapping,
) -> Verdict:
    """Compute metrics from arrays and judge them. Structural problems fail closed."""

    return _judge_measurement(
        _measure_arrays(native, reference, rows=len(token_ids)),
        gate,
        example=example,
        token_ids=token_ids,
        identities=identities,
    )


def judge_continuation(
    first_divergence: int | None,
    reference_margin: float | None,
    stable_margin: float,
) -> Failure | None:
    """Accept a free-running first mismatch only at or below the frozen margin.

    Later tokens are a different history and are not another comparison.
    """

    if first_divergence is None:
        return None
    if reference_margin is None or not np.isfinite(reference_margin):
        return Failure("non_finite", "greedy", "divergence margin is missing")
    if reference_margin > stable_margin:
        return Failure(
            "stable_top1",
            "greedy",
            f"index {first_divergence} margin {reference_margin} exceeds {stable_margin}",
        )
    return None


def _judge_measurement(
    measurement: _Measurement,
    gate: Mapping,
    *,
    example: int,
    token_ids: Sequence[int],
    identities: Mapping,
) -> Verdict:
    failures = list(measurement.failures)
    failures.extend(_identity_failures(gate.get("identities", {}), identities))
    try:
        expected = pinned_input_ids(gate, example, len(token_ids))
    except GateError as exc:
        failures.append(Failure("provenance", "token_ids", str(exc)))
        expected = None
    if expected is not None and list(token_ids) != expected:
        failures.append(
            Failure("provenance", "token_ids", "ids differ from the pinned prompt")
        )
    if measurement.failures:
        return Verdict(False, tuple(failures), {})
    limits = _unwrap_limits(gate["suite_limits"])
    missing = [name for name in expected_tensors() if name not in limits]
    failures.extend(
        Failure("missing_tensor", name, "no frozen limit") for name in missing
    )
    if missing:
        return Verdict(False, tuple(failures), {})
    failures.extend(
        _judge_summaries(
            measurement.summaries,
            limits,
            margins=measurement.margins,
            mismatches=measurement.mismatches,
            stable_margin=float(gate["stable_margin"]["threshold"]),
        )
    )
    if failures:
        return Verdict(False, tuple(failures), measurement.summaries)
    return Verdict(True, (), measurement.summaries)


def _measure_arrays(
    native: Mapping[str, np.ndarray],
    reference: Mapping[str, np.ndarray],
    *,
    rows: int,
) -> _Measurement:
    if rows <= 0:
        return _failed(Failure("missing_rows", "*", "row count must be positive"))
    spec = expected_tensors()
    failures: list[Failure] = []
    for label, arrays in (("native", native), ("reference", reference)):
        unexpected = sorted(set(arrays) - set(spec))
        failures.extend(
            Failure("unexpected_tensor", name, label) for name in unexpected
        )
        missing = sorted(set(spec) - set(arrays))
        failures.extend(Failure("missing_tensor", name, label) for name in missing)
    if failures:
        return _Measurement(tuple(failures), {}, None, None)
    for name, (kind, width) in spec.items():
        failures.extend(
            _array_shape(name, kind, width, rows, reference[name], native[name])
        )
    if failures:
        return _Measurement(tuple(failures), {}, None, None)
    summaries: dict[str, dict[str, float]] = {}
    margins = None
    mismatches = None
    for name, (kind, _width) in spec.items():
        if kind == "routing":
            summaries[name] = _routing_summary(reference[name], native[name])
        else:
            summary, logit = _numeric_summary(
                reference[name], native[name], logits=kind == "logits"
            )
            summaries[name] = summary
            if logit is not None:
                margins, mismatches = logit
    return _Measurement((), summaries, margins, mismatches)


def _array_shape(
    name: str,
    kind: str,
    width: int,
    rows: int,
    reference: np.ndarray,
    native: np.ndarray,
) -> list[Failure]:
    failures: list[Failure] = []
    for label, array in (("reference", reference), ("native", native)):
        if array.ndim != 2:
            failures.append(Failure("shape", name, f"{label} shape {array.shape}"))
            continue
        if kind == "routing" and array.shape[1] != TOP_K:
            failures.append(
                Failure("routing_cardinality", name, f"{label} width {array.shape[1]}")
            )
            continue
        if array.shape[1] != width:
            failures.append(
                Failure(
                    "shape", name, f"{label} shape {array.shape} is not (*, {width})"
                )
            )
            continue
        if array.shape[0] != rows:
            failures.append(
                Failure("missing_rows", name, f"{label} has {array.shape[0]} rows")
            )
        if kind == "routing":
            failures.extend(_routing_ids(name, label, array))
        elif not np.all(np.isfinite(array)):
            failures.append(Failure("non_finite", name, label))
    if reference.shape != native.shape:
        failures.append(Failure("shape", name, "native and reference shapes differ"))
    return failures


def _routing_ids(name: str, label: str, array: np.ndarray) -> list[Failure]:
    if array.shape[1] != TOP_K:
        return [Failure("routing_cardinality", name, f"{label} width {array.shape[1]}")]
    if not np.issubdtype(array.dtype, np.integer):
        return [Failure("routing_cardinality", name, f"{label} ids are not integers")]
    ids = array.astype(np.int64, copy=False)
    failures: list[Failure] = []
    if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= EXPERT_COUNT):
        failures.append(Failure("expert_range", name, label))
    ordered = np.sort(ids, axis=1)
    if ordered.shape[1] > 1 and np.any(ordered[:, 1:] == ordered[:, :-1]):
        failures.append(Failure("duplicate_expert", name, label))
    return failures


def _numeric_summary(
    reference: np.ndarray, native: np.ndarray, *, logits: bool
) -> tuple[dict[str, float], tuple[np.ndarray, np.ndarray] | None]:
    actual = reference.astype(np.float64, copy=False)
    other = native.astype(np.float64, copy=False)
    diff = actual - other
    max_abs = np.max(np.abs(diff), axis=-1)
    relative = np.linalg.norm(diff, axis=-1) / np.maximum(
        np.linalg.norm(actual, axis=-1), 1e-30
    )
    cosine = np.sum(actual * other, axis=-1) / np.maximum(
        np.linalg.norm(actual, axis=-1) * np.linalg.norm(other, axis=-1), 1e-30
    )
    summary = _distribution(max_abs, relative, cosine)
    if not logits:
        return summary, None
    first = np.argmax(actual, axis=-1)
    runners = actual.copy()
    runners[np.arange(len(actual)), first] = -np.inf
    second = np.argmax(runners, axis=-1)
    index = np.arange(len(actual))
    margin = actual[index, first] - actual[index, second]
    mismatch = first != np.argmax(other, axis=-1)
    summary["top1_error_max"] = float(np.max(np.abs(diff[index, first])))
    summary["top2_gap_error_max"] = float(
        np.max(np.abs(diff[index, first] - diff[index, second]))
    )
    summary["mismatch_margin_max"] = (
        float(margin[mismatch].max()) if mismatch.any() else 0.0
    )
    return summary, (margin, mismatch)


def _routing_summary(reference: np.ndarray, native: np.ndarray) -> dict[str, float]:
    equal = np.sort(reference, axis=1) == np.sort(native, axis=1)
    agreement = float(np.mean(np.all(equal, axis=1)))
    overlap = np.zeros(len(reference), dtype=np.float64)
    for slot in range(reference.shape[1]):
        overlap += np.any(native == reference[:, slot : slot + 1], axis=1)
    return {
        "set_disagreement": float(1 - agreement),
        "expert_overlap_loss": float(1 - np.mean(overlap / TOP_K)),
    }


def _distribution(
    max_abs: np.ndarray, relative: np.ndarray, cosine: np.ndarray
) -> dict[str, float]:
    distance = 1 - cosine
    return {
        "max_abs": float(np.max(max_abs)),
        "p99_abs": float(np.quantile(max_abs, 0.99, method="linear")),
        "mean_abs": float(np.mean(max_abs)),
        "relative_l2_max": float(np.max(relative)),
        "relative_l2_p99": float(np.quantile(relative, 0.99, method="linear")),
        "relative_l2_mean": float(np.mean(relative)),
        "cosine_distance_max": float(np.max(distance)),
        "cosine_distance_mean": float(np.mean(distance)),
    }


def _judge_summaries(
    summaries: Mapping[str, Mapping[str, float]],
    limits: Mapping[str, Mapping[str, float]],
    *,
    margins: np.ndarray | None,
    mismatches: np.ndarray | None,
    stable_margin: float,
) -> list[Failure]:
    failures: list[Failure] = []
    for name, bounds in limits.items():
        observed = summaries.get(name)
        if observed is None:
            failures.append(Failure("missing_tensor", name))
            continue
        for metric, limit in bounds.items():
            actual = observed.get(metric)
            if (
                actual is None
                or not np.isfinite(actual)
                or actual > limit + _ABS_TOLERANCE
            ):
                failures.append(
                    Failure("metric", name, f"{metric} {actual} exceeds {limit}")
                )
    if margins is None or mismatches is None:
        failures.append(Failure("missing_rows", "logits", "no margin vector"))
        return failures
    if not np.all(np.isfinite(margins)):
        failures.append(Failure("non_finite", "logits", "reference margin"))
        return failures
    bad = np.flatnonzero((margins > stable_margin) & mismatches)
    if len(bad):
        shown = ", ".join(str(int(index)) for index in bad[:8])
        failures.append(Failure("stable_top1", "logits", f"positions {shown}"))
    return failures


def _identity_failures(expected: Mapping, observed: Mapping) -> list[Failure]:
    failures: list[Failure] = []
    for key, value in expected.items():
        if key not in observed or observed[key] != value:
            failures.append(Failure("provenance", str(key)))
    return failures


def _unwrap_limits(spec: Mapping) -> dict[str, dict[str, float]]:
    limits: dict[str, dict[str, float]] = {}
    for name, metrics in spec.items():
        limits[name] = {}
        for metric, value in metrics.items():
            if isinstance(value, dict):
                if "threshold" not in value:
                    raise GateError(f"{name}.{metric} has no threshold")
                limits[name][metric] = float(value["threshold"])
            else:
                limits[name][metric] = float(value)
    return limits


def _require_complete_cases(cases: Sequence[ReferenceCase]) -> None:
    spec = expected_tensors()
    for case in cases:
        if case.length <= 0:
            raise GateError(f"{case.case_id} has no rows")
        missing = [name for name in spec if name not in case.summary]
        if missing:
            raise GateError(f"{case.case_id} missing {missing[0]}")
        for name, (kind, _width) in spec.items():
            metrics = case.summary[name]
            required = (
                ROUTING_METRICS
                if kind == "routing"
                else LOGIT_METRICS + ("mismatch_margin_max",)
                if kind == "logits"
                else NUMERIC_METRICS
            )
            absent = [metric for metric in required if metric not in metrics]
            if absent:
                raise GateError(f"{case.case_id} {name} missing {absent[0]}")
            for metric in required:
                if not np.isfinite(metrics[metric]):
                    raise GateError(f"{case.case_id} {name}.{metric} is non-finite")


def _extrema(
    cases: Sequence[ReferenceCase],
) -> tuple[dict, tuple[float, str], tuple[float, str]]:
    spec = expected_tensors()
    best: dict[tuple[str, str], tuple[float, str]] = {}
    flip = (0.0, cases[0].case_id)
    gap = (0.0, cases[0].case_id)
    for case in cases:
        flip_value = float(case.summary["logits"]["mismatch_margin_max"])
        if flip_value > flip[0]:
            flip = (flip_value, case.case_id)
        gap_value = float(case.summary["logits"]["top2_gap_error_max"])
        if gap_value > gap[0]:
            gap = (gap_value, case.case_id)
        for name, (kind, _width) in spec.items():
            metrics = (
                ROUTING_METRICS
                if kind == "routing"
                else LOGIT_METRICS
                if kind == "logits"
                else NUMERIC_METRICS
            )
            for metric in metrics:
                value = float(case.summary[name][metric])
                slot = best.get((name, metric))
                if slot is None or value > slot[0]:
                    best[(name, metric)] = (value, case.case_id)
    limits: dict[str, dict] = {}
    for (name, metric), (deviation, source) in best.items():
        limits.setdefault(name, {})[metric] = {
            "threshold": 2 * deviation,
            "deviation": deviation,
            "source": source,
        }
    return limits, flip, gap


def _take_routing(
    record: Mapping,
    name: str,
    position: int,
    routing: dict[str, list[tuple[int, int, float, float]]],
) -> Failure | None:
    rows = record.get("rows")
    agreement = record.get("set_agreement")
    overlap = record.get("expert_overlap")
    if rows is None or agreement is None or overlap is None:
        return Failure("missing_rows", name, "routing record is incomplete")
    if not np.isfinite(agreement) or not np.isfinite(overlap):
        return Failure("non_finite", name)
    if not 0 <= float(agreement) <= 1 or not 0 <= float(overlap) <= 1:
        return Failure("routing_cardinality", name, "agreement outside [0, 1]")
    routing[name].append((position, int(rows), float(agreement), float(overlap)))
    return None


def _take_numeric(
    record: Mapping,
    name: str,
    position: int,
    pieces: dict[str, list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]],
) -> np.ndarray | Failure:
    try:
        absolute = np.asarray(record["per_position_max_abs"], dtype=np.float64)
        relative = np.asarray(record["per_position_relative_l2"], dtype=np.float64)
        cosine = np.asarray(record["per_position_cosine"], dtype=np.float64)
    except KeyError as exc:
        return Failure("missing_rows", name, f"missing {exc.args[0]}")
    rows = int(record.get("rows", len(absolute)))
    if len(absolute) != rows or len(relative) != rows or len(cosine) != rows:
        return Failure("missing_rows", name, "per-position length differs from rows")
    if not (
        np.all(np.isfinite(absolute))
        and np.all(np.isfinite(relative))
        and np.all(np.isfinite(cosine))
    ):
        return Failure("non_finite", name)
    pieces[name].append((position, absolute, relative, cosine))
    return absolute


def _take_logits(
    record: Mapping, position: int, absolute: np.ndarray
) -> tuple[int, np.ndarray, np.ndarray, float, float, float] | Failure:
    try:
        margins = np.asarray(record["reference_margins"], dtype=np.float64)
        top1 = record["top1_error_max"]
        gap = record["top2_gap_error_max"]
        mismatch_margin = record["mismatch_margin_max"]
        indices = record["top1_mismatches"]
    except KeyError as exc:
        return Failure("missing_rows", "logits", f"missing {exc.args[0]}")
    if len(margins) != len(absolute) or not np.all(np.isfinite(margins)):
        return Failure(
            "non_finite" if len(margins) == len(absolute) else "missing_rows", "logits"
        )
    if not all(np.isfinite(value) for value in (top1, gap, mismatch_margin)):
        return Failure("non_finite", "logits")
    mask = np.zeros(len(absolute), dtype=bool)
    for index in indices:
        if index < 0 or index >= len(absolute):
            return Failure("shape", "logits", f"top-1 mismatch index {index}")
        mask[index] = True
    return position, margins, mask, float(top1), float(gap), float(mismatch_margin)


def _finish_numeric(
    name: str,
    pieces: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]],
    rows: int,
) -> tuple[Failure | None, dict[str, float]]:
    failure, absolute, relative, cosine = _assemble(name, pieces, rows)
    if failure:
        return failure, {}
    return None, _distribution(absolute, relative, cosine)


def _finish_routing(
    name: str, pieces: list[tuple[int, int, float, float]], rows: int
) -> tuple[Failure | None, dict[str, float]]:
    if not pieces:
        return Failure("missing_tensor", name), {}
    pieces = sorted(pieces, key=lambda item: item[0])
    covered = np.zeros(rows, dtype=bool)
    disagree = 0.0
    overlap_loss = 0.0
    total = 0
    for position, count, agreement, overlap in pieces:
        if (
            count <= 0
            or position < 0
            or position + count > rows
            or covered[position : position + count].any()
        ):
            return Failure(
                "missing_rows", name, "routing coverage is not a partition"
            ), {}
        covered[position : position + count] = True
        disagree += (1 - agreement) * count
        overlap_loss += (1 - overlap) * count
        total += count
    if not covered.all() or total != rows:
        return Failure("missing_rows", name, "routing rows do not cover the trace"), {}
    return None, {
        "set_disagreement": disagree / rows,
        "expert_overlap_loss": overlap_loss / rows,
    }


def _finish_logits(
    parts: list[tuple[int, np.ndarray, np.ndarray, float, float, float]], rows: int
) -> tuple[Failure | None, np.ndarray | None, np.ndarray | None, dict[str, float]]:
    if not parts:
        return Failure("missing_tensor", "logits"), None, None, {}
    margins = np.zeros(rows, dtype=np.float64)
    mismatches = np.zeros(rows, dtype=bool)
    covered = np.zeros(rows, dtype=bool)
    top1 = 0.0
    gap = 0.0
    mismatch_margin = 0.0
    for position, margin, mask, top1_error, gap_error, flip_margin in sorted(
        parts, key=lambda item: item[0]
    ):
        count = len(margin)
        if (
            position < 0
            or position + count > rows
            or covered[position : position + count].any()
        ):
            return (
                Failure("missing_rows", "logits", "logit coverage is not a partition"),
                None,
                None,
                {},
            )
        covered[position : position + count] = True
        margins[position : position + count] = margin
        mismatches[position : position + count] = mask
        top1 = max(top1, top1_error)
        gap = max(gap, gap_error)
        mismatch_margin = max(mismatch_margin, flip_margin)
    if not covered.all():
        return Failure("missing_rows", "logits"), None, None, {}
    return (
        None,
        margins,
        mismatches,
        {
            "top1_error_max": top1,
            "top2_gap_error_max": gap,
            "mismatch_margin_max": mismatch_margin,
        },
    )


def _assemble(
    name: str,
    pieces: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]],
    rows: int,
) -> tuple[Failure | None, np.ndarray, np.ndarray, np.ndarray]:
    empty = np.zeros(0)
    if not pieces:
        return Failure("missing_tensor", name), empty, empty, empty
    absolute = np.zeros(rows, dtype=np.float64)
    relative = np.zeros(rows, dtype=np.float64)
    cosine = np.zeros(rows, dtype=np.float64)
    covered = np.zeros(rows, dtype=bool)
    for position, abs_rows, rel_rows, cos_rows in sorted(
        pieces, key=lambda item: item[0]
    ):
        count = len(abs_rows)
        if (
            position < 0
            or position + count > rows
            or covered[position : position + count].any()
        ):
            return (
                Failure("missing_rows", name, "coverage is not a partition"),
                empty,
                empty,
                empty,
            )
        covered[position : position + count] = True
        absolute[position : position + count] = abs_rows
        relative[position : position + count] = rel_rows
        cosine[position : position + count] = cos_rows
    if not covered.all():
        return Failure("missing_rows", name), empty, empty, empty
    return None, absolute, relative, cosine


def _failed(failure: Failure) -> _Measurement:
    return _Measurement((failure,), {}, None, None)
