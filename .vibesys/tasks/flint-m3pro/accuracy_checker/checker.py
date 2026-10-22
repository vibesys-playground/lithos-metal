#!/usr/bin/env python3
"""Interim accuracy gate: source audit, semantic probes, margin-aware pins, 1k logits.

Published functions accept a faithful in-process Chat, pinned tokenizer and
caller-supplied arrays. Thresholds live only in interim_gate_v0.json. No
alternate reference, skip flag or candidate-supplied trace is accepted.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Protocol

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
from launcher import MODEL, REVISION, HttpChat, PinnedTokenizer, model_path, server_endpoint
from pack_policy import failure_exit_status

__all__ = [
    "load_pins",
    "load_gate",
    "load_pin_margins",
    "load_reference_logits",
    "check_accuracy",
    "compare_logits",
    "scan_engine",
    "PROBES",
    "PinChat",
    "TRACE_ROWS",
    "TRACE_VOCAB",
    "REFERENCE_TRACE_NAME",
]
PIN_LENGTH = 32
MIN_PINS = 4
TRACE_ROWS = 1024
TRACE_VOCAB = 150144
REFERENCE_TRACE_NAME = "rec-bf16-chunk32-logits-1024.npy"
# Twice the rec BF16 chunk-1 versus chunk-32 figures in the port handoff.
APPROVED_THRESHOLDS = {
    "margin": 0.3125,
    "max_logit_abs": 22.5,
    "relative_l2_max": 0.19047058,
    "min_cosine": 0.99509772,
}
PREFIX_IDS_SHA256 = "4930907e7e0ea1a8412c9649855ae8a2e7467c3a8951033e507b5c6ba4878e4b"
FORBIDDEN = {"mlx", "mlx_lm", "rec", "rec_mlx", "llama_cpp", "ollama", "vllm", "sglang"}
PROBES = (
    ([{"role": "user", "content": "What is 17 + 25? Reply with only the integer."}], r"\b42\b"),
    (
        [
            {
                "role": "user",
                "content": "In Python, what does len([2, 4, 6, 8]) return? Reply with only the integer.",
            }
        ],
        r"\b4\b",
    ),
    (
        [
            {"role": "user", "content": "Remember the build marker: cobalt-7319."},
            {"role": "assistant", "content": "I will remember the marker."},
            {"role": "user", "content": "What was the exact build marker?"},
        ],
        r"cobalt-7319",
    ),
    (
        [
            {
                "role": "user",
                "content": "The deployment region is cedar-west and the shard count is 13.",
            },
            {"role": "assistant", "content": "Recorded."},
            {"role": "user", "content": "Repeat the deployment region and shard count."},
        ],
        r"(?s)(?=.*cedar-west)(?=.*\b13\b)",
    ),
)


class PinChat(Protocol):
    def greedy(self, messages: list[dict], max_tokens: int) -> dict: ...


def scan_engine(workspace: Path) -> None:
    """Reject mechanical forbidden imports/execs; renamed engines need judge review.

    Scans candidate Python outside tests/tools and model oracle.py references.
    Existing lazy Torch layer oracles are allowed as reference code. Whether
    served execution uses the compiler/runtime is additionally judge-reviewed.
    """
    for path in workspace.rglob("*.py"):
        relative = path.relative_to(workspace)
        if any(
            p.startswith(".") or p in {"tests", "tools", "build", "dist", "__pycache__"}
            for p in relative.parts
        ):
            continue
        if path.name == "oracle.py":
            continue
        tree = ast.parse(path.read_text(), filename=str(relative))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [n.name for n in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or "", *(n.name for n in node.names)]
            if any(n.split(".")[0] in FORBIDDEN for n in names):
                raise ValueError(f"Forbidden serving import in {relative}:{node.lineno}: {names}")
            if isinstance(node, ast.Call) and (
                (isinstance(node.func, ast.Name) and node.func.id in {"__import__", "exec", "eval"})
                or (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr
                    in {
                        "import_module",
                        "run",
                        "Popen",
                        "call",
                        "system",
                        "check_call",
                        "check_output",
                    }
                )
            ):
                literals = [
                    n.value
                    for n in ast.walk(node)
                    if isinstance(n, ast.Constant) and isinstance(n.value, str)
                ]
                if any(
                    re.search(r"\b(mlx[-_]lm|rec_mlx|llama[._-]cpp|ollama)\b", s) for s in literals
                ):
                    raise ValueError(
                        f"Possible forbidden engine exec/import in {relative}:{node.lineno}"
                    )


def load_pins(path: Path) -> list[dict]:
    if not path.is_file():
        raise ValueError(
            "Missing reference/pins.json; reuse flint-model goldens or run reference/make_pins.py under GPU lock"
        )
    data = json.loads(path.read_text())
    if set(data) != {"version", "model", "revision", "reference", "kv_format", "pins"}:
        raise ValueError("Unknown/missing pins keys")
    if (data["version"], data["model"], data["revision"], data["kv_format"]) != (
        1,
        MODEL,
        REVISION,
        "bf16",
    ):
        raise ValueError("Pins provenance does not match pinned int4 weights and bf16 KV")
    if not isinstance(data["reference"], dict) or not data["reference"].get("commit"):
        raise ValueError("Pins require independent reference commit provenance")
    pins = data["pins"]
    if not isinstance(pins, list) or len(pins) < MIN_PINS:
        raise ValueError(f"At least {MIN_PINS} independent pins required")
    seen = set()
    for pin in pins:
        if set(pin) != {"id", "messages", "prompt_token_ids", "expected_token_ids"}:
            raise ValueError("Unknown/missing pin keys")
        if not isinstance(pin["id"], str) or pin["id"] in seen:
            raise ValueError("Duplicate/invalid pin id")
        seen.add(pin["id"])
        for name in ("prompt_token_ids", "expected_token_ids"):
            if (
                not isinstance(pin[name], list)
                or not pin[name]
                or any(type(n) is not int or n < 0 for n in pin[name])
            ):
                raise ValueError(f"Invalid pin {pin['id']} {name}")
        if len(pin["expected_token_ids"]) != PIN_LENGTH:
            raise ValueError(f"Pin {pin['id']} must contain exactly {PIN_LENGTH} tokens")
        if not isinstance(pin["messages"], list) or not pin["messages"]:
            raise ValueError("Pin messages must be nonempty")
        for message in pin["messages"]:
            if (
                set(message) != {"role", "content"}
                or message["role"] not in {"system", "user", "assistant"}
                or not isinstance(message["content"], str)
            ):
                raise ValueError("Invalid pin messages")
    if len({json.dumps(p["messages"], sort_keys=True) for p in pins}) != len(pins):
        raise ValueError("Pins must use distinct prompts")
    return pins


def response_text(body: dict) -> str:
    if "error" in body or len(body.get("choices", [])) != 1:
        raise ValueError("Invalid completion response")
    choice = body["choices"][0]
    if choice["message"].get("tool_calls"):
        raise ValueError("Expected text, received tool calls")
    text = choice["message"].get("content")
    if not isinstance(text, str) or not text:
        raise ValueError("Empty completion")
    return text


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _hex_sha256(value, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"Invalid {label} sha256")
    return value


def _finite_margin(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Invalid margin {label}")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"Invalid margin {label}")
    return number


def gate_cache_dir() -> Path:
    return Path.home() / ".cache" / "lithos-metal" / "flint-gate"


def load_gate(path: Path) -> dict:
    """Load the committed interim gate. Thresholds are fixed; hashes are pinned."""
    if not path.is_file():
        raise ValueError("Missing accuracy_checker/interim_gate_v0.json")
    data = json.loads(path.read_text())
    if set(data) != {
        "version",
        "name",
        "thresholds",
        "prefix",
        "reference_logits",
        "pin_margins",
    }:
        raise ValueError("Unknown/missing interim gate keys")
    if data["version"] != 0 or data["name"] != "interim_gate_v0":
        raise ValueError("Interim gate identity mismatch")
    thresholds = data["thresholds"]
    if not isinstance(thresholds, dict) or set(thresholds) != set(APPROVED_THRESHOLDS):
        raise ValueError("Unknown/missing interim gate thresholds")
    for key, approved in APPROVED_THRESHOLDS.items():
        entry = thresholds[key]
        if set(entry) != {"value", "source"} or not isinstance(entry["source"], str) or not entry["source"]:
            raise ValueError(f"Invalid threshold entry {key}")
        if entry["value"] != approved:
            raise ValueError(f"Threshold {key} does not match the approved interim value")
    prefix = data["prefix"]
    if set(prefix) != {"length", "source", "ids_sha256", "ids"}:
        raise ValueError("Unknown/missing prefix keys")
    if prefix["length"] != TRACE_ROWS or prefix["ids_sha256"] != PREFIX_IDS_SHA256:
        raise ValueError("Teacher-forced prefix identity mismatch")
    ids = prefix["ids"]
    if not isinstance(ids, list) or len(ids) != TRACE_ROWS or any(type(n) is not int or n < 0 for n in ids):
        raise ValueError("Teacher-forced prefix ids are invalid")
    if hashlib.sha256(json.dumps(ids).encode()).hexdigest() != PREFIX_IDS_SHA256:
        raise ValueError("Teacher-forced prefix sha256 mismatch")
    logits = data["reference_logits"]
    if set(logits) != {"file", "schedule", "sha256", "shape", "dtype"}:
        raise ValueError("Unknown/missing reference logit keys")
    if logits["file"] != REFERENCE_TRACE_NAME or logits["dtype"] != "float32":
        raise ValueError("Reference logit trace identity mismatch")
    if logits["shape"] != [TRACE_ROWS, TRACE_VOCAB]:
        raise ValueError("Reference logit shape mismatch")
    _hex_sha256(logits["sha256"], "reference logit")
    margins = data["pin_margins"]
    if set(margins) != {"file", "schedule", "sha256"} or margins["file"] != "pin_margins.json":
        raise ValueError("Unknown/missing pin margin keys")
    _hex_sha256(margins["sha256"], "pin margin")
    return data


def load_pin_margins(path: Path, gate: dict, pins: list[dict]) -> dict[str, list[float]]:
    if not path.is_file():
        raise ValueError("Missing reference/pin_margins.json")
    digest = _sha256(path)
    if digest != gate["pin_margins"]["sha256"]:
        raise ValueError("Pin margin hash mismatch")
    data = json.loads(path.read_text())
    if set(data) != {"version", "reference_commit", "schedule", "pins"}:
        raise ValueError("Unknown/missing pin margin file keys")
    if data["version"] != 1 or not isinstance(data["reference_commit"], str) or not data["schedule"]:
        raise ValueError("Pin margin provenance mismatch")
    stored = data["pins"]
    expected_ids = [pin["id"] for pin in pins]
    if not isinstance(stored, dict) or set(stored) != set(expected_ids):
        raise ValueError("Pin margin ids do not match the greedy pins")
    margins = {}
    for pin_id in expected_ids:
        entry = stored[pin_id]
        if set(entry) != {"margins", "top1"}:
            raise ValueError(f"Unknown/missing margin keys for pin {pin_id}")
        values = entry["margins"]
        top1 = entry["top1"]
        if (
            not isinstance(values, list)
            or len(values) != PIN_LENGTH
            or not isinstance(top1, list)
            or len(top1) != PIN_LENGTH
            or any(type(n) is not int or n < 0 for n in top1)
        ):
            raise ValueError(f"Pin {pin_id} margins are incomplete")
        margins[pin_id] = [_finite_margin(value, pin_id) for value in values]
    return margins


def load_reference_logits(path: Path, gate: dict) -> np.ndarray:
    if not path.is_file():
        raise ValueError("Missing reference logit trace")
    digest = _sha256(path)
    if digest != gate["reference_logits"]["sha256"]:
        raise ValueError("Reference logit hash mismatch")
    logits = np.load(path, mmap_mode="r")
    if logits.shape != (TRACE_ROWS, TRACE_VOCAB) or logits.dtype != np.float32:
        raise ValueError(f"Reference logit shape mismatch: {logits.shape} {logits.dtype}")
    if not np.isfinite(logits).all():
        raise ValueError("Reference logit trace contains a non-finite value")
    return logits


def _matched_prefix(got: list[int], expected: list[int], margins: list[float], limit: float, pin_id: str) -> int:
    for index, (actual, wanted) in enumerate(zip(got, expected, strict=True)):
        if actual == wanted:
            continue
        margin = margins[index]
        if margin >= limit:
            raise ValueError(
                f"Greedy pin {pin_id} diverged at token {index}: margin {margin}"
            )
        return index
    return PIN_LENGTH


def check_accuracy(
    chat: PinChat,
    tokenizer: PinnedTokenizer,
    pins: list[dict],
    margins: dict[str, list[float]],
    *,
    margin_limit: float = APPROVED_THRESHOLDS["margin"],
) -> dict:
    for index, (messages, pattern) in enumerate(PROBES):
        body = chat.greedy(messages, 64)
        if not re.search(pattern, response_text(body), re.IGNORECASE):
            raise ValueError(f"Semantic probe {index} failed: {response_text(body)!r}")
    if set(margins) != {pin["id"] for pin in pins}:
        raise ValueError("Pin margin ids do not match the greedy pins")
    tolerated = []
    # Reverse order on the second pass exercises session replacement and stale
    # prefix state as well as repeated deterministic decoding.
    for pin in [*pins, *reversed(pins)]:
        if tokenizer.ids(pin["messages"]) != pin["prompt_token_ids"]:
            raise ValueError(f"Chat-template mismatch for pin {pin['id']}")
        expected = pin["expected_token_ids"]
        body = chat.greedy(pin["messages"], PIN_LENGTH)
        got = body.get("completion_token_ids")
        if not isinstance(got, list) or any(type(n) is not int or n < 0 for n in got):
            raise ValueError(
                "Non-streaming response requires raw completion_token_ids; text strips Flint special tokens"
            )
        if len(got) != PIN_LENGTH or body.get("usage", {}).get("completion_tokens") != PIN_LENGTH:
            raise ValueError(f"Greedy pin {pin['id']} diverged at token {min(len(got), PIN_LENGTH)}: short output")
        pin_margins = margins[pin["id"]]
        if len(pin_margins) != PIN_LENGTH:
            raise ValueError(f"Pin {pin['id']} margins are incomplete")
        matched = _matched_prefix(got, expected, pin_margins, margin_limit, pin["id"])
        text = response_text(body)
        if text != tokenizer.visible(got):
            raise ValueError(f"Visible text mismatch for pin {pin['id']}")
        if matched != PIN_LENGTH:
            tolerated.append({"id": pin["id"], "token": matched, "margin": pin_margins[matched]})
    return {
        "probes_passed": len(PROBES),
        "pins_passed": len(pins),
        "repeat_passes": 2,
        "tolerated_pin_stops": tolerated,
    }


def compare_logits(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    margin_limit: float,
    max_logit_abs: float,
    relative_l2_max: float,
    min_cosine: float,
    expected_shape: tuple[int, int] | None = None,
) -> dict:
    """Whole-vocabulary metrics. Bounds are inclusive; a top-1 miss needs margin < limit."""
    if reference.shape != candidate.shape or reference.ndim != 2 or reference.shape[0] == 0:
        raise ValueError(
            f"Logit shape mismatch: reference {getattr(reference, 'shape', None)} candidate {getattr(candidate, 'shape', None)}"
        )
    if expected_shape is not None and tuple(reference.shape) != tuple(expected_shape):
        raise ValueError(f"Logit shape mismatch: {reference.shape}")
    if not np.issubdtype(reference.dtype, np.floating) or not np.issubdtype(candidate.dtype, np.floating):
        raise ValueError("Logit dtype mismatch")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("Logit trace contains a non-finite value")
    actual = np.asarray(reference, dtype=np.float64)
    other = np.asarray(candidate, dtype=np.float64)
    diff = actual - other
    max_abs = float(np.max(np.abs(diff)))
    reference_norm = np.linalg.norm(actual, axis=-1)
    relative = np.linalg.norm(diff, axis=-1) / np.maximum(reference_norm, 1e-30)
    cosine = np.sum(actual * other, axis=-1) / np.maximum(
        reference_norm * np.linalg.norm(other, axis=-1), 1e-30
    )
    top1 = np.argmax(actual, axis=-1)
    runners = actual.copy()
    runners[np.arange(len(actual)), top1] = -np.inf
    top2 = np.argmax(runners, axis=-1)
    margin = actual[np.arange(len(actual)), top1] - actual[np.arange(len(actual)), top2]
    mismatch = top1 != np.argmax(other, axis=-1)
    positions = np.flatnonzero(mismatch)
    record = {
        "rows": int(reference.shape[0]),
        "max_logit_abs": max_abs,
        "relative_l2_max": float(relative.max()),
        "cosine_min": float(cosine.min()),
        "top1_mismatches": int(positions.size),
        "mismatch_margin_max": float(margin[positions].max()) if positions.size else 0.0,
        "mismatch_positions": [int(i) for i in positions],
    }
    failures = []
    if positions.size and record["mismatch_margin_max"] >= margin_limit:
        offending = int(positions[np.argmax(margin[positions])])
        failures.append(
            f"Teacher-forced top-1 mismatch at position {offending}: margin {float(margin[offending])}"
        )
    if (
        max_abs > max_logit_abs
        or record["relative_l2_max"] > relative_l2_max
        or record["cosine_min"] < min_cosine
    ):
        failures.append(
            "Teacher-forced logits failed: "
            f"max_abs {max_abs} relative_l2 {record['relative_l2_max']} cosine {record['cosine_min']}"
        )
    if failures:
        summary = {key: record[key] for key in record if key != "mismatch_positions"}
        raise ValueError("; ".join(failures) + f"; metrics {json.dumps(summary)}")
    return record


def main() -> int:
    task = Path(__file__).resolve().parents[1]
    try:
        scan_engine(Path.cwd())
        gate = load_gate(Path(__file__).resolve().with_name("interim_gate_v0.json"))
        pins = load_pins(task / "reference" / "pins.json")
        margins = load_pin_margins(task / "reference" / "pin_margins.json", gate, pins)
        path = model_path()
        failures = []
        result = {}
        try:
            with server_endpoint(Path.cwd(), path) as url:
                result = check_accuracy(
                    HttpChat(url),
                    PinnedTokenizer.load(path),
                    pins,
                    margins,
                    margin_limit=gate["thresholds"]["margin"]["value"],
                )
        except Exception as exc:
            failures.append(exc)
        from teacher_forced import capture_logits

        cache = gate_cache_dir()
        cache.mkdir(parents=True, exist_ok=True)
        candidate_path = cache / "candidate-logits.npy"
        try:
            capture_logits(
                Path.cwd(),
                path,
                gate["prefix"]["ids"],
                destination=candidate_path,
            )
            reference = load_reference_logits(cache / REFERENCE_TRACE_NAME, gate)
            candidate = np.load(candidate_path, mmap_mode="r")
            try:
                limits = gate["thresholds"]
                result["logits"] = compare_logits(
                    reference,
                    candidate,
                    margin_limit=limits["margin"]["value"],
                    max_logit_abs=limits["max_logit_abs"]["value"],
                    relative_l2_max=limits["relative_l2_max"]["value"],
                    min_cosine=limits["min_cosine"]["value"],
                    expected_shape=(TRACE_ROWS, TRACE_VOCAB),
                )
            finally:
                del candidate
        except Exception as exc:
            failures.append(exc)
        finally:
            candidate_path.unlink(missing_ok=True)
        if failures:
            summary = "; ".join(f"{type(exc).__name__}: {exc}" for exc in failures)
            print(f"Accuracy failed: RuntimeError: {summary}", file=sys.stderr)
            return failure_exit_status(failures)
        print(json.dumps(result))
        return 0
    except Exception as exc:
        print(f"Accuracy failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return failure_exit_status([exc])


if __name__ == "__main__":
    raise SystemExit(main())
