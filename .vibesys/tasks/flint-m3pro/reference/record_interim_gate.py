#!/usr/bin/env python3
"""Record pinned reference margins and the 1k logit trace. Operator-only.

Run with lockf -k /tmp/flint-gpu.lock. Uses rec's plain QATModel.native
forward and unquantized BF16 KV. Does not download weights or write a pack.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
from launcher import model_path

REC_COMMIT = "e53663b262af4cb1ca498e596ca8b1a2d79e0800"
PREFILL_CHUNK = 128
TRACE_CHUNK = 32
TRACE_ROWS = 1024
CACHE = Path.home() / ".cache" / "lithos-metal" / "flint-gate"
TRACE_NAME = "rec-bf16-chunk32-logits-1024.npy"


def prefix_ids(fixture: Path) -> list[int]:
    example = json.loads(fixture.read_text())["examples"][1]
    base = example["prompt_ids"] + example["greedy_ids"][:-1]
    if len(base) != 72:
        raise ValueError(f"Second fixture base length is {len(base)}, expected 72")
    return (base * ((TRACE_ROWS + len(base) - 1) // len(base)))[:TRACE_ROWS]


def caches_for(model):
    from mlx_lm.models.cache import KVCache, RotatingKVCache

    return [
        KVCache()
        if kind == "full_attention"
        else RotatingKVCache(max_size=model.args.sliding_window, keep=0)
        for kind in model.model.layer_types
    ]


def last_row(mx, logits) -> np.ndarray:
    mx.eval(logits)
    row = np.array(logits[0, -1].astype(mx.float32), dtype=np.float64)
    if row.ndim != 1:
        raise ValueError(f"Expected one logit row, got {row.shape}")
    return row


def top_margin(row: np.ndarray) -> tuple[int, float]:
    top1 = int(np.argmax(row))
    runners = row.copy()
    runners[top1] = -np.inf
    top2 = int(np.argmax(runners))
    return top1, float(row[top1] - row[top2])


def pin_records(mx, model, pins: list[dict]) -> dict:
    recorded = {}
    for pin in pins:
        cache = caches_for(model)
        ids = pin["prompt_token_ids"]
        logits = None
        for start in range(0, len(ids), PREFILL_CHUNK):
            logits = model(mx.array([ids[start : start + PREFILL_CHUNK]]), cache=cache)
        margins, top1 = [], []
        expected = pin["expected_token_ids"]
        for token in expected:
            choice, margin = top_margin(last_row(mx, logits))
            margins.append(margin)
            top1.append(choice)
            logits = model(mx.array([[token]]), cache=cache)
        recorded[pin["id"]] = {"margins": margins, "top1": top1}
        disagreed = [i for i, (got, want) in enumerate(zip(top1, expected)) if got != want]
        print(
            f"pin {pin['id']} reference disagreements {disagreed}",
            file=sys.stderr,
            flush=True,
        )
    return recorded


def write_trace(mx, model, ids: list[int], destination: Path) -> None:
    cache = caches_for(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stored = None
    vocab = None
    try:
        for start in range(0, len(ids), TRACE_CHUNK):
            piece = ids[start : start + TRACE_CHUNK]
            logits = model(mx.array([piece]), cache=cache)
            mx.eval(logits)
            rows = np.array(logits[0].astype(mx.float32), dtype=np.float32)
            if stored is None:
                vocab = int(rows.shape[1])
                stored = np.lib.format.open_memmap(
                    destination, mode="w+", dtype=np.float32, shape=(len(ids), vocab)
                )
            if rows.shape != (len(piece), vocab):
                raise ValueError(f"Unexpected logit shape {rows.shape}")
            stored[start : start + len(piece)] = rows
            stored.flush()
            print(
                f"reference rows {start + len(piece)}/{len(ids)}",
                file=sys.stderr,
                flush=True,
            )
    finally:
        if stored is not None:
            stored.flush()


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rec-repo", type=Path, required=True)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path("tests/models/flint/goldens/int4.json"),
    )
    args = parser.parse_args()
    free = shutil.disk_usage(Path.home()).free
    if free < 8 * 1024**3:
        raise ValueError(f"Home volume has {free} bytes free; refusing to write the trace")
    commit = subprocess.check_output(
        ["git", "-C", str(args.rec_repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != REC_COMMIT:
        raise ValueError(f"Expected rec commit {REC_COMMIT}, got {commit}")
    sys.path.insert(0, str(args.rec_repo / "engine" / "src"))
    import mlx.core as mx
    from rec_mlx.quantized import QATModel

    ids = prefix_ids(args.fixture)
    digest = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    if digest != "4930907e7e0ea1a8412c9649855ae8a2e7467c3a8951033e507b5c6ba4878e4b":
        raise ValueError("1k prefix does not match the port handoff trace")
    pins = json.loads((Path(__file__).with_name("pins.json")).read_text())["pins"]
    mx.set_cache_limit(64 * 1024 * 1024)
    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    model = QATModel(model_path()).native
    payload = {
        "version": 1,
        "reference_commit": commit,
        "schedule": "bf16 QATModel.native, prefill_chunk 128, teacher-forced pinned tokens, unquantized KV",
        "pins": pin_records(mx, model, pins),
    }
    margin_path = Path(__file__).with_name("pin_margins.json")
    margin_path.write_text(json.dumps(payload, indent=2) + "\n")
    trace_path = CACHE / TRACE_NAME
    write_trace(mx, model, ids, trace_path)
    if any(name.startswith("rec_mlx.kernels") for name in sys.modules):
        raise ValueError("Reference unexpectedly imported megakernel modules")
    print(
        json.dumps(
            {
                "pin_margins_sha256": sha256(margin_path),
                "reference_logits_sha256": sha256(trace_path),
                "reference_logits_file": TRACE_NAME,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
