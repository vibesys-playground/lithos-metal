#!/usr/bin/env python3
"""Operator-only fallback; prefer reusing flint-model's independent goldens.

Run with lockf -k /tmp/flint-gpu.lock. Uses rec's plain MLX QATModel.native
forward, never build_decoder or rec's M4/M5 megakernel. No downloads.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
from launcher import MODEL, REVISION, PinnedTokenizer, model_path

REC_COMMIT = "e53663b262af4cb1ca498e596ca8b1a2d79e0800"
PROMPTS = (
    "Write a Python implementation of merge sort, including a docstring and several tests.",
    "Implement a bounded LRU cache in Python. Explain eviction and add tests for edge cases.",
    "Review this Python function for bugs and propose a corrected implementation: def mean(xs): return sum(xs) / len(xs)",
    "Explain how to test a transactional key-value store. Include code examples and crash-recovery cases.",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rec-repo", type=Path, required=True)
    args = parser.parse_args()
    commit = subprocess.check_output(
        ["git", "-C", str(args.rec_repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != REC_COMMIT:
        raise ValueError(f"Expected rec commit {REC_COMMIT}, got {commit}")
    sys.path.insert(0, str(args.rec_repo / "engine" / "src"))
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache, RotatingKVCache
    from rec_mlx.quantized import QATModel

    path = model_path()
    tokenizer = PinnedTokenizer(path)
    qat = QATModel(path)
    model = qat.native
    pins = []
    for index, prompt in enumerate(PROMPTS):
        messages = [{"role": "user", "content": prompt}]
        ids = tokenizer.ids(messages)
        # Do not call Model.make_cache(): its full-attention defaults quantize
        # KV. These explicit caches store the BF16 projections without quantization.
        cache = [
            KVCache()
            if kind == "full_attention"
            else RotatingKVCache(max_size=model.args.sliding_window, keep=0)
            for kind in model.model.layer_types
        ]
        logits = None
        for start in range(0, len(ids), 128):
            logits = model(mx.array([ids[start : start + 128]]), cache=cache)
            mx.eval(logits)
        generated = []
        for _ in range(32):
            token = int(mx.argmax(logits[0, -1]).item())
            generated.append(token)
            logits = model(mx.array([[token]]), cache=cache)
            mx.eval(logits)
        if tokenizer.encode(tokenizer.decode(generated)) != generated:
            raise ValueError(f"Pin {index} cannot be checked losslessly through text API")
        pins.append(
            {
                "id": f"code-{index}",
                "messages": messages,
                "prompt_token_ids": ids,
                "expected_token_ids": generated,
            }
        )
    if any(n.startswith("rec_mlx.kernels") for n in sys.modules):
        raise ValueError("Reference unexpectedly imported megakernel modules")
    payload = {
        "version": 1,
        "model": MODEL,
        "revision": REVISION,
        "kv_format": "bf16",
        "reference": {
            "repo": "https://github.com/recursive-oss/rec",
            "commit": commit,
            "path": "engine/src/rec_mlx/quantized.py:QATModel.native",
            "mlx_version": mx.__version__,
            "prefill_chunk": 128,
        },
        "pins": pins,
    }
    destination = Path(__file__).with_name("pins.json")
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {len(pins)} independent pins to reference/pins.json")


if __name__ == "__main__":
    main()
