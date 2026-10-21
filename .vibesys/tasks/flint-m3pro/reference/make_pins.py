#!/usr/bin/env python3
"""Operator-only fallback; prefer reusing flint-model's independent goldens.

Run with lockf -k /tmp/flint-gpu.lock. Uses rec's plain MLX QATModel.native
forward, never build_decoder or rec's M4/M5 megakernel. No downloads.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
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


def reuse_port_goldens(
    data: dict, tokenizer: PinnedTokenizer, eos: list[int]
) -> tuple[list[dict], list[int]]:
    """Reuse independently generated, template-identical 32-token continuations.

    An early-EOS continuation cannot fill this task's fixed 32-token request;
    record its exclusion and generate a different prompt rather than padding it.
    """
    if set(data) != {"model", "revision", "reference", "examples"}:
        raise ValueError("Unknown/missing model-port golden keys")
    if data["model"] != MODEL or data["revision"] != REVISION:
        raise ValueError("Model-port golden weight identity mismatch")
    if data["reference"].get("commit") != REC_COMMIT or "unquantized" not in data["reference"].get(
        "cache", ""
    ):
        raise ValueError("Model-port goldens require pinned plain reference and unquantized KV")
    pins, excluded = [], []
    for index, example in enumerate(data["examples"]):
        required = {"prompt", "prompt_ids", "greedy_ids", "top_logits"}
        if not required <= set(example) <= required | {"step_top2"}:
            raise ValueError("Unknown/missing model-port example keys")
        prompt = example["prompt"]
        if not prompt.startswith("<|user|>") or not prompt.endswith("<|assistant|>"):
            raise ValueError("Model-port prompt is not a single-user chat template")
        messages = [{"role": "user", "content": prompt[len("<|user|>") : -len("<|assistant|>")]}]
        if tokenizer.ids(messages) != example["prompt_ids"]:
            raise ValueError("Model-port prompt IDs do not match the served chat template")
        generated = example["greedy_ids"][:32]
        if len(generated) != 32 or any(type(t) is not int or t < 0 for t in generated):
            raise ValueError("Model-port golden lacks 32 raw greedy IDs")
        if any(t in eos for t in generated):
            excluded.append(index)
            continue
        pins.append(
            {
                "id": f"port-{index}",
                "messages": messages,
                "prompt_token_ids": example["prompt_ids"],
                "expected_token_ids": generated,
            }
        )
    return pins, excluded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rec-repo", type=Path, required=True)
    parser.add_argument(
        "--port-goldens", type=Path, default=Path("tests/models/flint/goldens/int4.json")
    )
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
    tokenizer = PinnedTokenizer.load(path)
    pins, excluded, reused = [], [], None
    if args.port_goldens.is_file():
        golden_bytes = args.port_goldens.read_bytes()
        data = json.loads(golden_bytes)
        eos = json.loads((path / "config.json").read_text())["eos_token_id"]
        pins, excluded = reuse_port_goldens(data, tokenizer, eos)
        reused = {
            "sha256": hashlib.sha256(golden_bytes).hexdigest(),
            "source": "tests/models/flint/goldens/int4.json",
            "reference": data["reference"],
        }
        print(
            f"Reused {len(pins)} model-port pins; early-EOS example indexes: {excluded}", flush=True
        )
    needed = max(0, 4 - len(pins))
    mx.set_cache_limit(64 * 1024 * 1024)
    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    qat = QATModel(path) if needed else None
    model = qat.native if qat else None
    for index, prompt in enumerate(PROMPTS[:needed]):
        print(f"Generating additional code pin {index + 1}/{needed}", flush=True)
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
        if any(t in qat.eos_tokens for t in generated):
            raise ValueError(f"Pin {index} reached EOS before its fixed 32-token budget")
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
            "versions": {
                name: importlib.metadata.version(name) for name in ("mlx", "mlx-lm", "transformers")
            },
            "cache": "unquantized KVCache / RotatingKVCache; no megakernel",
            "reused_port_goldens": reused,
            "excluded_port_examples_early_eos": excluded,
            "prefill_chunk": 128,
        },
        "pins": pins,
    }
    destination = Path(__file__).with_name("pins.json")
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {len(pins)} independent pins to reference/pins.json")


if __name__ == "__main__":
    main()
