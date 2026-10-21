"""Generate independent, unquantized-KV Flint goldens from rec's plain MLX model.

Run under the shared GPU lock. Requires an external recursive-oss/rec checkout;
only rec_mlx.quantized.QATModel's native model is executed, never its fused engine.
"""

import argparse
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

REVISION = "3b93f9053b530488405ae9fca51d4900d6ce21f0"
REFERENCE = "e53663b262af4cb1ca498e596ca8b1a2d79e0800"
PROMPTS = [
    "<|user|>What is 2 + 2? Answer briefly.<|assistant|>",
    "<|user|>Write one sentence about the ocean.<|assistant|>",
    "<|user|>Complete the Python expression: sum([1, 2, 3]) equals<|assistant|>",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--rec", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tokens", type=int, default=64)
    args = parser.parse_args()
    commit = subprocess.check_output(
        ["git", "-C", str(args.rec), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != REFERENCE:
        raise ValueError(f"rec revision: expected {REFERENCE}, got {commit}")
    sys.path.insert(0, str(args.rec / "engine" / "src"))
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache, RotatingKVCache
    from rec_mlx.quantized import QATModel
    from transformers import AutoTokenizer

    mx.set_cache_limit(64 * 1024 * 1024)
    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    model = QATModel(args.model).native
    print(
        json.dumps(
            {"loaded_bytes": mx.get_active_memory(), "device": mx.device_info()}
        ),
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, fix_mistral_regex=False)
    examples = []
    for prompt in PROMPTS:
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        cache = [
            KVCache()
            if kind == "full_attention"
            else RotatingKVCache(max_size=model.args.sliding_window, keep=0)
            for kind in model.model.layer_types
        ]
        tokens, top_logits, step_top2 = [], [], []
        inputs = mx.array([ids], mx.int32)
        for step in range(args.tokens):
            logits = model(inputs, cache)[:, -1, :].astype(mx.float32)
            token = int(mx.argmax(logits, axis=-1).item())
            leaders = mx.argsort(-logits[0])[:2].tolist()
            step_top2.append(
                {"ids": leaders, "logits": logits[0, mx.array(leaders)].tolist()}
            )
            if step < 4:
                top = mx.argsort(-logits[0])[:10].tolist()
                top_logits.append(
                    {"ids": top, "logits": logits[0, mx.array(top)].tolist()}
                )
            tokens.append(token)
            if step % 8 == 0:
                print(
                    json.dumps(
                        {
                            "step": step,
                            "token": token,
                            "active_bytes": mx.get_active_memory(),
                        }
                    ),
                    flush=True,
                )
            mx.eval([entry.state for entry in cache])
            inputs = mx.array([[token]], mx.int32)
        examples.append(
            {
                "prompt": prompt,
                "prompt_ids": ids,
                "greedy_ids": tokens,
                "top_logits": top_logits,
                "step_top2": step_top2,
            }
        )
        print(
            json.dumps(
                {"prompt_tokens": len(ids), "continuation": tokenizer.decode(tokens)}
            ),
            flush=True,
        )
        del cache
        mx.clear_cache()
    result = {
        "model": "RecursiveSI/flint-int4",
        "revision": REVISION,
        "reference": {
            "repo": "https://github.com/recursive-oss/rec",
            "commit": commit,
            "path": "engine/src/rec_mlx/quantized.py:QATModel.native",
            "cache": "unquantized KVCache / RotatingKVCache; no megakernel",
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("mlx", "mlx-lm", "transformers")
            },
        },
        "examples": examples,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
