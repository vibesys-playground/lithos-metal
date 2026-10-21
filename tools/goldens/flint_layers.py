"""Qualify the first dense and routed layers against the plain rec model.

Both modes must run under the GPU lock. Reference mode needs an external rec
checkout; engine mode reads only the saved arrays and the native packed weights.
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["reference", "engine"])
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--pack", type=Path)
    parser.add_argument("--rec", type=Path)
    parser.add_argument("--arrays", required=True, type=Path)
    parser.add_argument(
        "--fixture", type=Path, default=Path("tests/models/flint/goldens/int4.json")
    )
    args = parser.parse_args()
    ids = json.loads(args.fixture.read_text())["examples"][0]["prompt_ids"]
    if args.mode == "reference":
        sys.path.insert(0, str(args.rec / "engine/src"))
        import mlx.core as mx
        from mlx_lm.models.base import create_attention_mask
        from mlx_lm.models.cache import KVCache, RotatingKVCache
        from rec_mlx.quantized import QATModel

        mx.set_cache_limit(64 * 1024 * 1024)
        mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
        model = QATModel(args.model).native
        h = model.model.embed_tokens(mx.array([ids], mx.int32))
        mx.eval(h)
        arrays = {"embed_tokens.h": np.array(h[0].astype(mx.float32))}
        for i, layer in enumerate(model.layers[:3]):
            cache = (
                KVCache()
                if model.model.layer_types[i] == "full_attention"
                else RotatingKVCache(max_size=model.args.sliding_window, keep=0)
            )
            mask = create_attention_mask(
                h,
                cache,
                window_size=None
                if model.model.layer_types[i] == "full_attention"
                else model.args.sliding_window,
            )
            h = h + layer.self_attn(layer.input_layernorm(h), mask, cache)
            mx.eval(h)
            arrays[f"layers.{i}.self_attn.h"] = np.array(h[0].astype(mx.float32))
            h = h + layer.mlp(layer.post_attention_layernorm(h))
            mx.eval(h)
            arrays[f"layers.{i}.mlp.h"] = np.array(h[0].astype(mx.float32))
        np.savez(args.arrays, **arrays)
        print(json.dumps({"reference_arrays": list(arrays)}), flush=True)
        return
    from monolith.formats.fp import bf16_to_f32
    from monolith.generate import Session
    from monolith.models.flint import FlintConfig, FlintModel

    model = FlintModel(
        FlintConfig.from_pretrained(args.model), max_context=9216, num_layers_override=3
    )
    session = Session(
        model,
        str(args.pack),
        eos=-1,
        autotune=False,
        commute_norm=False,
        accelerator="off",
        prefill_optimizations=False,
        prefill_chunk_size=32,
    )
    session.generate(ids, 1, steps_per_cb=1, in_flight=1)
    with np.load(args.arrays) as arrays:
        for name in arrays:
            expected = arrays[name]
            actual = bf16_to_f32(np.frombuffer(session.read(name), np.uint16))[
                : expected.size
            ].reshape(expected.shape)
            a, b = actual.astype(np.float64), expected.astype(np.float64)
            cosine = float(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b)))
            error, scale = float(np.max(np.abs(a - b))), float(np.max(np.abs(b)))
            record = {
                "value": name,
                "cosine": cosine,
                "max_abs": error,
                "scale_over_64": scale / 64,
            }
            print(json.dumps(record), flush=True)
            if cosine <= 0.999 or error > scale / 64:
                raise RuntimeError(f"layer qualification failed: {name}")
    session.release_engines()
    del session
    gc.collect()


if __name__ == "__main__":
    main()
