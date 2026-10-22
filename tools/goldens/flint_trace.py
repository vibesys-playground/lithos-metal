"""Teacher-forced Flint traces, independent rec arithmetic and native execution.

Run capture commands under lockf. Inputs always use golden continuations, never
sampled predictions. Chunk size changes MLX's legitimate GEMV/GEMM and attention
reduction ordering without changing the checkpoint or BF16 layer boundaries.
"""

import argparse
import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile
from contextlib import nullcontext
from pathlib import Path

import numpy as np


def inputs(args):
    fixture = json.loads(args.fixture.read_text())
    example = fixture["examples"][args.example]
    ids = example["prompt_ids"] + example["greedy_ids"][:-1]
    if args.length:
        ids = (ids * ((args.length + len(ids) - 1) // len(ids)))[: args.length]
    return ids


def reference(args, ids, save):
    expected = json.loads(args.fixture.read_text())["reference"]["commit"]
    actual = subprocess.check_output(
        ["git", "-C", str(args.rec), "rev-parse", "HEAD"], text=True, timeout=10
    ).strip()
    if actual != expected:
        raise ValueError(f"reference revision: expected {expected}, got {actual}")
    sys.path.insert(0, str(args.rec / "engine/src"))
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask
    from mlx_lm.models.cache import KVCache, RotatingKVCache
    from rec_mlx.quantized import QATModel

    mx.set_cache_limit(64 * 1024 * 1024)
    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    model = QATModel(args.model).native
    if args.precision == "fp32":
        # Integer codes remain unchanged; BF16 scale/bias/norm/router values
        # widen exactly, and activations/accumulation use FP32 throughout.
        model.set_dtype(mx.float32)
        mx.eval(model.parameters())
    caches = [
        KVCache()
        if kind == "full_attention"
        else RotatingKVCache(max_size=model.args.sliding_window, keep=0)
        for kind in model.model.layer_types
    ]
    if args.greedy:
        example = json.loads(args.fixture.read_text())["examples"][args.example]
        prompt = example["prompt_ids"]
        for start in range(0, len(prompt), args.chunk):
            logits = model(mx.array([prompt[start : start + args.chunk]]), caches)[
                :, -1, :
            ]
            mx.eval(logits, [cache.state for cache in caches])
        tokens = []
        for step in range(len(example["greedy_ids"])):
            token = int(mx.argmax(logits[0]).item())
            tokens.append(token)
            logits = model(mx.array([[token]]), caches)[:, -1, :]
            mx.eval(logits, [cache.state for cache in caches])
        mismatches = [
            i
            for i, (a, b) in enumerate(zip(tokens, example["greedy_ids"], strict=True))
            if a != b
        ]
        result = {
            "tokens": tokens,
            "first_divergence": mismatches[0] if mismatches else None,
            "mismatch_count": len(mismatches),
            "total": len(tokens),
        }
        (args.arrays / "greedy.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        return
    boundaries = list(range(0, len(ids), args.chunk))
    if args.golden_schedule:
        prompt_size = len(
            json.loads(args.fixture.read_text())["examples"][args.example]["prompt_ids"]
        )
        boundaries = [0, *range(prompt_size, len(ids))]
    for j, start in enumerate(boundaries):
        end = boundaries[j + 1] if j + 1 < len(boundaries) else len(ids)
        h = model.model.embed_tokens(mx.array([ids[start:end]]))

        pending = []

        def emit(name, value, start=start, pending=pending):
            if args.batch_eval or len(ids) > 4096:
                pending.append((name, value[0].astype(mx.float32)))
            else:
                mx.eval(value)
                save(start, name, np.array(value[0].astype(mx.float32)))

        emit("embed_tokens.h", h)
        for i, (layer, cache) in enumerate(zip(model.layers, caches, strict=True)):
            if args.isolated_inputs:
                if len(boundaries) != 1:
                    raise ValueError(
                        "isolated reference requires one whole-prefix chunk"
                    )
                source = "embed_tokens.h" if i == 0 else f"layers.{i - 1}.mlp.h"
                h = mx.array(
                    load_array(args.isolated_inputs / f"{source}.npy")[None],
                    dtype=h.dtype,
                )
            mask = create_attention_mask(
                h,
                cache,
                window_size=None
                if model.model.layer_types[i] == "full_attention"
                else model.args.sliding_window,
            )
            if args.ops and i < 2:
                x = layer.input_layernorm(h)
                q, k, v = (
                    proj(x)
                    for proj in (
                        layer.self_attn.q_proj,
                        layer.self_attn.k_proj,
                        layer.self_attn.v_proj,
                    )
                )
                emit(
                    f"layers.{i}.self_attn.qkv.q_proj+k_proj+v_proj.y",
                    mx.concatenate([q, k, v], axis=-1),
                )
            h = h + layer.self_attn(layer.input_layernorm(h), mask, cache)
            emit(f"layers.{i}.self_attn.h", h)
            x = layer.post_attention_layernorm(h)
            if i not in model.args.mlp_only_layers:
                scores = mx.sigmoid(layer.mlp.gate(x).astype(mx.float32))
                choice = scores + layer.mlp.gate.expert_bias.astype(mx.float32)
                indices = mx.argpartition(-choice, kth=layer.mlp.top_k - 1, axis=-1)[
                    ..., : layer.mlp.top_k
                ]
                emit(f"layers.{i}.mlp.ids", indices)
                emit(f"layers.{i}.mlp.router.gate.y", layer.mlp.gate(x))
            if args.ops and i < 2:
                from mlx import nn

                emit(
                    f"layers.{i}.mlp.gate_up.gate_proj+up_proj.y",
                    nn.silu(layer.mlp.gate_proj(x)) * layer.mlp.up_proj(x),
                )
            h = h + layer.mlp(x)
            emit(f"layers.{i}.mlp.h", h)
        emit("logits", model.lm_head(model.model.norm(h)))
        mx.eval([value for _, value in pending], [cache.state for cache in caches])
        for name, value in pending:
            save(start, name, np.array(value))
        print(json.dumps({"position": start, "rows": h.shape[1]}), flush=True)


def native(args, ids, save):
    from monolith.formats.fp import bf16_to_f32
    from monolith.generate import Session
    from monolith.models.flint import FlintModel

    model = FlintModel.from_checkpoint(args.model, max_context=max(9216, len(ids) + 1))
    session = Session(
        model,
        str(args.pack),
        eos=-1,
        autotune=False,
        commute_norm=False,
        accelerator="off",
        prefill_optimizations=False,
        prefill_chunk_size=args.chunk,
    )
    if args.inject == "no-sinks":
        for block in model.blocks:
            block.mixer.sinks = False
    if args.greedy:
        example = json.loads(args.fixture.read_text())["examples"][args.example]
        try:
            generated = session.generate(
                example["prompt_ids"],
                len(example["greedy_ids"]),
                steps_per_cb=1,
                in_flight=1,
            )
            mismatches = [
                i
                for i, (a, b) in enumerate(
                    zip(generated.tokens, example["greedy_ids"], strict=True)
                )
                if a != b
            ]
            result = {
                "tokens": generated.tokens,
                "first_divergence": mismatches[0] if mismatches else None,
                "mismatch_count": len(mismatches),
                "total": len(generated.tokens),
            }
            (args.arrays / "greedy.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
            print(json.dumps(result), flush=True)
        finally:
            session.release_engines()
        return
    temporary = tempfile.TemporaryDirectory(prefix="flint-trace-tables-")
    try:
        if len(ids) + 1 > 9216:
            # Keep the sole immutable weight file. This tiny auxiliary-only pack
            # extends computed rotary tables; no checkpoint matrix is written.
            from monolith.compiler import compile_program
            from monolith.packs.packer import Packer, PackFile, TableRequest
            from monolith.runtime import Engine

            packer = Packer(args.model, temporary.name)
            for name, (dtype, array) in model.tables().items():
                packer.add_table(TableRequest(name, array, dtype))
            packer.write()
            base = copy.copy(session.pack)
            base.manifest = dict(
                base.manifest,
                aux=[a for a in base.manifest["aux"] if a["transform"] != "table"],
            )
            base.aux = {a["name"]: a for a in base.manifest["aux"]}
            program = compile_program(
                model,
                [base, PackFile(temporary.name)],
                session.profile,
                t=args.chunk,
                dynamic_t=True,
                layout=session.layout,
                eos=-1,
                commute_norm=False,
                accelerator="off",
                prefill=True,
            )
            engine = Engine(program, session.dev)
        else:
            engine = session.prefill_engine(rows=args.chunk)
        session.reset()
        state_buffer = engine.buffers[engine.program.step_state]
        names = ["embed_tokens.h", "logits"]
        for i in range(model.n_layers):
            names += [f"layers.{i}.self_attn.h", f"layers.{i}.mlp.h"]
            if i not in model.config.mlp_only_layers:
                names += [f"layers.{i}.mlp.ids", f"layers.{i}.mlp.router.gate.y"]
        if args.ops:
            for i in range(2):
                names += [
                    f"layers.{i}.self_attn.qkv.q_proj+k_proj+v_proj.y",
                    f"layers.{i}.mlp.gate_up.gate_proj+up_proj.y",
                ]
        for start in range(0, len(ids), args.chunk):
            chunk = ids[start : start + args.chunk]
            state = engine.state()
            state.update(
                t_this_step=len(chunk), pending_tokens=chunk, prefill_left=1, stop_at=1
            )
            state_buffer.write(session.layout.pack(state), 0)
            timing = engine.run(1, steps_per_cb=1, in_flight=1)
            for name in names:
                width = (
                    model.config.vocab_size
                    if name == "logits"
                    else model.config.num_experts_per_tok
                    if name.endswith(".ids")
                    else model.config.num_experts
                    if name.endswith(".gate.y")
                    else model.config.hidden_size
                )
                if "self_attn.qkv." in name:
                    width = (
                        model.config.num_attention_heads
                        + 2 * model.config.num_key_value_heads
                    ) * model.config.head_dim
                elif "mlp.gate_up." in name:
                    width = model.config.intermediate_size
                dtype = np.int32 if name.endswith(".ids") else np.uint16
                raw = np.frombuffer(
                    engine.read(name, len(chunk) * width * np.dtype(dtype).itemsize),
                    dtype,
                )
                value = raw if name.endswith(".ids") else bf16_to_f32(raw)
                save(start, name, value.reshape(len(chunk), width))
            print(
                json.dumps(
                    {
                        "position": start,
                        "rows": len(chunk),
                        "gpu_ms": timing.gpu_ms,
                        "wall_ms": timing.wall_ms,
                    }
                ),
                flush=True,
            )
    finally:
        session.release_engines()
        temporary.cleanup()


def metrics(name, a, b):
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError(f"{name}: incompatible trace shapes {a.shape} and {b.shape}")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError(f"{name}: non-finite trace value")
    if name.endswith(".ids"):
        equal = np.sort(a, axis=-1) == np.sort(b, axis=-1)
        record = {
            "name": name,
            "set_agreement": float(np.mean(np.all(equal, axis=-1))),
            "expert_overlap": float(
                np.mean(
                    [len(set(x) & set(y)) / len(x) for x, y in zip(a, b, strict=True)]
                )
            ),
        }
    else:
        aa, bb = a.astype(np.float64), b.astype(np.float64)
        diff = aa - bb
        max_abs = np.max(np.abs(diff), axis=-1)
        rel = np.linalg.norm(diff, axis=-1) / np.maximum(
            np.linalg.norm(aa, axis=-1), 1e-30
        )
        cosine = np.sum(aa * bb, axis=-1) / np.maximum(
            np.linalg.norm(aa, axis=-1) * np.linalg.norm(bb, axis=-1), 1e-30
        )
        record = {
            "name": name,
            "max_abs": float(max_abs.max()),
            "relative_l2_max": float(rel.max()),
            "cosine_min": float(cosine.min()),
            "per_position_max_abs": max_abs.tolist(),
            "per_position_relative_l2": rel.tolist(),
            "per_position_cosine": cosine.tolist(),
        }
        if name == "logits":
            first = np.argmax(aa, axis=-1)
            runners = aa.copy()
            runners[np.arange(len(aa)), first] = -np.inf
            top = np.column_stack((first, np.argmax(runners, axis=-1)))
            margin = (
                np.take_along_axis(aa, top[:, :1], -1)[:, 0]
                - np.take_along_axis(aa, top[:, 1:], -1)[:, 0]
            )
            mismatch = top[:, 0] != np.argmax(bb, axis=-1)
            record.update(
                top1_mismatches=np.flatnonzero(mismatch).tolist(),
                reference_margins=margin.tolist(),
                mismatch_margin_max=float(margin[mismatch].max())
                if mismatch.any()
                else 0,
                top1_error_max=float(
                    np.max(np.abs(diff[np.arange(len(aa)), top[:, 0]]))
                ),
                top2_gap_error_max=float(
                    np.max(
                        np.abs(
                            diff[np.arange(len(aa)), top[:, 0]]
                            - diff[np.arange(len(aa)), top[:, 1]]
                        )
                    )
                ),
            )
    return dict(rows=len(a), **record)


def load_array(path, start=None, end=None):
    from monolith.formats.fp import bf16_to_f32

    value = np.load(path, mmap_mode="r")[start:end]
    return bf16_to_f32(value) if value.dtype == np.uint16 else value


def compare(args):
    if (args.arrays / "inputs.json").read_text() != (
        args.other / "inputs.json"
    ).read_text():
        raise ValueError("comparison requires identical teacher-forced input IDs")
    records = []
    paths = sorted(args.arrays.glob("*.npy"))
    if not paths:
        raise ValueError(
            "stored comparison requires array traces; use --compare-to for compressed long traces"
        )
    for path in paths:
        a = load_array(path)
        b = load_array(args.other / path.name)
        name = path.stem
        record = metrics(name, a, b)
        records.append(record)
        print(
            json.dumps(
                {
                    k: v
                    for k, v in record.items()
                    if not k.startswith("per_position") and k != "reference_margins"
                }
            ),
            flush=True,
        )
    args.report.write_text(json.dumps(records, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["reference", "native", "compare"])
    p.add_argument("--model", type=Path)
    p.add_argument(
        "--rec", type=Path, default=Path.home() / "projs/flint-reference-rec"
    )
    p.add_argument(
        "--pack",
        type=Path,
        default=Path.home()
        / ".cache/lithos-metal/flint-int4/3b93f9053b530488405ae9fca51d4900d6ce21f0/block-r16-interleaved16-c9216",
    )
    p.add_argument(
        "--fixture", type=Path, default=Path("tests/models/flint/goldens/int4.json")
    )
    p.add_argument("--example", type=int, default=1)
    p.add_argument("--length", type=int)
    p.add_argument("--chunk", type=int, default=32)
    p.add_argument("--compact", action="store_true")
    p.add_argument("--batch-eval", action="store_true")
    p.add_argument("--ops", action="store_true")
    p.add_argument("--isolated-inputs", type=Path)
    p.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--compare-to", type=Path)
    p.add_argument("--golden-schedule", action="store_true")
    p.add_argument("--inject", choices=["no-sinks"])
    p.add_argument("--arrays", type=Path, required=True)
    p.add_argument("--other", type=Path)
    p.add_argument("--report", type=Path)
    args = p.parse_args()
    if args.mode == "compare":
        compare(args)
        return
    ids = inputs(args)
    args.arrays.mkdir(parents=True, exist_ok=True)
    (args.arrays / "inputs.json").write_text(json.dumps(ids) + "\n")
    marker = args.arrays / "complete.json"
    marker.unlink(missing_ok=True)
    input_hash = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    config_hash = hashlib.sha256((args.model / "config.json").read_bytes()).hexdigest()
    if args.compare_to:
        completed = json.loads((args.compare_to / "complete.json").read_text())
        if (
            completed["input_sha256"] != input_hash
            or completed["rows"] != len(ids)
            or completed["config_sha256"] != config_hash
        ):
            raise ValueError("reference trace has incompatible input IDs/row count")
    arrays = {}
    reference_arrays = {}
    reference_start = None
    seen_rows = {}
    compressed = args.compact and len(ids) > 4096
    staged = {}
    stage_start = None

    def flush_chunk():
        if staged:
            with zipfile.ZipFile(
                args.arrays / f"chunk-{stage_start:06d}.npz",
                "w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=1,
            ) as archive:
                for name, value in staged.items():
                    with archive.open(name + ".npy", "w") as entry:
                        np.lib.format.write_array(entry, value, allow_pickle=False)
            staged.clear()

    with (
        open(args.arrays / "comparison.jsonl", "w")
        if args.compare_to
        else nullcontext()
    ) as comparisons:

        def save(start, name, value):
            nonlocal reference_start, stage_start
            if start != seen_rows.get(name, 0):
                raise ValueError(f"{name}: missing or repeated trace rows at {start}")
            seen_rows[name] = start + len(value)
            if args.compare_to:
                from monolith.formats.fp import bf16_to_f32

                storage = completed.get("storage", "arrays")
                if storage == "compressed_chunks":
                    chunk_start = start // completed["chunk"] * completed["chunk"]
                    if reference_start != chunk_start:
                        reference_arrays.clear()
                        with np.load(
                            args.compare_to / f"chunk-{chunk_start:06d}.npz"
                        ) as archive:
                            reference_arrays.update(
                                {key: archive[key] for key in archive.files}
                            )
                        reference_start = chunk_start
                    offset = start - chunk_start
                    expected = reference_arrays[name][offset : offset + len(value)]
                    if len(expected) != len(value):
                        raise ValueError(
                            "comparison chunk straddles reference archive boundary"
                        )
                elif storage == "arrays":
                    if name not in reference_arrays:
                        reference_arrays[name] = np.load(
                            args.compare_to / f"{name}.npy", mmap_mode="r"
                        )
                    expected = reference_arrays[name][start : start + len(value)]
                else:
                    raise ValueError(f"unsupported trace storage: {storage}")
                if expected.dtype == np.uint16:
                    expected = bf16_to_f32(expected)
                record = dict(position=start, **metrics(name, expected, value))
                comparisons.write(json.dumps(record) + "\n")
                comparisons.flush()
                return
            if args.compact and not name.endswith(".ids"):
                from monolith.formats.fp import f32_to_bf16

                value = f32_to_bf16(value)
            if compressed:
                if stage_start != start:
                    flush_chunk()
                    stage_start = start
                staged[name] = value
                return
            if name not in arrays:
                arrays[name] = np.lib.format.open_memmap(
                    args.arrays / f"{name}.npy",
                    mode="w+",
                    dtype=value.dtype,
                    shape=(len(ids), *value.shape[1:]),
                )
            arrays[name][start : start + len(value)] = value
            arrays[name].flush()

        (reference if args.mode == "reference" else native)(args, ids, save)
        flush_chunk()
    if not args.greedy and (
        not seen_rows or any(rows != len(ids) for rows in seen_rows.values())
    ):
        raise RuntimeError("incomplete teacher-forced trace")
    marker.write_text(
        json.dumps(
            {
                "mode": args.mode,
                "precision": args.precision,
                "chunk": args.chunk,
                "storage": "comparisons"
                if args.compare_to
                else "greedy"
                if args.greedy
                else "compressed_chunks"
                if compressed
                else "arrays",
                "rows": len(ids),
                "input_sha256": input_hash,
                "config_sha256": config_hash,
                "batch_eval": args.batch_eval or len(ids) > 4096,
                "tensor_rows": seen_rows,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
