"""Replay each real native decoder block on identical saved reference inputs.

No error propagates between blocks. Every block consumes the reference's own
previous residual and a fresh cache, so this diagnostic requires a single short
prefill chunk. It executes the actual layer lowering, compiler and Metal ops.
Run under lockf. Compare with flint_trace's isolated FP32 reference capture.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from monolith.compiler import emit_program
from monolith.compiler.passes import DEFAULT_PASSES
from monolith.core.dtypes import DType
from monolith.core.ir import Graph
from monolith.core.shapes import T
from monolith.formats.fp import bf16_to_f32, f32_to_bf16
from monolith.generate import Session
from monolith.models.flint import FlintModel
from monolith.nn import LowerContext, state_shape
from monolith.runtime import Engine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    completed = json.loads((args.reference / "complete.json").read_text())
    rows = completed["rows"]
    if rows >= 128:
        raise ValueError(
            "local replay requires fewer than 128 rows in a single prefill"
        )
    model = FlintModel.from_checkpoint(args.model, max_context=128)
    session = Session(
        model,
        str(args.pack),
        eos=-1,
        autotune=False,
        commute_norm=False,
        accelerator="off",
        prefill_chunk_size=128,
        prefill_optimizations=False,
    )
    shared = None
    pipelines = {}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "inputs.json").write_text((args.reference / "inputs.json").read_text())
    names = []
    try:
        for i, block in enumerate(model.blocks):
            graph = Graph("isolated_decoder_block")
            hidden = graph.input("hidden", (T, model.config.hidden_size), DType.BF16)
            context = LowerContext(t=T)
            for entry in block.mixer.state_entries():
                context.states[entry.name] = graph.state(
                    entry.name, state_shape(entry), entry.dtype
                )
            for name, (dtype, array) in model.tables().items():
                context.consts[name] = graph.const(
                    name, array.shape, DType.parse(dtype.lower())
                )
            block.lower(graph, hidden, context)
            graph.check()
            for transform in DEFAULT_PASSES:
                transform(graph)
            program = emit_program(
                graph,
                pack=session.pack,
                profile=session.profile,
                t=128,
                dynamic_t=True,
                layout=session.layout,
                tail=None,
                commute_norm=False,
                accelerator="off",
            )
            engine = Engine(
                program, session.dev, buffers=shared, pipeline_cache=pipelines
            )
            shared = dict(engine.buffers)
            source = "embed_tokens.h" if i == 0 else f"layers.{i - 1}.mlp.h"
            value = np.load(args.reference / f"{source}.npy")
            engine.buffers["hidden"].write(f32_to_bf16(value).tobytes(), 0)
            state = engine.state()
            state.update(position=0, t_this_step=rows, done=0)
            engine.buffers[program.step_state].write(session.layout.pack(state), 0)
            result = engine.run(1, steps_per_cb=1, in_flight=1)
            selected = [f"layers.{i}.self_attn.h", f"layers.{i}.mlp.h"]
            if i not in model.config.mlp_only_layers:
                selected += [f"layers.{i}.mlp.ids", f"layers.{i}.mlp.router.gate.y"]
            if i < 2:
                selected += [
                    f"layers.{i}.self_attn.qkv.q_proj+k_proj+v_proj.y",
                    f"layers.{i}.mlp.gate_up.gate_proj+up_proj.y",
                ]
            for name in selected:
                is_ids = name.endswith(".ids")
                width = program.buffers[name].nbytes // (128 * (4 if is_ids else 2))
                raw = np.frombuffer(
                    engine.read(name, rows * width * (4 if is_ids else 2)),
                    np.int32 if is_ids else np.uint16,
                )
                actual = raw if is_ids else bf16_to_f32(raw)
                np.save(args.out / f"{name}.npy", actual.reshape(rows, width))
                names.append(name)
            print(json.dumps({"layer": i, "gpu_ms": result.gpu_ms}), flush=True)
            del engine
    finally:
        session.release_engines()
    (args.out / "complete.json").write_text(
        json.dumps(
            {
                **completed,
                "mode": "native_isolated",
                "reference_input_sha256": completed["input_sha256"],
                "tensor_rows": {n: rows for n in names},
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
