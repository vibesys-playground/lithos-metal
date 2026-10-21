"""A static decoder appends after prefill and never rewrites the committed prefix."""

import numpy as np

from monolith.formats import PackLayout
from monolith.generate import Session
from monolith.models.qwen3_moe import Qwen3MoeModel
from monolith.nn import CausalAttention
from monolith.nn.pack_plan import pack_model
from tests.moe_synth import write_checkpoint


def test_static_decode_preserves_prefill_cache_and_appends(tmp_path):
    write_checkpoint(tmp_path)
    model = Qwen3MoeModel.from_checkpoint(tmp_path, max_context=16)
    model.blocks[0].mixer = CausalAttention(
        256,
        4,
        2,
        64,
        10000.0,
        1e-6,
        hf_prefix="model.layers.0.self_attn.",
        prefix="layers.0.self_attn.",
        max_context=16,
        rotary=False,
        sinks=False,
    )
    pack_model(model, str(tmp_path), str(tmp_path / "pack"), PackLayout())
    session = Session(
        model,
        str(tmp_path / "pack"),
        eos=-1,
        autotune=False,
        commute_norm=False,
        accelerator="off",
        prefill_optimizations=False,
        prefill_chunk_size=4,
    )
    try:
        ids = [1, 2, 3]
        session.generate(ids, 1, steps_per_cb=1, in_flight=1)
        prefix = {
            entry.name: session.read(entry.name)[: 3 * 2 * 64 * 2]
            for entry in model.state_spec().entries
        }
        result = session.generate(ids, 4, steps_per_cb=1, in_flight=1)
        assert len(result.tokens) == 4
        for name, expected in prefix.items():
            cache = session.read(name)
            assert cache[: len(expected)] == expected, name
            new_rows = np.frombuffer(cache, np.uint16).reshape(16, 2, 64)[3:6]
            assert all(np.any(row != 0) for row in new_rows), name
    finally:
        session.release_engines()
