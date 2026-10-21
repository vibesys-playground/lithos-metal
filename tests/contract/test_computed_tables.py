"""Capacity growth reuses immutable packed weights and checks every old table bit."""

import json

import numpy as np
import pytest

from monolith.formats.safetensors_reader import write_safetensors
from monolith.packs import Packer, PackFile, TableRequest, extended_tables


@pytest.mark.parametrize("old,new,width", [(1, 2, 1), (3, 17, 8), (8, 32, 64)])
def test_extension_preserves_every_packed_value_without_writing(
    tmp_path, old, new, width
):
    values = np.arange(new * width, dtype=np.float32).reshape(new, width)
    write_safetensors(tmp_path / "empty.safetensors", {})
    writer = Packer(tmp_path, tmp_path)
    writer.add_table(TableRequest("positions", values[:old], "F32"))
    writer.write()
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    pack = PackFile(tmp_path)
    data = extended_tables(pack, {"positions": ("F32", values)})
    assert data == {"positions": values.tobytes()}
    assert extended_tables(pack, {"positions": ("F32", values[:old])}) == {}
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    for row in range(old):
        broken = values.copy()
        broken[row, 0] += 1
        with pytest.raises(ValueError, match="packed prefix"):
            extended_tables(pack, {"positions": ("F32", broken)})


@pytest.mark.parametrize(
    "values,dtype",
    [(np.zeros((10, 3), np.float32), "F32"), (np.zeros((10, 2), np.float64), "F64")],
)
def test_shape_and_dtype_changes_fail(tmp_path, values, dtype):
    write_safetensors(tmp_path / "empty.safetensors", {})
    writer = Packer(tmp_path, tmp_path)
    writer.add_table(TableRequest("positions", np.zeros((2, 2), np.float32), "F32"))
    writer.write()
    with pytest.raises(ValueError, match="only extend F32"):
        extended_tables(PackFile(tmp_path), {"positions": (dtype, values)})


@pytest.mark.parametrize("capacity", [32, 32768])
def test_compile_large_context_from_small_pack(tmp_path, capacity):
    # Public model/checkpoint fixture; no model weights or Metal are loaded.
    from monolith.backends.metal import load_configs
    from monolith.compiler import compile_program
    from monolith.formats import PackLayout
    from monolith.models.flint import FlintModel
    from monolith.nn.pack_plan import pack_model
    from monolith.serving.cache import ensure_pack
    from tests.contract.test_flint_package import write_checkpoint

    write_checkpoint(tmp_path)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    config["max_position_embeddings"] = capacity
    config_path.write_text(json.dumps(config))
    small = FlintModel.from_checkpoint(tmp_path, max_context=8)
    root = tmp_path / "pack"
    pack_model(
        small,
        str(tmp_path),
        str(root),
        PackLayout(),
        extra={"options": {"max_context": 8}},
    )
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    large = FlintModel.from_checkpoint(tmp_path, max_context=capacity)
    assert (
        ensure_pack(
            tmp_path,
            large,
            root,
            capacity=capacity,
            layout=PackLayout(),
            backend="m3_pro",
            read_only=True,
        )
        == root
    )
    profile = load_configs()["apple-m3-pro-18c"]
    for dynamic in (False, True):
        program = compile_program(
            large, PackFile(root), profile, t=1, dynamic_t=dynamic, commute_norm=False
        )
        assert program.context_capacity == capacity
        for name, (_, values) in large.tables().items():
            spec = program.buffers["computed." + name]
            assert spec.init == values.tobytes() and spec.file is None
        rotary = [
            op
            for op in program.ops
            if op.name == "causal_attention"
            and program.kernels[op.kernel].macros["ROTARY"] == "1"
        ]
        assert rotary
        for op in rotary:
            assert (5, "computed.causal_rope_cos", 0) in op.bindings
            assert (6, "computed.causal_rope_sin", 0) in op.bindings
    assert before == {p.name: p.read_bytes() for p in root.iterdir()}


@pytest.mark.parametrize("length", [2, 4, 8])
def test_changed_f32_prefix_is_rejected_at_smaller_equal_and_larger_capacity(
    tmp_path, length
):
    write_safetensors(tmp_path / "empty.safetensors", {})
    writer = Packer(tmp_path, tmp_path)
    writer.add_table(TableRequest("positions", np.zeros((4, 2), np.float32), "F32"))
    writer.write()
    changed = np.zeros((length, 2), np.float32)
    changed[0, 1] = 1
    with pytest.raises(ValueError, match="packed prefix"):
        extended_tables(PackFile(tmp_path), {"positions": ("F32", changed)})
