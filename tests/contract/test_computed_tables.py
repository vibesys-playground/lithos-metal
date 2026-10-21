"""Capacity growth reuses immutable packed weights and checks every old table bit."""

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


def test_compile_large_context_from_small_pack(tmp_path):
    # Public model/checkpoint fixture; no model weights or Metal are loaded.
    from monolith.backends.metal import load_configs
    from monolith.compiler import compile_program
    from monolith.formats import PackLayout
    from monolith.models.flint import FlintModel
    from monolith.nn.pack_plan import pack_model
    from monolith.serving.cache import ensure_pack
    from tests.contract.test_flint_package import write_checkpoint

    write_checkpoint(tmp_path)
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
    large = FlintModel.from_checkpoint(tmp_path, max_context=32)
    assert (
        ensure_pack(
            tmp_path,
            large,
            root,
            capacity=32,
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
        assert program.context_capacity == 32
        for name, (_, values) in large.tables().items():
            spec = program.buffers["computed." + name]
            assert spec.init == values.tobytes() and spec.file is None
    assert before == {p.name: p.read_bytes() for p in root.iterdir()}
