"""Affine INT8 preserves the checkpoint grid, distinct from signed INT8."""

import numpy as np
import pytest

from monolith.formats import FORMATS, PackLayout
from monolith.formats.fp import f32_to_bf16


@pytest.mark.parametrize("k", [256, 512, 2048, 3584])
@pytest.mark.parametrize("dtype", ["bf16", "f16"])
@pytest.mark.parametrize("placement", ["inline", "block"])
def test_affine_int8_checkpoint_pack_preserves_all_values(k, dtype, placement):
    rng = np.random.default_rng(19)
    n = 16
    codes = rng.integers(0, 256, (n, k), dtype=np.uint8)
    scales = rng.uniform(-0.2, 0.2, (n, k // 64)).astype(np.float32)
    biases = rng.uniform(-1, 1, scales.shape).astype(np.float32)
    encode = f32_to_bf16 if dtype == "bf16" else lambda x: x.astype(np.float16)
    raw_scales, raw_biases = encode(scales), encode(biases)
    fmt = FORMATS.get("int8_affine")
    spec = fmt.unpack(
        {"weight": codes.view(np.uint32), "scales": raw_scales, "biases": raw_biases},
        shape=(n, k),
    )
    expected = (
        codes.reshape(n, k // 64, 64).astype(np.float32)
        * spec.tensors["scales"][..., None]
        + spec.tensors["biases"][..., None]
    ).reshape(n, k)
    assert np.array_equal(fmt.dequantize(spec), expected)
    data, info = fmt.pack(spec, PackLayout(scale_placement=placement))
    assert info.format == "int8_affine"
    back = fmt.unpack_pack(data, info)
    assert back.format == "int8_affine"
    for key in spec.tensors:
        assert np.array_equal(back.tensors[key], spec.tensors[key]), key
    assert np.array_equal(fmt.dequantize(back), expected)


def test_affine_int8_quantizer_uses_unsigned_byte_range():
    fmt = FORMATS.get("int8_affine")
    w = np.linspace(-2, 3, 2048, dtype=np.float32).reshape(2, 1024)
    spec = fmt.quantize(w)
    assert spec.tensors["weight"].shape == w.shape
    assert np.max(np.abs(fmt.dequantize(spec) - w)) < 0.04
