"""Unsigned affine INT8 gather against its independent decoded byte grid."""

import numpy as np
import pytest

from monolith import kernels
from monolith.formats import FORMATS, PackLayout
from monolith.formats.fp import bf16_to_f32, f32_to_bf16
from monolith.runtime import _native as nt


@pytest.mark.parametrize("placement", ["inline", "block"])
@pytest.mark.parametrize("k", [256, 512, 2048, 3584])
@pytest.mark.parametrize("dtype", ["bf16", "f16"])
def test_affine_byte_embedding(placement, k, dtype):
    dev = nt.Device()
    rng = np.random.default_rng(21)
    n = 32
    codes = rng.integers(0, 256, (n, k), dtype=np.uint8)
    sc = rng.uniform(-0.02, 0.02, (n, k // 64)).astype(np.float32)
    bi = rng.uniform(-1, 1, sc.shape).astype(np.float32)
    encode = f32_to_bf16 if dtype == "bf16" else lambda x: x.astype(np.float16)
    sc, bi = encode(sc), encode(bi)
    decode = bf16_to_f32 if dtype == "bf16" else lambda x: x.astype(np.float32)
    expected = f32_to_bf16(
        (
            codes.reshape(n, -1, 64).astype(np.float32) * decode(sc)[..., None]
            + decode(bi)[..., None]
        ).reshape(n, k)
    )
    fmt = FORMATS.get("int8_affine")
    spec = fmt.unpack(
        {"weight": codes.view(np.uint32), "scales": sc, "biases": bi}, shape=(n, k)
    )
    data, info = fmt.pack(spec, PackLayout(scale_placement=placement))
    if k == 256 and placement == "block":
        with pytest.raises(ValueError, match="embed: sub-word units"):
            kernels.embed_macros(info)
        return
    ids = np.array([0, 31, 7, 7], np.int32)
    out = nt.Buffer(dev, 5 * k * 2)
    out.fill(0xFF)
    pipeline = nt.Pipeline(
        nt.Library(dev, kernels.embed_source(fmt.name), kernels.embed_macros(info)),
        "embed",
    )
    dispatch = (
        nt.Dispatch()
        .pipeline(pipeline)
        .buffer(0, nt.Buffer(dev, ids.tobytes()))
        .buffer(1, nt.Buffer(dev, data))
        .buffer(2, out)
        .bytes(3, kernels.embed_params(k, len(ids), n))
        .grid(5)
        .threadgroup(32)
    )
    result = nt.Queue(dev).run([dispatch])
    assert not result.error, result.error
    actual = np.frombuffer(out.read(0, 5 * k * 2), np.uint16).reshape(5, k)
    assert np.array_equal(actual[:4], expected[ids])
    assert np.all(actual[4] == 0xFFFF)
