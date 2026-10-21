"""Independent sigmoid-routing and separately rounded mixture arithmetic."""

import struct

import numpy as np
import pytest

from monolith import kernels
from monolith.formats.fp import bf16_to_f32, f32_to_bf16
from monolith.ops.sigmoid_moe.metal import SOURCE
from monolith.runtime import _native as nt


def rounded(x):
    return bf16_to_f32(f32_to_bf16(x))


@pytest.mark.parametrize("experts,k", [(4, 4), (33, 3), (256, 6)])
def test_sigmoid_routing(experts, k):
    dev = nt.Device()
    rng = np.random.default_rng(42)
    logits = rounded(rng.normal(size=(5, experts)).astype(np.float32))
    logits[1] = 0  # All ties choose the lowest IDs.
    logits[2] = -1000  # Zero scores still produce a defined, zero-weight selection.
    scores = np.empty_like(logits, dtype=np.float64)
    positive = logits >= 0
    scores[positive] = 1 / (1 + np.exp(-logits[positive].astype(np.float64)))
    exp_negative = np.exp(logits[~positive].astype(np.float64))
    scores[~positive] = exp_negative / (1 + exp_negative)
    ids = np.argsort(-scores, axis=-1, kind="stable")[:, :k]
    chosen = np.take_along_axis(scores, ids, axis=-1)
    expected = f32_to_bf16(chosen / (chosen.sum(-1, keepdims=True) + 1e-20))
    ib, wb = nt.Buffer(dev, 6 * k * 4), nt.Buffer(dev, 6 * k * 4)
    ib.fill(0xFF)
    wb.fill(0xFF)
    pipeline = nt.Pipeline(
        nt.Library(dev, kernels.PRELUDE + SOURCE, {}), "sigmoid_route"
    )
    dispatch = (
        nt.Dispatch()
        .pipeline(pipeline)
        .buffer(0, nt.Buffer(dev, f32_to_bf16(logits).tobytes()))
        .buffer(1, ib)
        .buffer(2, wb)
        .bytes(3, struct.pack("<4I", experts, k, 5, 0))
        .grid(6)
        .threadgroup(32)
    )
    result = nt.Queue(dev).run([dispatch])
    assert not result.error, result.error
    actual_ids = np.frombuffer(ib.read(0, 6 * k * 4), np.int32).reshape(6, k)
    actual_weights = np.frombuffer(wb.read(0, 5 * k * 4), np.float32).reshape(5, k)
    assert np.array_equal(actual_ids[:5], ids)
    assert np.array_equal(f32_to_bf16(actual_weights), expected)
    assert np.all(actual_ids[5] == -1)


@pytest.mark.parametrize("k", [1, 6, 16])
def test_separate_mixture_rounding(k):
    dev = nt.Device()
    rng = np.random.default_rng(43)
    t, h = 3, 256
    experts = rounded(rng.normal(size=(t, k, h)).astype(np.float32))
    weights = rounded(rng.uniform(size=(t, k)).astype(np.float32))
    shared, residual = (
        rounded(rng.normal(size=(t, h)).astype(np.float32)) for _ in range(2)
    )
    products = rounded(experts * weights[..., None])
    expected = f32_to_bf16(rounded(rounded(products.sum(axis=1)) + shared) + residual)
    arrays = [f32_to_bf16(experts), weights, f32_to_bf16(shared), f32_to_bf16(residual)]
    buffers = [nt.Buffer(dev, array.tobytes()) for array in arrays]
    out = nt.Buffer(dev, (t + 1) * h * 2)
    out.fill(0xFF)
    pipeline = nt.Pipeline(nt.Library(dev, kernels.PRELUDE + SOURCE, {}), "moe_sum")
    dispatch = nt.Dispatch().pipeline(pipeline)
    for i, buffer in enumerate(buffers):
        dispatch.buffer(i, buffer)
    result = nt.Queue(dev).run(
        [
            dispatch.buffer(4, out)
            .bytes(5, struct.pack("<4I", 0, k, t, h))
            .grid(t + 1)
            .threadgroup(32)
        ]
    )
    assert not result.error, result.error
    actual = np.frombuffer(out.read(0, (t + 1) * h * 2), np.uint16).reshape(t + 1, h)
    assert np.array_equal(actual[:t], expected)
    assert np.all(actual[t] == 0xFFFF)


@pytest.mark.parametrize("format_name", ["bf16", "int4_affine"])
def test_expert_projection_preserves_separate_silu_rounding(format_name):
    from monolith.bench import bf16_ulp_diff, pack_spec
    from monolith.formats import FORMATS, PackLayout
    from monolith.packs.transforms import interleave_chunks

    dev = nt.Device()
    rng = np.random.default_rng(44)
    e, rows, k, top_k = 4, 64, 256, 2
    fmt = FORMATS.get(format_name)
    weight = rng.normal(0, 0.1, (e * rows * 2, k)).astype(np.float32)
    spec = fmt.quantize(weight)
    decoded = fmt.dequantize(spec)
    order = np.concatenate(
        [np.asarray(interleave_chunks(rows, rows, 8)) + i * rows * 2 for i in range(e)]
    )
    permuted = type(spec)(
        spec.format,
        spec.shape,
        {name: value[order] for name, value in spec.tensors.items()},
        spec.params,
    )
    data, info, scales = pack_spec(permuted, PackLayout())
    x = rounded(rng.normal(size=(1, k)).astype(np.float32))
    ids = np.array([[0, 3]], np.int32)
    expected = []
    for expert in ids[0]:
        acc = decoded[expert * 2 * rows : (expert + 1) * 2 * rows].astype(
            np.float64
        ) @ x[0].astype(np.float64)
        gate, up = np.split(rounded(acc), 2)
        expected.append(rounded(rounded(gate / (1 + np.exp(-gate))) * up))
    macros = kernels.gemv_macros(
        info,
        t=1,
        epilogue="silu_mul",
        out_bf16=True,
        pairs=(top_k, 2 * rows // 16, False),
    )
    macros["SILU_ROUND"] = "1"
    pipeline = nt.Pipeline(
        nt.Library(dev, kernels.gemv_source(format_name), macros), "gemv_T"
    )
    out = nt.Buffer(dev, top_k * rows * 2)
    n_sg = 12 * dev.info().gpu_cores
    dispatch = (
        nt.Dispatch()
        .pipeline(pipeline)
        .buffer(0, nt.Buffer(dev, data))
        .buffer(1, nt.Buffer(dev, scales.tobytes()))
        .buffer(2, nt.Buffer(dev, f32_to_bf16(x).tobytes()))
        .buffer(3, out)
        .bytes(4, kernels.gemv_params(2 * rows, 2 * rows // 16, n_sg, 1))
        .buffer(9, nt.Buffer(dev, ids.tobytes()))
        .grid(dev.info().gpu_cores)
        .threadgroup(384)
    )
    result = nt.Queue(dev).run([dispatch])
    assert not result.error, result.error
    actual = bf16_to_f32(
        np.frombuffer(out.read(0, top_k * rows * 2), np.uint16)
    ).reshape(top_k, rows)
    assert bf16_ulp_diff(actual, np.array(expected)).max() <= 1
