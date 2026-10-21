"""Parameterized attention against independent NumPy FP64 arithmetic."""

import struct

import numpy as np
import pytest

from monolith import kernels
from monolith.bench import bf16_ulp_diff
from monolith.formats.fp import bf16_to_f32, f32_to_bf16
from monolith.ops.causal_attention.metal import SOURCE
from monolith.runtime import _native as nt


@pytest.mark.parametrize("rotary", [False, True])
@pytest.mark.parametrize("window", [0, 1, 3, 513])
@pytest.mark.parametrize("sinks", [False, True])
@pytest.mark.parametrize("position,t", [(0, 3), (7, 3), (514, 1)])
def test_causal_attention(rotary, window, sinks, position, t):
    dev = nt.Device()
    rng = np.random.default_rng(41)
    heads, kv, d, capacity, eps = 4, 2, 128, position + t + 2, 1e-6

    def rounded(x):
        return bf16_to_f32(f32_to_bf16(x))

    def array(shape):
        return rounded(rng.normal(size=shape).astype(np.float32))

    proj = array((t, (heads + 2 * kv) * d))
    kc, vc = array((capacity, kv, d)), array((capacity, kv, d))
    qn, kn = array((d,)), array((d,))
    angles = (
        np.arange(capacity, dtype=np.float32)[:, None]
        * (10000.0 ** (-np.arange(0, d, 2, dtype=np.float32) / d))[None]
    )
    angles = np.concatenate((angles, angles), axis=-1)
    cos, sin = np.cos(angles), np.sin(angles)
    offsets = np.array([-3.0, 0.0, 4.0, 12.0], np.float32)

    def prepare(x, w):
        y = rounded(
            x.astype(np.float64)
            / np.sqrt(np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True) + eps)
            * w
        )
        if rotary:
            a, b = np.split(y.astype(np.float64), 2, axis=-1)
            partner = np.concatenate((-b, a), axis=-1)
            y = rounded(
                y * cos[position : position + t, None]
                + partner * sin[position : position + t, None]
            )
        return y

    q = prepare(proj[:, : heads * d].reshape(t, heads, d), qn)
    k = prepare(proj[:, heads * d : (heads + kv) * d].reshape(t, kv, d), kn)
    v = proj[:, (heads + kv) * d :].reshape(t, kv, d)
    refk, refv = kc.copy(), vc.copy()
    refk[position : position + t], refv[position : position + t] = k, v
    expected = np.empty((t, heads, d), np.float64)
    for row in range(t):
        end = position + row + 1
        begin = max(0, end - window) if window else 0
        for h in range(heads):
            j = h // (heads // kv)
            score = (
                refk[begin:end, j].astype(np.float64)
                @ q[row, h].astype(np.float64)
                / np.sqrt(d)
            )
            maximum = max(score.max(), offsets[h]) if sinks else score.max()
            probs = np.exp(score - maximum)
            den = probs.sum() + (np.exp(offsets[h] - maximum) if sinks else 0)
            expected[row, h] = probs @ refv[begin:end, j].astype(np.float64) / den
    macros = {
        "D": str(d),
        "ROTARY": str(int(rotary)),
        "WINDOW": str(window),
        "SINKS": str(int(sinks)),
    }
    pipeline = nt.Pipeline(
        nt.Library(dev, kernels.PRELUDE + SOURCE, macros), "causal_attention"
    )
    arrays = [
        f32_to_bf16(proj),
        f32_to_bf16(kc),
        f32_to_bf16(vc),
        qn,
        kn,
        cos,
        sin,
        offsets,
    ]
    buffers = [nt.Buffer(dev, a.tobytes()) for a in arrays]
    out = nt.Buffer(dev, (t + 1) * heads * d * 2)
    out.fill(0xFF)
    dispatch = nt.Dispatch().pipeline(pipeline)
    for i, buffer in enumerate(buffers):
        dispatch.buffer(i, buffer)
    dispatch.buffer(8, out).bytes(
        9, struct.pack("<5If", heads, kv, capacity, t, position, eps)
    )
    result = nt.Queue(dev).run([dispatch.grid((t + 1) * heads).threadgroup(32)])
    assert not result.error, result.error
    got = bf16_to_f32(np.frombuffer(out.read(0, t * heads * d * 2), np.uint16)).reshape(
        t, heads, d
    )
    assert bf16_ulp_diff(got, expected).max() <= 2
    for buffer, ref in zip(buffers[1:3], (refk, refv), strict=True):
        actual = bf16_to_f32(
            np.frombuffer(buffer.read(0, capacity * kv * d * 2), np.uint16)
        ).reshape(capacity, kv, d)
        assert bf16_ulp_diff(actual, ref).max() <= 1
        assert np.array_equal(actual[:position], ref[:position])
        assert np.array_equal(actual[position + t :], ref[position + t :])
    assert np.all(
        np.frombuffer(out.read(t * heads * d * 2, heads * d * 2), np.uint16) == 0xFFFF
    )
