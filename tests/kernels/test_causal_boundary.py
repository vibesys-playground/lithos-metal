"""Adversarial long-history boundary/sink checks and semantic mutation sensitivity."""

import struct

import numpy as np
import pytest

from monolith import kernels
from monolith.bench import bf16_ulp_diff
from monolith.formats.fp import bf16_to_f32, f32_to_bf16

# test-isolation: direct dispatch checks the kernel buffer ABI and deliberately
# incorrect compile-time bindings against the same independent numerical oracle.
from monolith.ops.causal_attention.metal import SOURCE
from monolith.runtime import _native as nt


@pytest.mark.parametrize("position", [513, 1026, 4096, 16384])
@pytest.mark.parametrize("rotary,window", [(True, 513), (False, 0)])
def test_oldest_visible_key_and_sink_survive_long_history(position, rotary, window):
    """A decisive oldest key makes a one-token window error observable above noise.

    Full attention places that key at absolute position zero and disables RoPE.
    Sliding attention places it exactly at p-512 and applies RoPE. A sink with
    twice that key's exponential mass must reduce its contribution to one third.
    """
    dev = nt.Device()
    d, heads, kv, eps = 128, 2, 1, 1e-6
    capacity = position + 1
    angles = (
        np.arange(capacity, dtype=np.float32)[:, None]
        * (10000.0 ** (-np.arange(0, d, 2, dtype=np.float32) / d))[None]
    )
    angles = np.concatenate([angles, angles], axis=-1)
    cos, sin = np.cos(angles), np.sin(angles)
    proj = np.zeros((1, (heads + 2 * kv) * d), np.float32)
    query = np.zeros(d, np.float32)
    query[0] = 1 / np.sqrt(1 / d + eps)
    query = bf16_to_f32(f32_to_bf16(query))
    if rotary:
        partner = np.concatenate([-query[d // 2 :], query[: d // 2]])
        query = bf16_to_f32(
            f32_to_bf16(query * cos[position] + partner * sin[position])
        )
    for head in range(heads):
        proj[0, head * d] = 1
    kc = np.zeros((capacity, kv, d), np.float32)
    vc = np.zeros_like(kc)
    oldest = max(0, position + 1 - window) if window else 0
    kc[oldest, 0], vc[oldest, 0] = query, 1
    score = np.dot(query.astype(np.float64), query.astype(np.float64)) / np.sqrt(d)
    sinks = np.full(heads, score + np.log(2), np.float32)
    visible = position - oldest + 1
    maximum = float(sinks[0])
    expected = np.full(
        (1, heads, d),
        np.exp(score - maximum)
        / (1 + np.exp(score - maximum) + (visible - 1) * np.exp(-maximum)),
    )
    arrays = [
        f32_to_bf16(proj),
        f32_to_bf16(kc),
        f32_to_bf16(vc),
        np.ones(d, np.float32),
        np.ones(d, np.float32),
        cos,
        sin,
        sinks,
    ]
    buffers = [nt.Buffer(dev, array.tobytes()) for array in arrays]

    def run(actual_window, actual_sinks):
        pipeline = nt.Pipeline(
            nt.Library(
                dev,
                kernels.PRELUDE + SOURCE,
                {
                    "D": str(d),
                    "ROTARY": str(int(rotary)),
                    "WINDOW": str(actual_window),
                    "SINKS": str(int(actual_sinks)),
                },
            ),
            "causal_attention",
        )
        out = nt.Buffer(dev, heads * d * 2)
        dispatch = nt.Dispatch().pipeline(pipeline)
        for index, buffer in enumerate(buffers):
            dispatch.buffer(index, buffer)
        result = nt.Queue(dev).run(
            [
                dispatch.buffer(8, out)
                .bytes(9, struct.pack("<5If", heads, kv, capacity, 1, position, eps))
                .grid(heads)
                .threadgroup(32)
            ]
        )
        assert not result.error, result.error
        return bf16_to_f32(
            np.frombuffer(out.read(0, heads * d * 2), np.uint16)
        ).reshape(1, heads, d)

    actual = run(window, True)
    assert bf16_ulp_diff(actual, expected).max() <= 2
    # Same inputs and oracle, deliberately incorrect semantic bindings. These
    # checks prove this gate detects bugs that a random 1/513 contribution hides.
    wrong_window = run(512, True)
    no_sinks = run(window, False)
    assert np.max(np.abs(wrong_window - expected)) > 0.25
    assert np.max(np.abs(no_sinks - expected)) > 0.5
