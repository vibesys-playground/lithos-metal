"""Shared extensions expose state, optional inputs, routing and arithmetic contracts."""

import pytest

from monolith.compiler.coverage import check_coverage
from monolith.core import Graph
from monolith.core.dtypes import DType
from monolith.core.profile import Profile
from monolith.core.shapes import T
from monolith.nn import CausalAttention, LowerContext, RMSNorm, SigmoidMoE, state_shape
from monolith.ops import OPS


@pytest.mark.parametrize(
    "window,rotary,sinks", [(0, False, True), (3, True, True), (1, False, False)]
)
def test_shared_attention_graph_contract(window, rotary, sinks):
    layer = CausalAttention(
        256,
        4,
        2,
        64,
        10000.0,
        1e-6,
        hf_prefix="attn.",
        prefix="attn.",
        max_context=16,
        window=window,
        rotary=rotary,
        sinks=sinks,
    )
    graph, ctx = Graph("attention"), LowerContext()
    h = graph.input("h", (T, 256), DType.BF16)
    for entry in layer.state_entries():
        ctx.states[entry.name] = graph.state(
            entry.name, state_shape(entry), entry.dtype
        )
    if rotary:
        for key in ("causal_rope_cos", "causal_rope_sin"):
            ctx.consts[key] = graph.const(key, (16, 64), DType.F32)
    norm = RMSNorm(256, 1e-6, "norm.weight", prefix="norm.")
    out = layer.lower(graph, h, norm.lower(graph, h), ctx)
    graph.check()
    assert out.shape == h.shape
    (op,) = [op for op in graph.ops if op.kind == "causal_attention"]
    assert (
        op.attrs["window"] == window
        and op.attrs["rotary"] is rotary
        and op.attrs["sinks"] is sinks
    )
    assert op.attrs["updates"] == ["attn.k_cache", "attn.v_cache"]
    assert ("sinks" in layer.weight_map()) is sinks
    definition = OPS.get(op.kind)
    assert definition.cache_input == 1 and callable(definition.metal_emitter)
    profile = Profile.from_dict(
        "test",
        {
            "gpu_cores": 18,
            "nominal_gbps": 150,
            "engine": {"family": "Apple9", "lane_order": "interleaved16"},
        },
    )
    check_coverage(graph, profile)


@pytest.mark.parametrize(
    "kwargs,key",
    [
        ({"window": -1}, "window"),
        ({"max_context": 0}, "max_context"),
        ({"head_dim": 63}, "head_dim"),
        ({"kv_heads": 0}, "head counts"),
    ],
)
def test_shared_attention_rejects_invalid_parameters(kwargs, key):
    args = {
        "hidden": 256,
        "heads": 4,
        "kv_heads": 2,
        "head_dim": 64,
        "rope_theta": 10000.0,
        "eps": 1e-6,
        "hf_prefix": "attn.",
        "max_context": 16,
    }
    args.update(kwargs)
    with pytest.raises(ValueError, match=key):
        CausalAttention(**args)


def test_sigmoid_moe_composes_existing_expert_projections():
    layer = SigmoidMoE(256, 4, 2, 256, 256, hf_prefix="mlp.", prefix="mlp.")
    graph, ctx = Graph("moe"), LowerContext()
    h = graph.input("h", (T, 256), DType.BF16)
    norm = RMSNorm(256, 1e-6, "norm.weight", prefix="norm.")
    result = layer.lower(graph, h, norm.lower(graph, h), ctx)
    graph.check()
    assert result.shape == h.shape
    assert [op.kind for op in graph.ops].count("moe_gemv") == 2
    assert [op.kind for op in graph.ops].count("sigmoid_route") == 1
    assert [op.kind for op in graph.ops].count("moe_sum") == 1
    assert all(
        op.attrs.get("round_silu")
        for op in graph.ops
        if op.kind == "moe_gemv" and op.attrs.get("epilogue") == "silu_mul"
    )
    assert not any("shared_expert_gate" in name for name in layer.full_weight_map())


def test_attention_rejects_unsupported_draft_mixer_mode():
    layer = CausalAttention(
        256, 4, 2, 64, 10000.0, 1e-6, hf_prefix="attn.", max_context=16
    )
    graph = Graph("draft")
    h = graph.input("h", (T, 256), DType.BF16)
    with pytest.raises(ValueError, match="unsupported mixer attribute lm_mode"):
        layer.lower(graph, h, None, LowerContext(mixer_attrs={"lm_mode": 2}))
