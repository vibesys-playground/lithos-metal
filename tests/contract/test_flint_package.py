"""Flint's public model contract on real metadata and a tiny stored checkpoint."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from monolith.compiler import compile_program
from monolith.compiler.coverage import check_coverage
from monolith.core import Graph
from monolith.core.profile import Profile
from monolith.formats import FORMATS, PackLayout
from monolith.formats.fp import f32_to_bf16
from monolith.formats.safetensors_reader import write_safetensors
from monolith.models import resolve_model
from monolith.models.flint import FlintConfig, FlintModel
from monolith.nn.pack_plan import pack_model
from monolith.packs import PackFile

CONFIG = json.loads(
    (Path(__file__).parents[1] / "models/flint/config.json").read_text()
)


def tiny_config():
    cfg = copy.deepcopy(CONFIG)
    cfg.update(
        hidden_size=256,
        intermediate_size=256,
        moe_intermediate_size=256,
        shared_expert_intermediate_size=256,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        num_experts=4,
        num_local_experts=4,
        num_experts_per_tok=2,
        vocab_size=256,
        max_position_embeddings=1024,
        layer_types=["full_attention", "sliding_attention", "full_attention"],
        no_rope_layer_indices=[0, 2],
        mlp_only_layers=[0],
        eos_token_id=[255],
        bos_token_id=254,
        pad_token_id=255,
        sliding_window=3,
    )
    cfg["mlx_export"]["full_attention_layers"] = [0, 2]
    return cfg


def write_checkpoint(path):
    cfg = tiny_config()
    model = FlintModel(FlintConfig.from_dict(cfg), max_context=16)
    rng = np.random.default_rng(31)
    tensors, stacked = {}, {}
    for name, (_, _, spec) in model.full_weight_map().items():
        w = rng.normal(0, 0.02, spec.shape).astype(np.float32)
        if spec.aux or ".mlp.gate.weight" in name:
            values = {name: ("BF16", f32_to_bf16(w))}
        else:
            fmt = FORMATS.get(
                "int8_affine" if "embed_tokens" in name else "int4_affine"
            )
            quant = fmt.quantize(w)
            values = {
                name: ("U8", quant.tensors["weight"]),
                name[:-6] + "scales": ("BF16", f32_to_bf16(quant.tensors["scales"])),
                name[:-6] + "biases": ("BF16", f32_to_bf16(quant.tensors["biases"])),
            }
        for key, (dtype, value) in values.items():
            if ".mlp.experts." in key:
                before, after = key.split(".mlp.experts.")
                expert, suffix = after.split(".", 1)
                source = f"{before}.mlp.switch_mlp.{suffix}"
                stacked.setdefault(source, {})[int(expert)] = (dtype, value)
            else:
                tensors[key] = (dtype, value)
    for key, experts in stacked.items():
        tensors[key] = (
            experts[0][0],
            np.stack([experts[i][1] for i in range(cfg["num_experts"])]),
        )
    (path / "config.json").write_text(json.dumps(cfg))
    write_safetensors(path / "model-00000.safetensors", tensors)
    return model


def test_pinned_config_facts():
    cfg = FlintConfig.from_dict(CONFIG)
    assert cfg.num_hidden_layers == 49
    assert cfg.no_rope_layer_indices == (
        0,
        1,
        5,
        9,
        13,
        17,
        21,
        25,
        29,
        33,
        37,
        41,
        45,
        48,
    )
    assert cfg.num_experts == 256 and cfg.num_experts_per_tok == 6
    assert cfg.sliding_window == 513 and cfg.mlp_only_layers == (0, 1)


@pytest.mark.parametrize(
    "path,value",
    [
        ("unrecognized", 1),
        ("hidden_act", "gelu"),
        ("attention_bias", True),
        ("quantization.unrecognized", 1),
        ("quantization.bits", 8),
        ("mlx_export.embedding_bits", 4),
        ("mlx_export.routed_expert_bits", 3),
        ("rope_parameters.rope_type", "yarn"),
        ("rope_parameters.unknown", 1),
        ("router_expert_bias", True),
        ("num_local_experts", 255),
        ("hidden_size", 0),
        ("no_rope_layer_indices", [49]),
        ("layer_types", ["bad"]),
        ("eos_token_id", [-1]),
    ],
)
def test_rejects_unsupported_keys_by_name(path, value):
    cfg = copy.deepcopy(CONFIG)
    node = cfg
    parts = path.split(".")
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value
    with pytest.raises(ValueError, match=path.replace(".", r"\.")):
        FlintConfig.from_dict(cfg)


def test_checkpoint_graph_and_pack_contract(tmp_path):
    write_checkpoint(tmp_path)
    model = FlintModel.from_checkpoint(tmp_path, max_context=16)
    assert resolve_model("FlintForCausalLM") is FlintModel
    norms = [
        model.norm,
        *[
            norm
            for block in model.layers()
            for norm in (block.input_norm, block.post_norm)
        ],
    ]
    assert all(norm.weight_map()["weight"].transform == "bf16_f32" for norm in norms)
    assert model.embed_tokens.format_of("weight") == "int8_affine"
    assert model.lm_head.format_of("weight") == "int4_affine"
    graph = Graph("step")
    model.lower(graph)
    graph.check()
    profile = Profile.from_dict(
        "m3",
        {
            "gpu_cores": 18,
            "nominal_gbps": 150,
            "engine": {"family": "Apple9", "lane_order": "interleaved16"},
        },
    )
    check_coverage(graph, profile)
    attention = [op for op in graph.ops if op.kind == "causal_attention"]
    assert [op.attrs["window"] for op in attention] == [0, 3, 0]
    assert [op.attrs["rotary"] for op in attention] == [False, True, False]
    assert all(op.attrs["sinks"] and len(op.attrs["updates"]) == 2 for op in attention)
    pack_model(model, str(tmp_path), str(tmp_path / "pack"), PackLayout())
    pack = PackFile(tmp_path / "pack")
    for dynamic in (False, True):
        program = compile_program(
            model, pack, profile, t=3, dynamic_t=dynamic, commute_norm=False
        )
        assert program.context_capacity == 16
        emitted = [op for op in program.ops if op.name == "causal_attention"]
        assert len(emitted) == 3
        assert all(op.meta["writes"] == [1, 2, 8] for op in emitted)
        for op in program.ops:
            if op.name in ("causal_attention", "sigmoid_route", "moe_sum"):
                assert program.kernels[op.kernel].macros["STEP_STATE"] == "1"
                assert (15, program.step_state, 0) in op.bindings
