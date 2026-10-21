"""Validated text-only Flint configuration; unknown keys fail at the boundary."""

import json
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class FlintConfig:
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_experts: int
    num_experts_per_tok: int
    vocab_size: int
    max_position_embeddings: int
    sliding_window: int
    rms_norm_eps: float
    rope_theta: float
    layer_types: tuple[Literal["full_attention", "sliding_attention"], ...]
    no_rope_layer_indices: tuple[int, ...]
    mlp_only_layers: tuple[int, ...]
    eos_token_id: list[int]
    tie_word_embeddings: bool = False
    architecture: str = "FlintForCausalLM"

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise TypeError("config must be an object")
        supported = {
            "architectures": ["FlintForCausalLM"],
            "model_type": "flint",
            "attention_bias": False,
            "attention_dropout": 0.0,
            "attention_sink": True,
            "decoder_sparse_step": 1,
            "dtype": "bfloat16",
            "torch_dtype": "bfloat16",
            "hidden_act": "silu",
            "norm_topk_prob": True,
            "output_router_logits": False,
            "router_aux_loss_coef": 0.0,
            "router_expert_bias": False,
            "scoring_func": "sigmoid",
            "router_score_function": "sigmoid",
            "softmax_type": "learnable",
            "softmax_type_source": "model.attention_pattern.softmax_type",
            "use_cache": True,
            "use_sliding_window": True,
            # Cache hints are accepted metadata; this engine keeps BF16 KV.
            "kv_group_size": 64,
            "kv_key_bits": 8,
            "kv_value_bits": 8,
        }
        special = {
            "mlx_export",
            "quantization",
            "rope_parameters",
            "num_local_experts",
            "bos_token_id",
            "pad_token_id",
        }
        names = {f.name for f in fields(cls)} - {"architecture"}
        unknown = data.keys() - names - supported.keys() - special
        if unknown:
            raise ValueError(f"unknown config key {min(unknown)}")
        for key, value in supported.items():
            if key in data and (
                data[key] != value or type(data[key]) is not type(value)
            ):
                raise ValueError(
                    f"unsupported {key}: {data[key]!r}; expected {value!r}"
                )
        for key, allowed in {
            "quantization": {"bits", "group_size", "mode"},
            "mlx_export": {
                "bits",
                "routed_expert_bits",
                "embedding_bits",
                "full_attention_layers",
                "scale_rule",
            },
            "rope_parameters": {"rope_type", "rope_theta"},
        }.items():
            values = data.get(key, {})
            if not isinstance(values, dict):
                raise TypeError(f"{key} must be an object")
            for subkey in values.keys() - allowed:
                raise ValueError(f"unknown config key {key}.{subkey}")
        required = names - {"tie_word_embeddings"}
        for key in sorted(required - data.keys()):
            raise ValueError(f"missing config key {key}")
        values = {key: data[key] for key in names if key in data}
        for key in (
            "layer_types",
            "no_rope_layer_indices",
            "mlp_only_layers",
            "eos_token_id",
        ):
            if not isinstance(values[key], list):
                raise TypeError(f"{key} must be a list")
            if key != "eos_token_id":
                values[key] = tuple(values[key])
        cfg = cls(**values)
        for key in (
            "hidden_size",
            "intermediate_size",
            "moe_intermediate_size",
            "shared_expert_intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "num_experts",
            "num_experts_per_tok",
            "vocab_size",
            "max_position_embeddings",
            "sliding_window",
        ):
            value = getattr(cfg, key)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        for key in ("rms_norm_eps", "rope_theta"):
            value = getattr(cfg, key)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{key} must be finite and positive")
        for key in (
            "hidden_size",
            "intermediate_size",
            "moe_intermediate_size",
            "shared_expert_intermediate_size",
        ):
            if getattr(cfg, key) % 256:
                raise ValueError(
                    f"{key} must be a multiple of 256 for packed projections"
                )
        if cfg.num_attention_heads * cfg.head_dim % 256:
            raise ValueError("num_attention_heads * head_dim must be a multiple of 256")
        if cfg.tie_word_embeddings is not False:
            raise ValueError("unsupported tie_word_embeddings")
        if cfg.num_attention_heads % cfg.num_key_value_heads:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        if cfg.head_dim % 32:
            raise ValueError("head_dim must be a multiple of 32")
        if not 1 <= cfg.num_experts_per_tok <= min(16, cfg.num_experts):
            raise ValueError("num_experts_per_tok must be in 1..min(16, num_experts)")
        if len(cfg.layer_types) != cfg.num_hidden_layers or any(
            k not in ("full_attention", "sliding_attention") for k in cfg.layer_types
        ):
            raise ValueError(
                "layer_types must contain one supported attention kind per layer"
            )
        for key in ("no_rope_layer_indices", "mlp_only_layers"):
            indices = getattr(cfg, key)
            if any(
                type(i) is not int or not 0 <= i < cfg.num_hidden_layers
                for i in indices
            ) or len(set(indices)) != len(indices):
                raise ValueError(f"{key} must contain distinct layer indices")
        if not cfg.eos_token_id or any(
            type(i) is not int or not 0 <= i < cfg.vocab_size for i in cfg.eos_token_id
        ):
            raise ValueError("eos_token_id must contain valid vocabulary IDs")
        for key in ("bos_token_id", "pad_token_id"):
            if key in data and (
                type(data[key]) is not int or not 0 <= data[key] < cfg.vocab_size
            ):
                raise ValueError(f"{key} must be a vocabulary ID")
        if data.get("num_local_experts", cfg.num_experts) != cfg.num_experts:
            raise ValueError("num_local_experts must equal num_experts")
        expected = {
            "quantization": {"bits": 4, "group_size": 64},
            "mlx_export": {"bits": 4, "routed_expert_bits": 4, "embedding_bits": 8},
        }
        for key, entries in expected.items():
            for subkey, value in entries.items():
                if data.get(key, {}).get(subkey) != value:
                    raise ValueError(f"unsupported {key}.{subkey}: expected {value}")
        quant = data["quantization"]
        if quant.get("mode", "affine") != "affine":
            raise ValueError("unsupported quantization.mode")
        export = data["mlx_export"]
        if export.get("scale_rule", "mse") != "mse":
            raise ValueError("unsupported mlx_export.scale_rule")
        full = [i for i, kind in enumerate(cfg.layer_types) if kind == "full_attention"]
        if export.get("full_attention_layers", full) != full:
            raise ValueError(
                "mlx_export.full_attention_layers disagrees with layer_types"
            )
        rope = data.get("rope_parameters", {})
        if rope.get("rope_type", "default") != "default":
            raise ValueError("unsupported rope_parameters.rope_type")
        if rope.get("rope_theta", cfg.rope_theta) != cfg.rope_theta:
            raise ValueError("rope_parameters.rope_theta disagrees with rope_theta")
        return cfg

    @classmethod
    def from_pretrained(cls, path):
        return cls.from_dict(json.loads((Path(path) / "config.json").read_text()))

    @property
    def rotary_dim(self):
        return self.head_dim
