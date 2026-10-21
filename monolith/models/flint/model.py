"""Flint composition through shared dense, causal-attention and sigmoid-MoE layers."""

import numpy as np

from ...core.dtypes import DType
from ...core.shapes import T
from ...nn import (
    DecoderLayer,
    Embedding,
    GatedMLP,
    GreedySampler,
    LMHead,
    LowerContext,
    Model,
    RMSNorm,
    StateSpec,
    state_shape,
)
from ...nn.causal_attention import CausalAttention
from ...nn.rope import rope_tables_permuted
from ...nn.sigmoid_moe import SigmoidMoE
from ..registry import register_model
from .config import FlintConfig
from .weights import bind_checkpoint_formats, checkpoint_views


@register_model("FlintForCausalLM")
class FlintModel(Model):
    checkpoint_views = staticmethod(checkpoint_views)

    def __init__(
        self, config, *, max_context=4096, pack_rows=16, num_layers_override=None
    ):
        super().__init__(prefix="")
        c = self.config = config
        if not 0 < max_context <= c.max_position_embeddings:
            raise ValueError("max_context must be in 1..max_position_embeddings")
        self.max_context = max_context
        self.n_layers = (
            c.num_hidden_layers if num_layers_override is None else num_layers_override
        )
        if not 0 < self.n_layers <= c.num_hidden_layers:
            raise ValueError("num_layers_override must be in 1..num_hidden_layers")
        self.embed_tokens = Embedding(
            c.vocab_size,
            c.hidden_size,
            "model.embed_tokens.weight",
            prefix="embed_tokens.",
        )
        self.blocks = []
        for i in range(self.n_layers):
            lp, hp = f"layers.{i}.", f"model.layers.{i}."
            mixer = CausalAttention(
                c.hidden_size,
                c.num_attention_heads,
                c.num_key_value_heads,
                c.head_dim,
                c.rope_theta,
                c.rms_norm_eps,
                hf_prefix=f"{hp}self_attn.",
                prefix=f"{lp}self_attn.",
                max_context=max_context,
                rotary=i not in c.no_rope_layer_indices,
                window=c.sliding_window
                if c.layer_types[i] == "sliding_attention"
                else 0,
            )
            if i in c.mlp_only_layers:
                mlp = GatedMLP(
                    c.hidden_size,
                    c.intermediate_size,
                    hf_prefix=f"{hp}mlp.",
                    prefix=f"{lp}mlp.",
                    chunk=pack_rows // 2,
                )
                mlp.gate_up.round_silu = True
                mlp.down.round_residual = True
            else:
                mlp = SigmoidMoE(
                    c.hidden_size,
                    c.num_experts,
                    c.num_experts_per_tok,
                    c.moe_intermediate_size,
                    c.shared_expert_intermediate_size,
                    hf_prefix=f"{hp}mlp.",
                    prefix=f"{lp}mlp.",
                    chunk=pack_rows // 2,
                )
            self.blocks.append(
                DecoderLayer(
                    i,
                    RMSNorm(
                        c.hidden_size,
                        c.rms_norm_eps,
                        f"{hp}input_layernorm.weight",
                        prefix=f"{lp}input_norm.",
                        one_plus=False,
                    ),
                    mixer,
                    RMSNorm(
                        c.hidden_size,
                        c.rms_norm_eps,
                        f"{hp}post_attention_layernorm.weight",
                        prefix=f"{lp}post_norm.",
                        one_plus=False,
                    ),
                    mlp,
                    prefix=lp,
                )
            )
        self.norm = RMSNorm(
            c.hidden_size,
            c.rms_norm_eps,
            "model.norm.weight",
            prefix="norm.",
            one_plus=False,
        )
        self.lm_head = LMHead(
            c.hidden_size, c.vocab_size, hf_name="lm_head.weight", prefix="lm_head."
        )
        self.sampler = GreedySampler(prefix="sampler.")
        self.tap_values = {}

    @classmethod
    def from_checkpoint(cls, path, **options):
        model = cls(FlintConfig.from_pretrained(path), **options)
        bind_checkpoint_formats(model, path)
        return model

    def layers(self):
        return self.blocks

    def state_spec(self):
        return StateSpec(
            tuple(
                entry for block in self.blocks for entry in block.mixer.state_entries()
            )
        )

    def feature_taps(self):
        return list(range(self.n_layers))

    def tables(self):
        if all(i in self.config.no_rope_layer_indices for i in range(self.n_layers)):
            return {}
        cos, sin = rope_tables_permuted(
            self.config.rope_theta,
            self.config.head_dim,
            self.config.head_dim,
            self.max_context,
        )
        return {"causal_rope_cos": ("F32", cos), "causal_rope_sin": ("F32", sin)}

    def forward(self, ids, state, pos):
        h = self.embed_tokens.forward(ids)
        hidden = [h]
        for block in self.blocks:
            h = block.forward(h, state, pos)
            hidden.append(h)
        return self.lm_head.forward(self.norm.forward(h)), hidden, state

    def lower(self, g, *xs):
        tokens = xs[0] if xs else g.input("tokens", (T,), DType.I32)
        ctx = LowerContext(t=tokens.shape[0])
        for entry in self.state_spec().entries:
            ctx.states[entry.name] = g.state(
                entry.name, state_shape(entry), entry.dtype
            )
        for name, (dtype, arr) in self.tables().items():
            ctx.consts[name] = g.const(
                name, np.asarray(arr).shape, DType.parse(dtype.lower())
            )
        h = self.embed_tokens.lower(g, tokens, ctx)
        self.tap_values = {-1: h}
        for block in self.blocks:
            h = block.lower(g, h, ctx)
            self.tap_values[block.index] = h
        logits = self.lm_head.lower(g, h, self.norm.lower(g, h), ctx)
        return self.sampler.lower(g, logits, ctx)
