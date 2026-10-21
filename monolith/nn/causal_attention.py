"""Shared QK-normalized causal GQA: window, rotary and sink policy are explicit.

Composes existing projection, residual and cache mechanisms. The registered
operation owns arithmetic and Metal emission, independently of any model.
"""

from ..core.dtypes import DType
from ..core.ir import BlockDomain, OpClass
from .attention import GQAAttention
from .module import WeightSpec


class CausalAttention(GQAAttention):
    def __init__(
        self,
        hidden,
        heads,
        kv_heads,
        head_dim,
        rope_theta,
        eps,
        *,
        hf_prefix,
        prefix="",
        max_context,
        window=0,
        rotary=True,
        sinks=True,
    ):
        if type(window) is not int or window < 0:
            raise ValueError("CausalAttention.window must be a nonnegative integer")
        if max_context <= 0:
            raise ValueError("CausalAttention.max_context must be positive")
        if head_dim <= 0 or head_dim % 32:
            raise ValueError(
                "CausalAttention.head_dim must be a positive multiple of 32"
            )
        if heads <= 0 or kv_heads <= 0:
            raise ValueError("CausalAttention head counts must be positive")
        super().__init__(
            hidden,
            heads,
            kv_heads,
            head_dim,
            head_dim if rotary else 0,
            rope_theta,
            eps,
            hf_prefix=hf_prefix,
            prefix=prefix,
            max_context=max_context,
            gate=False,
            norm_one_plus=False,
        )
        self.window, self.rotary, self.sinks = window, rotary, sinks
        self.o_proj.round_residual = True

    def weight_map(self):
        weights = super().weight_map()
        if self.sinks:
            weights["sinks"] = WeightSpec(
                f"{self.hf_prefix}softmax_offset",
                (self.heads,),
                "f32",
                transform="f32",
                aux=True,
            )
        return weights

    def mix(self, proj, state, pos):
        import torch

        from . import oracle

        t, d = proj.shape[0], self.head_dim
        hd, kd = self.heads * d, self.kv_heads * d
        q = proj[:, :hd].reshape(t, self.heads, d)
        k = proj[:, hd : hd + kd].reshape(t, self.kv_heads, d)
        v = proj[:, hd + kd :].reshape(t, self.kv_heads, d)
        q = oracle.rms_norm(q, self.param("q_norm"), self.eps, one_plus=False)
        k = oracle.rms_norm(k, self.param("k_norm"), self.eps, one_plus=False)
        if self.rotary:
            positions = torch.arange(pos, pos + t, dtype=torch.float32)
            inv = self.rope_theta ** (-torch.arange(0, d, 2, dtype=torch.float32) / d)
            angles = positions[:, None] * inv[None, :]
            cos = angles.cos()[:, None, :]
            sin = angles.sin()[:, None, :]

            def rotate(x):
                a, b = x.float().chunk(2, dim=-1)
                return torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1).to(
                    proj.dtype
                )

            q, k = rotate(q), rotate(k)
        kc, vc = state[f"{self.prefix}k_cache"], state[f"{self.prefix}v_cache"]
        kc[pos : pos + t], vc[pos : pos + t] = k, v
        keys, values = kc[: pos + t], vc[: pos + t]
        keys = keys.repeat_interleave(self.heads // self.kv_heads, dim=1).float()
        values = values.repeat_interleave(self.heads // self.kv_heads, dim=1).float()
        scores = torch.einsum("thd,khd->htk", q.float(), keys) * d**-0.5
        qp, kp = torch.arange(pos, pos + t)[:, None], torch.arange(pos + t)[None, :]
        valid = kp <= qp
        if self.window:
            valid &= kp > qp - self.window
        scores = scores.masked_fill(~valid[None], -torch.inf)
        if self.sinks:
            sink = self.param("sinks").float()[:, None, None].expand(-1, t, 1)
            probs = torch.softmax(torch.cat((scores, sink), dim=-1), dim=-1)[..., :-1]
        else:
            probs = torch.softmax(scores, dim=-1)
        return torch.einsum("htk,khd->thd", probs, values).to(proj.dtype).reshape(t, hd)

    def lower(self, g, h, norm, ctx):
        if ctx.mixer_attrs:
            raise ValueError(
                f"causal_attention: unsupported mixer attribute {min(ctx.mixer_attrs)}"
            )
        proj = self.qkv.lower(g, h, norm=norm).value
        kc, vc = (
            ctx.states[f"{self.prefix}k_cache"],
            ctx.states[f"{self.prefix}v_cache"],
        )
        qn = self.const_value(g, f"{self.prefix}q_norm", (self.head_dim,), DType.F32)
        kn = self.const_value(g, f"{self.prefix}k_norm", (self.head_dim,), DType.F32)
        # Disabled optional inputs alias an existing constant and are never read.
        cos, sin = (
            (ctx.consts["causal_rope_cos"], ctx.consts["causal_rope_sin"])
            if self.rotary
            else (qn, qn)
        )
        sinks = (
            self.const_value(g, f"{self.prefix}sinks", (self.heads,), DType.F32)
            if self.sinks
            else qn
        )
        out = g.value(
            f"{self.prefix}attn", (h.shape[0], self.heads * self.head_dim), DType.BF16
        )
        g.op(
            "causal_attention",
            [proj, kc, vc, qn, kn, cos, sin, sinks],
            [out],
            domain=BlockDomain("heads", self.heads),
            klass=OpClass.MAP,
            updates=[kc.name, vc.name],
            heads=self.heads,
            kv_heads=self.kv_heads,
            head_dim=self.head_dim,
            eps=self.eps,
            window=self.window,
            rotary=self.rotary,
            sinks=self.sinks,
        )
        return self.o_proj.lower(g, out, residual=h, name=f"{self.prefix}h").value
