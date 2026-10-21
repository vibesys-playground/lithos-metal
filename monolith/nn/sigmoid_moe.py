"""Sigmoid-routed experts plus an ungated shared expert, with explicit BF16 boundaries."""

from ..core.dtypes import DType
from ..core.ir import BlockDomain, OpClass
from ..packs.transforms import interleave_chunks
from .linear import Linear, Part
from .module import Module
from .moe import Experts


class SigmoidMoE(Module):
    def __init__(
        self,
        hidden,
        n_experts,
        top_k,
        intermediate,
        shared_intermediate,
        *,
        hf_prefix,
        prefix="",
        chunk=8,
    ):
        super().__init__(prefix=prefix)
        if not 1 <= top_k <= min(16, n_experts):
            raise ValueError("SigmoidMoE.top_k must be in 1..min(16, n_experts)")
        self.hidden, self.n_experts, self.top_k = hidden, n_experts, top_k
        self.router = Linear(
            hidden,
            [Part("gate", f"{hf_prefix}gate.weight", n_experts)],
            prefix=f"{prefix}router.",
        )
        template = f"{hf_prefix}experts.{{e}}."
        self.gate_up = Experts(
            hidden,
            n_experts,
            [
                ("gate_proj", "gate_proj.weight", intermediate),
                ("up_proj", "up_proj.weight", intermediate),
            ],
            hf_template=template,
            prefix=f"{prefix}experts_gate_up.",
            epilogue="silu_mul",
            chunk=chunk,
            round_silu=True,
        )
        self.down = Experts(
            intermediate,
            n_experts,
            [("down_proj", "down_proj.weight", hidden)],
            hf_template=template,
            prefix=f"{prefix}experts_down.",
        )
        si = shared_intermediate
        self.shared_gate_up = Linear(
            hidden,
            [
                Part("gate_proj", f"{hf_prefix}shared_expert.gate_proj.weight", si),
                Part("up_proj", f"{hf_prefix}shared_expert.up_proj.weight", si),
            ],
            prefix=f"{prefix}shared_gate_up.",
            row_perm=interleave_chunks(si, si, chunk),
            epilogue="silu_mul",
            chunk=chunk,
            round_silu=True,
        )
        self.shared_down = Linear(
            si,
            [Part("down_proj", f"{hf_prefix}shared_expert.down_proj.weight", hidden)],
            prefix=f"{prefix}shared_down.",
        )

    def route(self, x):
        import torch

        scores = torch.sigmoid(self.router.forward(x).float())
        ids = torch.argsort(scores, dim=-1, descending=True, stable=True)[
            :, : self.top_k
        ]
        chosen = torch.gather(scores, -1, ids)
        return ids, (chosen / (chosen.sum(-1, keepdim=True) + 1e-20)).to(x.dtype)

    def forward(self, x, residual):
        import torch

        ids, weights = self.route(x)
        routed = torch.zeros_like(x, dtype=torch.float32)
        for t in range(x.shape[0]):
            for slot in range(self.top_k):
                expert = int(ids[t, slot])
                act = self.gate_up.forward_expert(expert, x[t : t + 1])
                out = self.down.forward_expert(expert, act)
                routed[t] += (out[0] * weights[t, slot]).float()
        shared = self.shared_down.forward(self.shared_gate_up.forward(x))
        return (routed.to(x.dtype) + shared) + residual

    def lower(self, g, h, norm, ctx):
        t, p = h.shape[0], self.prefix
        logits = self.router.lower(g, h, norm=norm).value
        xn = g.value(f"{p}xn", h.shape, DType.BF16)
        g.op(
            "norm_apply",
            [h, norm.stat, norm.weight],
            [xn],
            domain=BlockDomain("span", self.hidden),
            klass=OpClass.MAP,
            eps=norm.eps,
        )
        ids = g.value(f"{p}ids", (t, self.top_k), DType.I32)
        weights = g.value(f"{p}weights", (t, self.top_k), DType.F32)
        g.op(
            "sigmoid_route",
            [logits],
            [ids, weights],
            domain=BlockDomain("rows", t),
            klass=OpClass.MAP,
            n_experts=self.n_experts,
            top_k=self.top_k,
        )
        act = self.gate_up.lower(
            g, xn, ids, self.top_k, x_per_slot=False, name=f"{p}act"
        )
        experts = self.down.lower(
            g, act, ids, self.top_k, x_per_slot=True, name=f"{p}experts"
        )
        shared = self.shared_down.lower(
            g, self.shared_gate_up.lower(g, h, norm=norm).value
        ).value
        out = g.value(f"{p}h", h.shape, DType.BF16)
        g.op(
            "moe_sum",
            [experts, weights, shared, h],
            [out],
            domain=BlockDomain("rows", t),
            klass=OpClass.MAP,
            top_k=self.top_k,
        )
        return out
