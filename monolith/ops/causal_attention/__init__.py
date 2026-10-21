"""Parameterized causal GQA with optional rotary positions and attention sinks.

Inputs: projected Q/K/V BF16 [T,(H+2*KV)*D], K/V BF16 [capacity,KV,D],
Q/K norm weights F32 [D], cos/sin F32 [capacity,D], sinks F32 [H].
Output BF16 [T,H*D]. Cache updates append every new K/V at absolute positions;
old entries are immutable. Q/K normalization precedes optional half-split RoPE.

``window=0`` means all causal history; W>0 retains keys p-W+1..p inclusive.
``rotary=False`` never reads tables. ``sinks=False`` never reads sinks; otherwise
softmax includes one learned logit per query head with zero value. Scores,
softmax and accumulation are F32; norm, RoPE, output and stored K/V round to BF16.
All dimensions and positive capacities are validated by the shared layer/emitter.
No ring buffer, speculative cache commit, fusion, or device-specific policy.
"""

from ...core.ir import OpClass
from ..registry import KernelBinding, OpDef, register_op
from .metal import emit

CAUSAL_ATTENTION = register_op(
    OpDef(
        "causal_attention", OpClass.MAP, "heads", metal_emitter=emit, cache_input=1
    ).bind("*", KernelBinding("causal_attention"))
)
