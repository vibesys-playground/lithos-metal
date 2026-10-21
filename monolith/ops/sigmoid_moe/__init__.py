"""Sigmoid routing and separately rounded mixture summation.

sigmoid_route: logits BF16 [T,E] -> selected ids I32 [T,k], weights F32
[T,k] holding BF16 values. Select largest sigmoid(logit), ties lowest id,
normalize selected scores with denominator sum + 1e-20, then round weights.
No selection bias. E>0, 1<=k<=min(E,16).

moe_sum: expert outputs BF16 [T,k*H], weights F32 [T,k], shared BF16
[T,H], residual BF16 [T,H] -> BF16 [T,H]. Round each weighted expert
product, accumulate F32, round routed sum, add shared and round, add
residual and round. The shared expert is ungated. No persistent state.
"""

from ...core.ir import OpClass
from ..registry import KernelBinding, OpDef, register_op
from .metal import emit_route, emit_sum

SIGMOID_ROUTE = register_op(
    OpDef("sigmoid_route", OpClass.MAP, "rows", metal_emitter=emit_route).bind(
        "*", KernelBinding("sigmoid_route")
    )
)
MOE_SUM = register_op(
    OpDef("moe_sum", OpClass.MAP, "rows", metal_emitter=emit_sum).bind(
        "*", KernelBinding("moe_sum")
    )
)
