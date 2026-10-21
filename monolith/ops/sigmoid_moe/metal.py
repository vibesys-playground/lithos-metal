"""Native, unfused sigmoid selection and rounded expert summation."""

import struct

from ... import kernels
from ...core.ir import Op
from ..registry import MetalEmitterContext

SOURCE = r"""
#ifndef STEP_STATE
#define STEP_STATE 0
#endif
static inline float bf(ushort x) { return as_type<float>(uint(x)<<16); }
static inline ushort bits(float x) { uint u=as_type<uint>(x); u+=0x7fffu+((u>>16)&1u); return ushort(u>>16); }
static inline float rnd(float x) { return bf(bits(x)); }
struct RouteParams { uint experts, top_k, t_active, hidden; };
kernel void sigmoid_route(device const ushort* logits [[buffer(0)]], device int* ids [[buffer(1)]],
                          device float* weights [[buffer(2)]], constant RouteParams& p [[buffer(3)]],
#if STEP_STATE
                          device const StepState* st [[buffer(15)]],
#endif
                          uint t [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
#if STEP_STATE
  if(st->done || t>=st->t_this_step) return;
#else
  if(t>=p.t_active) return;
#endif
  if(lane!=0u) return;
  // Deliberately serial selection: bounded by E*k and independent of SIMD layout.
  uint chosen[16]; float scores[16], total=0.f;
  for(uint slot=0;slot<p.top_k;++slot) {
    float best=-1.f; uint id=0u;
    for(uint e=0;e<p.experts;++e) {
      bool taken=false;
      for(uint prior=0;prior<slot;++prior) taken|=chosen[prior]==e;
      float score=1.f/(1.f+exp(-bf(logits[t*p.experts+e])));
      if(!taken && score>best) { best=score; id=e; }
    }
    chosen[slot]=id; scores[slot]=best; total+=best;
  }
  for(uint slot=0;slot<p.top_k;++slot) {
    ids[t*p.top_k+slot]=int(chosen[slot]);
    weights[t*p.top_k+slot]=rnd(scores[slot]/(total+1e-20f));
  }
}
kernel void moe_sum(device const ushort* experts [[buffer(0)]], device const float* weights [[buffer(1)]],
                    device const ushort* shared [[buffer(2)]], device const ushort* residual [[buffer(3)]],
                    device ushort* out [[buffer(4)]], constant RouteParams& p [[buffer(5)]],
#if STEP_STATE
                    device const StepState* st [[buffer(15)]],
#endif
                    uint t [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
#if STEP_STATE
  if(st->done || t>=st->t_this_step) return;
#else
  if(t>=p.t_active) return;
#endif
  for(uint col=lane;col<p.hidden;col+=32u) {
    float sum=0.f;
    for(uint slot=0;slot<p.top_k;++slot) sum+=rnd(bf(experts[(t*p.top_k+slot)*p.hidden+col])*weights[t*p.top_k+slot]);
    float y=rnd(rnd(sum)+bf(shared[t*p.hidden+col]));
    out[t*p.hidden+col]=bits(y+bf(residual[t*p.hidden+col]));
  }
}
"""


def emit_route(ctx: MetalEmitterContext, op: Op) -> None:
    (logits,) = op.inputs
    ids, weights = op.outputs
    t, _ = ctx.rows_of(op)
    experts, top_k = op.attrs["n_experts"], op.attrs["top_k"]
    if not 1 <= top_k <= min(16, experts) or ctx.shape(logits) != (t, experts):
        raise ValueError("sigmoid_route: invalid logits shape or top_k")
    k = ctx.kernel(
        "sigmoid_route", kernels.PRELUDE + SOURCE, "sigmoid_route", {"STEP_STATE": "1"}
    )
    params = ctx.params("sigmoid_route", struct.pack("<4I", experts, top_k, t, 0))
    ctx.add(
        k,
        [
            (0, *ctx.buf(logits)),
            (1, *ctx.buf(ids)),
            (2, *ctx.buf(weights)),
            (3, params, 0),
            (15, *ctx.step_state_buffer()),
        ],
        (t, 1, 1),
        (32, 1, 1),
        op.kind,
        writes=[1, 2],
    )


def emit_sum(ctx: MetalEmitterContext, op: Op) -> None:
    experts, weights, shared, residual = op.inputs
    (out,) = op.outputs
    t, _ = ctx.rows_of(op)
    hidden, top_k = ctx.shape(out)[1], op.attrs["top_k"]
    if (
        ctx.shape(experts) != (t, top_k * hidden)
        or ctx.shape(weights) != (t, top_k)
        or ctx.shape(shared) != (t, hidden)
        or ctx.shape(residual) != (t, hidden)
    ):
        raise ValueError("moe_sum: incompatible expert/weight/shared/residual shapes")
    k = ctx.kernel("moe_sum", kernels.PRELUDE + SOURCE, "moe_sum", {"STEP_STATE": "1"})
    params = ctx.params("moe_sum", struct.pack("<4I", 0, top_k, t, hidden))
    bindings = [
        (i, *ctx.buf(v))
        for i, v in enumerate([experts, weights, shared, residual, out])
    ]
    ctx.add(
        k,
        [*bindings, (5, params, 0), (15, *ctx.step_state_buffer())],
        (t, 1, 1),
        (32, 1, 1),
        op.kind,
        writes=[4],
    )
