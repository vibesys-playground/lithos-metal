"""Conservative native Metal implementation of the causal attention contract."""

import struct

from ... import kernels
from ...core.ir import Op
from ..registry import MetalEmitterContext

SOURCE = r"""
#ifndef STEP_STATE
#define STEP_STATE 0
#endif
#define DL (D / 32u)
static inline float bf(ushort x) { return as_type<float>(uint(x) << 16); }
static inline ushort bits(float x) {
  uint u = as_type<uint>(x); u += 0x7fffu + ((u >> 16) & 1u); return ushort(u >> 16);
}
static inline float rnd(float x) { return bf(bits(x)); }
struct CausalParams { uint heads, kv_heads, capacity, t_active, position; float eps; };
static inline void prepare(thread float* f, device const float* w, device const float* cs,
                           device const float* sn, uint pos, float eps, uint lane) {
  float ss = 0.f;
  for (uint e=0; e<DL; ++e) ss += f[e]*f[e];
  float r = rsqrt(simd_sum(ss) / float(D) + eps);
  for (uint e=0; e<DL; ++e) f[e] = rnd(f[e]*r*w[lane*DL+e]);
#if ROTARY
  float rotated[DL];
  for (uint e=0; e<DL; ++e) {
    uint dim=lane*DL+e;
    float partner=simd_shuffle_xor(f[e], 16u);
    float sign=lane<16u ? -1.f : 1.f;
    rotated[e]=rnd(f[e]*cs[pos*D+dim]+sign*partner*sn[pos*D+dim]);
  }
  for (uint e=0; e<DL; ++e) f[e]=rotated[e];
#endif
}
kernel void causal_attention(device const ushort* proj [[buffer(0)]],
                             device ushort* kc [[buffer(1)]], device ushort* vc [[buffer(2)]],
                             device const float* qn [[buffer(3)]], device const float* kn [[buffer(4)]],
                             device const float* cs [[buffer(5)]], device const float* sn [[buffer(6)]],
                             device const float* sinks [[buffer(7)]], device ushort* out [[buffer(8)]],
                             constant CausalParams& p [[buffer(9)]],
#if STEP_STATE
                             device const StepState* st [[buffer(15)]],
#endif
                             uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
#if STEP_STATE
  if(st->done) return;
  uint T=st->t_this_step, pos=st->position;
#else
  uint T=p.t_active, pos=p.position;
#endif
  uint t=row/p.heads, h=row%p.heads, rep=p.heads/p.kv_heads, j=h/rep;
  if(t>=T || pos>=p.capacity || T>p.capacity-pos) return;
  uint stride=(p.heads+2u*p.kv_heads)*D, qp=pos+t;
  float q[DL];
  for(uint e=0; e<DL; ++e) q[e]=bf(proj[t*stride+h*D+lane*DL+e]);
  prepare(q,qn,cs,sn,qp,p.eps,lane);
#if SINKS
  float m=sinks[h], den=1.f;
#else
  float m=-INFINITY, den=0.f;
#endif
  float acc[DL];
  for(uint e=0;e<DL;++e) acc[e]=0.f;
  // New keys are read from proj, never from another threadgroup's stores.
  // The first query of the first head per KV head owns every append.
  for(uint key=0;key<pos+T;++key) {
    bool live=key<=qp && (WINDOW==0u || key+WINDOW>qp);
    bool append=key>=pos && t==0u && h%rep==0u;
    if(!live && !append) continue;
    float k[DL],v[DL];
    for(uint e=0;e<DL;++e) {
      uint dim=lane*DL+e;
      k[e]=key<pos ? bf(kc[(key*p.kv_heads+j)*D+dim]) : bf(proj[(key-pos)*stride+p.heads*D+j*D+dim]);
      v[e]=key<pos ? bf(vc[(key*p.kv_heads+j)*D+dim]) : bf(proj[(key-pos)*stride+(p.heads+p.kv_heads)*D+j*D+dim]);
    }
    if(key>=pos) prepare(k,kn,cs,sn,key,p.eps,lane);
    if(append) for(uint e=0;e<DL;++e) {
      uint off=(key*p.kv_heads+j)*D+lane*DL+e;
      kc[off]=bits(k[e]); vc[off]=bits(v[e]);
    }
    if(!live) continue;
    float dot=0.f;
    for(uint e=0;e<DL;++e) dot+=q[e]*k[e];
    float score=simd_sum(dot)*rsqrt(float(D));
    float next=max(m,score), a=m==-INFINITY ? 0.f : exp(m-next), b=exp(score-next);
    den=den*a+b;
    for(uint e=0;e<DL;++e) acc[e]=acc[e]*a+b*v[e];
    m=next;
  }
  for(uint e=0;e<DL;++e) out[t*p.heads*D+h*D+lane*DL+e]=bits(acc[e]/den);
}
"""


def emit(ctx: MetalEmitterContext, op: Op) -> None:
    proj, kc, vc, qn, kn, cos, sin, sinks = op.inputs
    (out,) = op.outputs
    a = op.attrs
    heads, kv, d = a["heads"], a["kv_heads"], a["head_dim"]
    capacity = ctx.shape(kc)[0]
    t, _ = ctx.rows_of(op)
    if (
        ctx.shape(proj) != (t, (heads + 2 * kv) * d)
        or ctx.shape(kc) != (capacity, kv, d)
        or ctx.shape(vc) != ctx.shape(kc)
        or ctx.shape(out) != (t, heads * d)
        or heads % kv
        or d % 32
        or capacity <= 0
    ):
        raise ValueError("causal_attention: incompatible projection/cache/output shape")
    macros = {
        "STEP_STATE": "1",
        "D": str(d),
        "ROTARY": str(int(a["rotary"])),
        "SINKS": str(int(a["sinks"])),
        "WINDOW": str(a["window"]),
    }
    source = kernels.PRELUDE + SOURCE
    kernel = ctx.kernel("causal_attention", source, "causal_attention", macros)
    params = ctx.params(
        "causal_attention", struct.pack("<5If", heads, kv, capacity, t, 0, a["eps"])
    )
    bindings = [
        (i, *ctx.buf(v))
        for i, v in enumerate([proj, kc, vc, qn, kn, cos, sin, sinks, out])
    ]
    bindings.extend([(9, params, 0), (15, *ctx.step_state_buffer())])
    ctx.add(kernel, bindings, (t * heads, 1, 1), (32, 1, 1), op.kind, writes=[1, 2, 8])
