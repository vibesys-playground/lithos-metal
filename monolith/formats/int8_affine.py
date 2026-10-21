"""Unsigned affine INT8, group 64: ``w = scale * code + bias``.

MLX stores four low-byte-first codes per uint32, plus F16/BF16 scales and
biases. Shares the affine format's exact pair preservation and BLM layout;
The shared embedding gather refuses sub-word block packs (K=256); inline
packing supports that width. This differs from symmetric signed ``int8`` in both zero point and grouping.
"""

from .int4_affine import GROUP, INT4Affine
from .registry import register_format


@register_format("int8_affine")
class INT8Affine(INT4Affine):
    bits = 8
    bytes_per_weight = 1.0 + 4.0 / GROUP
    weights_per_word = 16
    msl_decode = """
#define WEIGHTS_PER_WORD 16u
#define SCALE_GROUP 64u
#define SCALE_BIAS 1
#ifndef SCALE_F16
#define SCALE_F16 0
#endif
static inline void decode_word(uint4 q, thread float* out) {
  uint w[4] = {q.x, q.y, q.z, q.w};
  for (uint e = 0; e < 16; e++) out[e] = float((w[e >> 2] >> ((e & 3u) * 8u)) & 0xFFu);
}
#if SCALE_F16
static inline float decode_scale(thread const uint* sw, uint g) { return float(as_type<half>(ushort(sw[g] & 0xFFFFu))); }
static inline float decode_bias(thread const uint* sw, uint g) { return float(as_type<half>(ushort(sw[g] >> 16))); }
#else
static inline float decode_scale(thread const uint* sw, uint g) { return as_type<float>(sw[g] << 16); }
static inline float decode_bias(thread const uint* sw, uint g) { return as_type<float>(sw[g] & 0xFFFF0000u); }
#endif
"""
