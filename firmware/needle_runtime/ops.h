// Elementwise and small ops of the Needle 3 forward (architecture.py).
#pragma once
#include <stdint.h>

// int8 absmax fake quantisation over the whole vector (quantize.fake_quant,
// bits 8): what the reference applies before every matmul with quant=True.
void op_aq(const float* x, uint32_t n, float* out);
// Same, returning int8 codes and the scale (the KV cache and the query).
float op_q8(const float* x, uint32_t n, int8_t* codes);
// x / sqrt(mean(x^2) + 1e-6)
void op_rms_unit(const float* x, uint32_t n, float* out);
// ZCRMSNorm: (1 + scale) * rms_unit(x)
void op_zcrms(const float* x, const float* scale, uint32_t n, float* out);
float op_sigmoid(float x);
float op_silu(float x);
void op_softmax(float* x, uint32_t n);
// Sinkhorn normalisation of a 4x4 (in general n x n, n <= 8) matrix in log space.
void op_sinkhorn(float* m, uint32_t n, int iters);
// RoPE over heads of width hd (split halves), at position pos.
void op_rope(float* x, uint32_t heads, uint32_t hd, uint32_t pos, float theta);
