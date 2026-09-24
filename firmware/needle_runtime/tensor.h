// Weight access straight from the archive bytes: FP16 vectors, and Cactus
// Quant (CQ) matrices consumed without materialising them.
//
// CQ recap (export.py): each run of 128 weights along a row is rotated by the
// normalised Walsh-Hadamard matrix H, stored as an fp16 L2 norm plus b-bit
// Lloyd-Max indices of the unit direction. Reconstruction per group is
//     w = (codebook[idx] * norm) @ H
// so a dot product with an activation x is
//     w . x = norm * sum_k codebook[idx_k] * (H x)_k
// The kernel rotates the activation once per group (fwht128) and never builds
// a float weight. Matrices are stored [out, in]; rows are streamed in tiles.
#pragma once
#include <stddef.h>
#include <stdint.h>

#include "cact_reader.h"

float nr_fp16(uint16_t h);
uint16_t nr_to_fp16(float f);   // round to nearest even

// n FP16 values starting at element `first` of an FP16 tensor, as floats.
bool read_fp16(TensorReader* r, const CactRec& t, uint32_t first, uint32_t n, float* dst);
// n FP32 values.
bool read_fp32(TensorReader* r, const CactRec& t, uint32_t first, uint32_t n, float* dst);

// In-place normalised Walsh-Hadamard transform of 128 floats (x @ H).
void fwht128(float* x);

// Rotate an activation for CQ matmuls: copies n values (zero-padding to
// in_pad, a multiple of 128) and applies fwht128 to each group.
void cq_rotate(const float* x, uint32_t n, uint32_t in_pad, float* xr);

// Streaming tile buffer shared by every matmul (one per model).
struct CqScratch {
    uint8_t* buf = nullptr;
    uint32_t size = 0;
    bool init(uint32_t bytes);
    void release();
};

struct CqStats {
    uint64_t macs = 0;      // weight x activation products
    uint64_t us = 0;        // time in the matmul kernels (including reads)
};
extern CqStats g_cq;

// y[b*ldy + (r - row0)] = sum_k W[r,k] x_b[k] for r in [row0, row0+nrows), b < B.
// xr holds B rotated activations, each rec.in_pad() long.
bool cq_matmul(TensorReader* rd, const float* cb, const CactRec& t, uint32_t row0,
               uint32_t nrows, const float* xr, uint32_t B, float* y, uint32_t ldy,
               CqScratch* s);

// Dequantise one row (the embedding and engram gathers): cols() floats.
bool cq_row(TensorReader* rd, const float* cb, const CactRec& t, uint32_t row, float* out,
            CqScratch* s);
