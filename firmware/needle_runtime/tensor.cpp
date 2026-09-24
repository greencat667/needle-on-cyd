#include "tensor.h"

#include <math.h>
#include <string.h>

#include "nr_platform.h"

CqStats g_cq;

float nr_fp16(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t man = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {  // subnormal: renormalise
            exp = 127 - 15 + 1;
            while (!(man & 0x400)) {
                man <<= 1;
                exp--;
            }
            man &= 0x3FF;
            bits = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

uint16_t nr_to_fp16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t man = x & 0x7FFFFF;
    if (((x >> 23) & 0xFF) == 0xFF) return (uint16_t)(sign | 0x7C00 | (man ? 0x200 : 0));
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    if (exp <= 0) {  // subnormal or zero
        if (exp < -10) return (uint16_t)sign;
        man |= 0x800000;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half = man >> shift;
        uint32_t rem = man & ((1u << shift) - 1);
        uint32_t mid = 1u << (shift - 1);
        if (rem > mid || (rem == mid && (half & 1))) half++;
        return (uint16_t)(sign | half);
    }
    uint32_t h = sign | ((uint32_t)exp << 10) | (man >> 13);
    uint32_t rem = man & 0x1FFF;
    if (rem > 0x1000 || (rem == 0x1000 && (h & 1))) h++;
    return (uint16_t)h;
}

bool read_fp16(TensorReader* r, const CactRec& t, uint32_t first, uint32_t n, float* dst) {
    // One read: the raw halves land in the top half of dst, then widen in
    // place front to back (dst[i] never overtakes the half at 2n + 2i).
    if (!n) return true;
    uint16_t* raw = (uint16_t*)dst + n;
    if (!r->read(t.offset + first * 2, raw, n * 2)) return false;
    for (uint32_t i = 0; i < n; i++) dst[i] = nr_fp16(raw[i]);
    return true;
}

bool read_fp32(TensorReader* r, const CactRec& t, uint32_t first, uint32_t n, float* dst) {
    return r->read(t.offset + first * 4, dst, n * 4);
}

void fwht128(float* x) {
    for (int h = 1; h < 128; h <<= 1) {
        for (int i = 0; i < 128; i += h << 1) {
            for (int j = i; j < i + h; j++) {
                float a = x[j], b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
    }
    const float s = 0.08838834764831845f;  // 1/sqrt(128)
    for (int i = 0; i < 128; i++) x[i] *= s;
}

void cq_rotate(const float* x, uint32_t n, uint32_t in_pad, float* xr) {
    memcpy(xr, x, n * sizeof(float));
    for (uint32_t i = n; i < in_pad; i++) xr[i] = 0.0f;
    for (uint32_t g = 0; g < in_pad; g += 128) fwht128(xr + g);
}

bool CqScratch::init(uint32_t bytes) {
    buf = (uint8_t*)nr_alloc(bytes, "cq tile buffer");
    size = buf ? bytes : 0;
    return buf != nullptr;
}

void CqScratch::release() {
    nr_free(buf);
    buf = nullptr;
    size = 0;
}

// Sum of codebook[idx] * xr over one 128-weight group at 2 bits. A 4 KB table
// maps each packed byte to its four codebook values, so a byte costs four
// independent multiply-adds and no branches (the Xtensa LX6 has no branch
// predictor; a per-weight switch was the slowest part of the kernel).
static float s_lut2[256][4];
static const float* s_lut2_cb = nullptr;

static void build_lut2(const float* cb) {
    for (int b = 0; b < 256; b++)
        for (int k = 0; k < 4; k++) s_lut2[b][k] = cb[(b >> (2 * k)) & 3];
    s_lut2_cb = cb;
}

static inline float group_dot2(const uint8_t* p, const float* xr, const float* cb) {
    if (s_lut2_cb != cb) build_lut2(cb);
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    for (int i = 0; i < 32; i++) {
        const float* w = s_lut2[p[i]];
        const float* x = xr + i * 4;
        a0 += w[0] * x[0];
        a1 += w[1] * x[1];
        a2 += w[2] * x[2];
        a3 += w[3] * x[3];
    }
    return (a0 + a1) + (a2 + a3);
}

// 4 bits: low nibble first; two codebook loads and two multiply-adds per byte.
static inline float group_dot4(const uint8_t* p, const float* xr, const float* cb) {
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    for (int i = 0; i < 64; i += 2) {
        const uint8_t b0 = p[i], b1 = p[i + 1];
        const float* x = xr + 2 * i;
        a0 += cb[b0 & 15] * x[0];
        a1 += cb[b0 >> 4] * x[1];
        a2 += cb[b1 & 15] * x[2];
        a3 += cb[b1 >> 4] * x[3];
    }
    return (a0 + a1) + (a2 + a3);
}

// Generic LSB-first bitstream (3-bit and anything else).
static inline uint32_t idx_at(const uint8_t* p, uint32_t k, uint32_t bits) {
    uint32_t bit = k * bits;
    uint32_t v = p[bit >> 3] | ((uint32_t)p[(bit >> 3) + 1] << 8);
    return (v >> (bit & 7)) & ((1u << bits) - 1);
}

static inline float group_dotn(const uint8_t* p, uint32_t g, const float* xr, const float* cb,
                               uint32_t bits) {
    float acc = 0;
    for (uint32_t k = 0; k < 128; k++) acc += cb[idx_at(p, g * 128 + k, bits)] * xr[k];
    return acc;
}

bool cq_matmul(TensorReader* rd, const float* cb, const CactRec& t, uint32_t row0,
               uint32_t nrows, const float* xr, uint32_t B, float* y, uint32_t ldy,
               CqScratch* s) {
    if (t.dtype != CACT_CQ || t.group != 128) return false;
    uint64_t t0 = nr_micros();
    const uint32_t prow = t.packed_row_bytes();
    const uint32_t ng = t.groups_per_row();
    const uint32_t in_pad = t.in_pad();
    const uint32_t per_row = prow + ng * 2;
    const uint32_t slack = 1024;  // sector alignment on both ends of a direct read
    if (s->size <= slack + per_row) return false;
    uint32_t tile = (s->size - slack) / per_row;
    for (uint32_t r0 = row0; r0 < row0 + nrows; r0 += tile) {
        uint32_t n = row0 + nrows - r0;
        if (n > tile) n = tile;
        // norms at the end of the scratch, the packed tile viewed into the rest
        const uint32_t nb = (n * ng * 2 + 3) & ~3u;
        uint16_t* norms = (uint16_t*)(s->buf + s->size - nb);
        if (!rd->read(t.norms_offset() + r0 * ng * 2, norms, n * ng * 2)) return false;
        const uint8_t* packed = rd->view(t.offset + r0 * prow, n * prow, s->buf, s->size - nb);
        if (!packed) return false;
        for (uint32_t i = 0; i < n; i++) {
            const uint8_t* p = packed + i * prow;
            for (uint32_t b = 0; b < B; b++) {
                const float* x = xr + b * in_pad;
                float acc = 0;
                for (uint32_t g = 0; g < ng; g++) {
                    float d;
                    if (t.bits == 2) d = group_dot2(p + g * 32, x + g * 128, cb);
                    else if (t.bits == 4) d = group_dot4(p + g * 64, x + g * 128, cb);
                    else d = group_dotn(p, g, x + g * 128, cb, t.bits);
                    acc += nr_fp16(norms[i * ng + g]) * d;
                }
                y[b * ldy + (r0 - row0) + i] = acc;
            }
        }
    }
    g_cq.macs += (uint64_t)nrows * in_pad * B;
    g_cq.us += nr_micros() - t0;
    return true;
}

bool cq_row(TensorReader* rd, const float* cb, const CactRec& t, uint32_t row, float* out,
            CqScratch* s) {
    const uint32_t prow = t.packed_row_bytes();
    const uint32_t ng = t.groups_per_row();
    if (s->size < 2 * (prow + 1024) || ng > 16) return false;
    // Gathers (embedding rows, engram rows) land anywhere in a large tensor:
    // view() reads just the covering sectors, without evicting cached blocks.
    uint16_t norms[16];
    const uint8_t* nv = rd->view(t.norms_offset() + row * ng * 2, ng * 2, s->buf + s->size / 2,
                                 s->size / 2);
    if (!nv) return false;
    memcpy(norms, nv, ng * 2);
    const uint8_t* packed = rd->view(t.offset + row * prow, prow, s->buf, s->size / 2);
    if (!packed) return false;
    float g128[128];
    const uint32_t cols = t.cols();
    for (uint32_t g = 0; g < ng; g++) {
        float nrm = nr_fp16(norms[g]);
        for (uint32_t k = 0; k < 128; k++) g128[k] = cb[idx_at(packed, g * 128 + k, t.bits)] * nrm;
        fwht128(g128);
        uint32_t n = cols - g * 128 < 128 ? cols - g * 128 : 128;
        memcpy(out + g * 128, g128, n * sizeof(float));
    }
    return true;
}
