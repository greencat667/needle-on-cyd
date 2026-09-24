#include "ops.h"

#include <math.h>

// jnp.round is round-half-to-even; rintf matches under the default FP mode.
static inline float q8(float v, float s) {
    float r = rintf(v / s);
    if (r > 127.f) r = 127.f;
    if (r < -128.f) r = -128.f;
    return r;
}

static inline float absmax_scale(const float* x, uint32_t n) {
    float m = 0;
    for (uint32_t i = 0; i < n; i++) {
        float a = fabsf(x[i]);
        if (a > m) m = a;
    }
    return m > 0 ? m / 127.0f : 1.0f;
}

void op_aq(const float* x, uint32_t n, float* out) {
    float s = absmax_scale(x, n);
    for (uint32_t i = 0; i < n; i++) out[i] = q8(x[i], s) * s;
}

float op_q8(const float* x, uint32_t n, int8_t* codes) {
    float s = absmax_scale(x, n);
    for (uint32_t i = 0; i < n; i++) codes[i] = (int8_t)q8(x[i], s);
    return s;
}

void op_rms_unit(const float* x, uint32_t n, float* out) {
    float ss = 0;
    for (uint32_t i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / (float)n + 1e-6f);
    for (uint32_t i = 0; i < n; i++) out[i] = x[i] * inv;
}

void op_zcrms(const float* x, const float* scale, uint32_t n, float* out) {
    op_rms_unit(x, n, out);
    for (uint32_t i = 0; i < n; i++) out[i] *= 1.0f + scale[i];
}

float op_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
float op_silu(float x) { return x * op_sigmoid(x); }

void op_softmax(float* x, uint32_t n) {
    float m = x[0];
    for (uint32_t i = 1; i < n; i++) if (x[i] > m) m = x[i];
    float s = 0;
    for (uint32_t i = 0; i < n; i++) {
        x[i] = expf(x[i] - m);
        s += x[i];
    }
    float inv = 1.0f / s;
    for (uint32_t i = 0; i < n; i++) x[i] *= inv;
}

void op_sinkhorn(float* m, uint32_t n, int iters) {
    for (int it = 0; it < iters; it++) {
        for (uint32_t i = 0; i < n; i++) {  // rows
            float mx = m[i * n];
            for (uint32_t j = 1; j < n; j++) if (m[i * n + j] > mx) mx = m[i * n + j];
            float s = 0;
            for (uint32_t j = 0; j < n; j++) s += expf(m[i * n + j] - mx);
            float lse = mx + logf(s);
            for (uint32_t j = 0; j < n; j++) m[i * n + j] -= lse;
        }
        for (uint32_t j = 0; j < n; j++) {  // columns
            float mx = m[j];
            for (uint32_t i = 1; i < n; i++) if (m[i * n + j] > mx) mx = m[i * n + j];
            float s = 0;
            for (uint32_t i = 0; i < n; i++) s += expf(m[i * n + j] - mx);
            float lse = mx + logf(s);
            for (uint32_t i = 0; i < n; i++) m[i * n + j] -= lse;
        }
    }
    for (uint32_t i = 0; i < n * n; i++) m[i] = expf(m[i]);
}

void op_rope(float* x, uint32_t heads, uint32_t hd, uint32_t pos, float theta) {
    uint32_t half = hd / 2;
    for (uint32_t i = 0; i < half; i++) {
        float freq = 1.0f / powf(theta, (float)(2 * i) / (float)hd);
        float ang = (float)pos * freq;
        float c = cosf(ang), s = sinf(ang);
        for (uint32_t h = 0; h < heads; h++) {
            float* v = x + h * hd;
            float x1 = v[i], x2 = v[i + half];
            v[i] = x1 * c - x2 * s;
            v[i + half] = x2 * c + x1 * s;
        }
    }
}
