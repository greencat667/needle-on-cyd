#include "model.h"

#include <math.h>
#include <string.h>

#include "nr_platform.h"
#include "ops.h"

// Per-layer tensor positions (export._tensors, qkv_conv_taps > 0).
enum {
    T_NORM_IN, T_Q, T_K, T_V, T_QTAPS, T_KTAPS, T_VTAPS, T_QNORM, T_KNORM, T_GATE, T_OUT,
    T_POST_NORM, T_ATTN_GATE, T_PRE_HADA, T_D1, T_D2, T_B2, T_D3, T_D4, T_W1A, T_W1B, T_W2A,
    T_W2B, T_W3A, T_W3B, T_COND_V, T_COND_U, T_PER_LAYER
};

static const uint32_t SEED = 0x9E3779B9u, PRIME = 0x01000193u;
static float s_w[768];  // shared FP16 staging: one storage read per vector

// The probe-pool accumulator may sit in the ESP32's spare IRAM, which only
// takes 32-bit integer loads and stores (an FPU lsi/ssi there faults), so it
// is always touched through these.
static inline float ld32(const float* p) {
    uint32_t u = *(const volatile uint32_t*)p;
    float f;
    memcpy(&f, &u, 4);
    return f;
}
static inline void st32(float* p, float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    *(volatile uint32_t*)p = u;
}

// ---------------------------------------------------------------------------
bool ProbePool::init(uint32_t rows_, uint32_t k_, uint32_t D_) {
    rows = rows_; k = k_; D = D_;
    m = (float*)nr_alloc(rows * k * sizeof(float), "probe pool max");
    s = (float*)nr_alloc(rows * k * sizeof(float), "probe pool sum");
    acc = (float*)nr_alloc_fast32(rows * k * D * sizeof(float), "probe pool acc");
    if (!m || !s || !acc) return false;
    reset();
    return true;
}
void ProbePool::reset() {
    for (uint32_t i = 0; i < rows * k; i++) { m[i] = -INFINITY; s[i] = 0; }
    for (uint32_t i = 0; i < rows * k * D; i++) st32(acc + i, 0.0f);
}
void ProbePool::release() {
    nr_free(m); nr_free(s); nr_free(acc);
    m = s = acc = nullptr;
}

// ---------------------------------------------------------------------------
static float* falloc(uint32_t n, const char* tag) { return (float*)nr_alloc(n * sizeof(float), tag); }

bool Model::load(TensorReader* reader, uint32_t cap) {
    rd = reader;
    if (!cact.parse(rd)) return false;
    const CactHeader& h = cact.h;
    L = h.num_layers; D = h.d_model; H = h.num_heads; KVH = h.num_kv_heads;
    QK = h.qk_head_dim; VD = h.v_head_dim; lanes = h.mhc_lanes; taps = h.qkv_conv_taps;
    hada_n = h.hada_n; vocab = h.vocab; out_vocab = h.out_vocab ? h.out_vocab : h.vocab;
    n_sites = h.num_engram_sites; n_tables = h.num_engram_tables; slots = h.engram_slots;
    sub = h.engram_sub_dim; etaps = h.engram_conv_taps; edil = h.engram_conv_dilation;
    n_orders = h.num_engram_orders;
    window = h.sliding_window;
    if (L > NR_MAX_LAYERS || n_sites > NR_MAX_SITES || lanes > NR_MAX_LANES || taps == 0 ||
        n_tables * sub != D || hada_n != 1024) {
        nr_log("model: unsupported geometry\n");
        return false;
    }
    i_emb = 0;
    i_layer0 = 1;
    i_mhc = i_layer0 + T_PER_LAYER * L;
    i_perm = i_mhc + 9;
    i_engram0 = i_perm + 2;
    i_final = i_engram0 + 4 * n_sites;
    uint32_t tail = cact.n_recs - (i_final + 1) - 1;  // probe-head tensors incl. manifest
    i_heads = tail ? i_final + 1 : 0;
    has_confidence = false;
    if (tail) {
        float codes[4];
        uint32_t nh = rec(i_heads).shape[0];
        read_fp16(rd, rec(i_heads), 0, nh, codes);
        uint32_t i = i_heads + 1;
        for (uint32_t c = 0; c < nh; c++) {
            if ((int)codes[c] == 2) { i_conf = i; has_confidence = true; }
            i += 6 + ((int)codes[c] == 3 ? 1 : 0);
        }
    }
    if (rec(cact.n_recs - 1).dtype != CACT_RAW || rec(i_emb).dtype != CACT_CQ ||
        lrec(0, T_Q).dtype != CACT_CQ || lrec(0, T_COND_U).dtype != CACT_FP16) {
        nr_log("model: tensor order does not match tag 0x05E12A84\n");
        return false;
    }
    ctx_cap = cap;

    // tile buffer: enough for a good run of rows of the widest matrix (3072 in)
    if (!scratch.init(8192)) return false;

    // mHC small parameters are resident: a few hundred floats.
    a_pre = falloc(L, "mhc"); a_post = falloc(L, "mhc"); a_res = falloc(L, "mhc");
    b_pre = falloc(L * lanes, "mhc"); b_post = falloc(L * lanes, "mhc");
    b_res = falloc(L * lanes * lanes, "mhc");
    if (!a_pre || !a_post || !a_res || !b_pre || !b_post || !b_res) return false;
    read_fp16(rd, rec(i_mhc + 0), 0, L, a_pre);
    read_fp16(rd, rec(i_mhc + 1), 0, L, a_post);
    read_fp16(rd, rec(i_mhc + 2), 0, L, a_res);
    read_fp16(rd, rec(i_mhc + 3), 0, L * lanes, b_pre);
    read_fp16(rd, rec(i_mhc + 4), 0, L * lanes, b_post);
    read_fp16(rd, rec(i_mhc + 5), 0, L * lanes * lanes, b_res);

    // Hadamard-MLP permutations: FP32 index vectors in the file, u16 here.
    perm1 = (uint16_t*)nr_alloc(hada_n * 2, "hada perms");
    perm2 = (uint16_t*)nr_alloc(hada_n * 2, "hada perms");
    if (!perm1 || !perm2) return false;
    float tmp[64];
    for (uint32_t i = 0; i < hada_n; i += 64) {
        read_fp32(rd, rec(i_perm), i, 64, tmp);
        for (int j = 0; j < 64; j++) perm1[i + j] = (uint16_t)tmp[j];
        read_fp32(rd, rec(i_perm + 1), i, 64, tmp);
        for (int j = 0; j < 64; j++) perm2[i + j] = (uint16_t)tmp[j];
    }

    // per-position state
    const uint32_t qkv_n = (H + KVH) * QK + KVH * VD;
    for (uint32_t l = 0; l < L; l++) {
        kc[l] = (int8_t*)nr_alloc(ctx_cap * KVH * QK, "kv cache K");
        vc[l] = (int8_t*)nr_alloc(ctx_cap * KVH * VD, "kv cache V");
        ks[l] = falloc(ctx_cap * KVH, "kv scales");
        vs[l] = falloc(ctx_cap * KVH, "kv scales");
        qkv_hist[l] = falloc((taps - 1) * qkv_n, "qkv conv history");
        if (!kc[l] || !vc[l] || !ks[l] || !vs[l] || !qkv_hist[l]) return false;
    }
    ehist = (etaps - 1) * edil + 1;
    for (uint32_t s = 0; s < n_sites; s++) {
        ev_hist[s] = (uint16_t*)nr_alloc(ehist * D * 2, "engram value history (fp16)");
        if (!ev_hist[s]) return false;
    }
    if (has_confidence) {
        const CactRec& gain = rec(i_conf + 1);
        if (!pool.init(gain.shape[0], gain.shape[1], D)) return false;
    }

    // work buffers
    const uint32_t C = NR_MAX_CHUNK, W = lanes * D;
    X = falloc(C * W, "residual lanes");
    xr = falloc(C * W, "rotated activation");
    u0 = falloc(C * D, "block io");
    u = falloc(C * D, "block io");
    hb = falloc(C * D, "block io");
    qkv = falloc(C * qkv_n, "qkv");
    gate = falloc(C * H * VD, "attn gate");
    att = falloc(C * H * VD, "attn out");
    ek = falloc((n_sites ? n_sites : 1) * C * D, "engram keys");
    ev = falloc((n_sites ? n_sites : 1) * C * D, "engram values");
    mix = falloc(C * (2 * lanes + lanes * lanes), "mhc mix");
    scores = falloc(ctx_cap, "attn scores");
    mlp = falloc(3 * hada_n, "hada mlp");
    pbuf = falloc(hada_n, "param stream");
    if (!X || !xr || !u0 || !u || !hb || !qkv || !gate || !att || !ek || !ev || !mix ||
        !scores || !mlp || !pbuf)
        return false;
    reset();
    return true;
}

void Model::release() {
    for (uint32_t l = 0; l < NR_MAX_LAYERS; l++) {
        nr_free(kc[l]); nr_free(vc[l]); nr_free(ks[l]); nr_free(vs[l]); nr_free(qkv_hist[l]);
        kc[l] = vc[l] = nullptr; ks[l] = vs[l] = nullptr; qkv_hist[l] = nullptr;
    }
    for (uint32_t s = 0; s < NR_MAX_SITES; s++) { nr_free(ev_hist[s]); ev_hist[s] = nullptr; }
    pool.release();
    float** bufs[] = {&X, &xr, &u0, &u, &hb, &qkv, &gate, &att, &ek, &ev, &mix, &scores,
                      &mlp, &pbuf, &a_pre, &a_post, &a_res, &b_pre, &b_post, &b_res};
    for (float** b : bufs) { nr_free(*b); *b = nullptr; }
    nr_free(perm1); nr_free(perm2); perm1 = perm2 = nullptr;
    scratch.release();
    cact.release();
}

void Model::reset() {
    pos = 0;
    memset(tok_hist, 0, sizeof(tok_hist));
    if (has_confidence) pool.reset();
}

// ---------------------------------------------------------------------------
bool Model::matmul(const CactRec& t, const float* x, uint32_t n, float* y, uint32_t ldy,
                   uint32_t row0, uint32_t rows) {
    if (!rows) rows = t.rows();
    return cq_matmul(rd, cact.cb(t.bits), t, row0, rows, x, n, y, ldy, &scratch);
}

float* Model::layer_param(uint32_t l, uint32_t k, uint32_t first, uint32_t count, float* dst) {
    read_fp16(rd, lrec(l, k), first, count, dst);
    return dst;
}

// Streamed FP16 helpers: y[i] op= w[first + i], in 64-element pieces.
// in 512-element pieces through the staging buffer
static bool fp16_mul(TensorReader* r, const CactRec& t, uint32_t first, float* y, uint32_t n) {
    for (uint32_t i = 0; i < n; i += 512) {
        uint32_t k = n - i < 512 ? n - i : 512;
        if (!read_fp16(r, t, first + i, k, s_w)) return false;
        for (uint32_t j = 0; j < k; j++) y[i + j] *= s_w[j];
    }
    return true;
}
static bool fp16_add(TensorReader* r, const CactRec& t, uint32_t first, float* y, uint32_t n,
                     float alpha) {
    for (uint32_t i = 0; i < n; i += 512) {
        uint32_t k = n - i < 512 ? n - i : 512;
        if (!read_fp16(r, t, first + i, k, s_w)) return false;
        for (uint32_t j = 0; j < k; j++) y[i + j] += alpha * s_w[j];
    }
    return true;
}

// ---------------------------------------------------------------------------
// Engram: n-gram hashes of the token history index per-site tables; the
// fetched rows project to a key (gates the injection) and a value (dilated
// causal conv over positions). Hashes do not depend on the site.
bool Model::engram(const uint16_t* toks, uint32_t n) {
    if (!n_sites) return true;
    const uint32_t heads = n_tables / n_orders;
    const uint32_t stride = cact.h.engram_seed_heads ? cact.h.engram_seed_heads : heads;
    uint32_t idx[16];
    float* e = pbuf;  // D floats
    for (uint32_t s = 0; s < n_sites; s++) {
        const CactRec& tables = rec(i_engram0 + 4 * s);
        uint16_t hist[4];
        memcpy(hist, tok_hist, sizeof(hist));
        for (uint32_t b = 0; b < n; b++) {
            const uint32_t t = pos + b;
            // newest-first history including this token
            for (int j = 3; j > 0; j--) hist[j] = hist[j - 1];
            hist[0] = toks[b];
            uint32_t ti = 0;
            for (uint32_t oi = 0; oi < n_orders; oi++) {
                const uint32_t order = cact.h.engram_orders[oi];
                for (uint32_t hh = 0; hh < heads; hh++, ti++) {
                    uint32_t acc = SEED * (oi * stride + hh + 1);
                    for (uint32_t j = 0; j < order; j++) {
                        uint32_t tj = t >= j ? hist[j] : 0;
                        acc = (acc ^ tj) * PRIME;
                    }
                    acc ^= acc >> 15;
                    idx[ti] = acc % slots;
                    float* row = e + ti * sub;
                    if (t + 1 >= order) {
                        if (!cq_row(rd, cact.cb(tables.bits), tables, ti * slots + idx[ti], row,
                                    &scratch))
                            return false;
                    } else {
                        memset(row, 0, sub * sizeof(float));
                    }
                }
            }
            op_aq(e, D, e);
            cq_rotate(e, D, D, xr + b * D);
        }
        float* k = ek + s * NR_MAX_CHUNK * D;
        float* v = ev + s * NR_MAX_CHUNK * D;
        if (!matmul(rec(i_engram0 + 4 * s + 1), xr, n, k, D)) return false;
        if (!matmul(rec(i_engram0 + 4 * s + 2), xr, n, v, D)) return false;
        // dilated causal conv over the raw values
        const CactRec& tp = rec(i_engram0 + 4 * s + 3);
        for (uint32_t b = 0; b < n; b++) {
            const uint32_t t = pos + b;
            float* vb = v + b * D;
            // past values come from the fp16 ring; this position's is exact
            uint16_t* slot = ev_hist[s] + (t % ehist) * D;
            float* cur = hb;  // D scratch
            memcpy(cur, vb, D * sizeof(float));
            for (uint32_t d = 0; d < D; d++) slot[d] = nr_to_fp16(cur[d]);
            memset(vb, 0, D * sizeof(float));
            for (uint32_t j = 0; j < etaps; j++) {
                if (t < j * edil) break;
                const uint16_t* src = ev_hist[s] + ((t - j * edil) % ehist) * D;
                float* w = s_w;
                read_fp16(rd, tp, j * D, D, w);
                if (j == 0)
                    for (uint32_t q = 0; q < D; q++) vb[q] += w[q] * cur[q];
                else
                    for (uint32_t q = 0; q < D; q++) vb[q] += w[q] * nr_fp16(src[q]);
            }
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
bool Model::attention(uint32_t l, uint32_t n) {
    const uint32_t Qn = H * QK, Kn = KVH * QK, Vn = KVH * VD, qkv_n = Qn + Kn + Vn;
    const uint32_t rep = H / KVH;
    const float inv_sqrt = 1.0f / sqrtf((float)QK);
    // file-static work arrays keep the task stack small on the ESP32
    static float qn[48 * 16];  // per-head normalised q (H*QK <= 768)
    static float kn[48 * 4];
    static float vv[64 * 4];
    static float qs_[16];
    static int8_t qq[48 * 16];
    float* norm_q = pbuf;       // QK
    float* norm_k = pbuf + 64;  // QK
    layer_param(l, T_QNORM, 0, QK, norm_q);
    layer_param(l, T_KNORM, 0, QK, norm_k);
    const bool global = cact.is_global(l);
    for (uint32_t b = 0; b < n; b++) {
        const uint32_t t = pos + b;
        float* raw = qkv + b * qkv_n;

        // causal depthwise conv over q, k, v
        static float conv[48 * 16 + 48 * 4 + 64 * 4];
        memset(conv, 0, qkv_n * sizeof(float));
        for (uint32_t j = 0; j < taps && j <= t; j++) {
            // ring of the taps-1 previous positions' raw projections
            const float* src = j == 0 ? raw : qkv_hist[l] + ((t - j) % (taps - 1)) * qkv_n;
            float* w = s_w;
            read_fp16(rd, lrec(l, T_QTAPS), j * Qn, Qn, w);
            for (uint32_t q = 0; q < Qn; q++) conv[q] += w[q] * src[q];
            read_fp16(rd, lrec(l, T_KTAPS), j * Kn, Kn, w);
            for (uint32_t q = 0; q < Kn; q++) conv[Qn + q] += w[q] * src[Qn + q];
            read_fp16(rd, lrec(l, T_VTAPS), j * Vn, Vn, w);
            for (uint32_t q = 0; q < Vn; q++) conv[Qn + Kn + q] += w[q] * src[Qn + Kn + q];
        }
        memcpy(qkv_hist[l] + (t % (taps - 1)) * qkv_n, raw, qkv_n * sizeof(float));
        for (uint32_t h = 0; h < H; h++) op_zcrms(conv + h * QK, norm_q, QK, qn + h * QK);
        for (uint32_t h = 0; h < KVH; h++) op_zcrms(conv + Qn + h * QK, norm_k, QK, kn + h * QK);
        memcpy(vv, conv + Qn + Kn, Vn * sizeof(float));
        op_rope(qn, H, QK, t, cact.h.rope_theta);
        op_rope(kn, KVH, QK, t, cact.h.rope_theta);
        for (uint32_t h = 0; h < H; h++) qs_[h] = op_q8(qn + h * QK, QK, qq + h * QK);
        for (uint32_t h = 0; h < KVH; h++) {
            ks[l][t * KVH + h] = op_q8(kn + h * QK, QK, kc[l] + (t * KVH + h) * QK);
            vs[l][t * KVH + h] = op_q8(vv + h * VD, VD, vc[l] + (t * KVH + h) * VD);
        }
        uint32_t lo = (global || !window || t + 1 < window) ? 0 : t + 1 - window;
        float* out = att + b * H * VD;
        for (uint32_t h = 0; h < H; h++) {
            const uint32_t g = h / rep;
            float qf[48];
            for (uint32_t d = 0; d < QK; d++) qf[d] = qq[h * QK + d] * qs_[h];
            const uint32_t np = t + 1 - lo;
            for (uint32_t p = 0; p < np; p++) {
                const int8_t* kr = kc[l] + ((lo + p) * KVH + g) * QK;
                float dot = 0;
                for (uint32_t d = 0; d < QK; d++) dot += (float)kr[d] * qf[d];
                scores[p] = dot * ks[l][(lo + p) * KVH + g] * inv_sqrt;
            }
            op_softmax(scores, np);
            float* o = out + h * VD;
            memset(o, 0, VD * sizeof(float));
            for (uint32_t p = 0; p < np; p++) {
                const int8_t* vr = vc[l] + ((lo + p) * KVH + g) * VD;
                const float w = scores[p] * vs[l][(lo + p) * KVH + g];
                for (uint32_t d = 0; d < VD; d++) o[d] += w * (float)vr[d];
            }
        }
        const float* gt = gate + b * H * VD;
        for (uint32_t i = 0; i < H * VD; i++) out[i] *= op_sigmoid(gt[i]);
    }
    return true;
}

// ---------------------------------------------------------------------------
// out[k][l] = sum_ij Z[i][j] a[i][k] b[j][l]   (architecture._kron_apply)
// The two Monarch factors stay as the archive's raw FP16 (2 KB each).
static void kron_apply(float* z, const uint16_t* a, const uint16_t* bm, float* tmp, uint32_t ba,
                       uint32_t bb) {
    float col[64];
    for (uint32_t l = 0; l < bb; l++) {
        for (uint32_t j = 0; j < bb; j++) col[j] = nr_fp16(bm[j * bb + l]);
        for (uint32_t i = 0; i < ba; i++) {
            float s = 0;
            for (uint32_t j = 0; j < bb; j++) s += z[i * bb + j] * col[j];
            tmp[i * bb + l] = s;
        }
    }
    for (uint32_t k = 0; k < ba; k++) {
        for (uint32_t i = 0; i < ba; i++) col[i] = nr_fp16(a[i * ba + k]);
        for (uint32_t l = 0; l < bb; l++) {
            float s = 0;
            for (uint32_t i = 0; i < ba; i++) s += col[i] * tmp[i * bb + l];
            z[k * bb + l] = s;
        }
    }
}

static bool read_raw16(TensorReader* r, const CactRec& t, uint32_t n, uint16_t* dst) {
    return r->read(t.offset, dst, n * 2);
}

bool Model::hada_mlp(uint32_t l, float* x) {
    const uint32_t n = hada_n;
    const uint32_t ba = lrec(l, T_W1A).shape[0], bb = lrec(l, T_W1B).shape[0];
    float* z = mlp;
    float* cond = mlp + n;
    uint16_t* fa = (uint16_t*)(mlp + 2 * n);   // ba*ba <= n raw FP16
    uint16_t* fb = fa + n;                      // bb*bb
    float* tmp = pbuf;         // n floats
    // cond = 1 + softmax(x @ cond_v) @ cond_u
    const CactRec& cv = lrec(l, T_COND_V);
    const uint32_t R = cv.shape[1];
    float c8[16] = {0};
    for (uint32_t i = 0; i < D * R; i += 768) {
        read_fp16(rd, cv, i, 768, s_w);
        for (uint32_t q = 0; q < 768; q++) c8[(i + q) % R] += x[(i + q) / R] * s_w[q];
    }
    op_softmax(c8, R);
    for (uint32_t i = 0; i < n; i++) cond[i] = 1.0f;
    for (uint32_t r = 0; r < R; r++) fp16_add(rd, lrec(l, T_COND_U), r * n, cond, n, c8[r]);

    memcpy(z, x, D * sizeof(float));
    for (uint32_t i = D; i < n; i++) z[i] = 0;
    // stage 1
    fp16_mul(rd, lrec(l, T_D1), 0, z, n);
    read_raw16(rd, lrec(l, T_W1A), ba * ba, fa);
    read_raw16(rd, lrec(l, T_W1B), bb * bb, fb);
    kron_apply(z, fa, fb, tmp, ba, bb);
    for (uint32_t i = 0; i < n; i++) tmp[i] = z[perm1[i]];
    memcpy(z, tmp, n * sizeof(float));
    // stage 2
    fp16_mul(rd, lrec(l, T_D2), 0, z, n);
    for (uint32_t i = 0; i < n; i++) z[i] *= cond[i];
    fp16_add(rd, lrec(l, T_B2), 0, z, n, 1.0f);
    for (uint32_t i = 0; i < n; i++) z[i] = op_silu(z[i]);
    read_raw16(rd, lrec(l, T_W2A), ba * ba, fa);
    read_raw16(rd, lrec(l, T_W2B), bb * bb, fb);
    kron_apply(z, fa, fb, tmp, ba, bb);
    for (uint32_t i = 0; i < n; i++) tmp[i] = z[perm2[i]];
    memcpy(z, tmp, n * sizeof(float));
    // stage 3
    fp16_mul(rd, lrec(l, T_D3), 0, z, n);
    read_raw16(rd, lrec(l, T_W3A), ba * ba, fa);
    read_raw16(rd, lrec(l, T_W3B), bb * bb, fb);
    kron_apply(z, fa, fb, tmp, ba, bb);
    fp16_mul(rd, lrec(l, T_D4), 0, z, n);
    memcpy(x, z, D * sizeof(float));
    return true;
}

// ---------------------------------------------------------------------------
bool Model::pool_update(uint32_t row, const float* cell) {
    if (!has_confidence) return true;
    const CactRec& probes = rec(i_conf);
    float* cr = mlp;  // D rotated
    cq_rotate(cell, D, probes.in_pad(), cr);
    float sc[8];
    const uint32_t k = pool.k;
    if (!matmul(probes, cr, 1, sc, k, row * k, k)) return false;
    const float inv = 1.0f / sqrtf((float)D);
    for (uint32_t j = 0; j < k; j++) {
        const uint32_t i = row * k + j;
        const float v = sc[j] * inv;
        const float mn = v > pool.m[i] ? v : pool.m[i];
        const float old = pool.m[i] == -INFINITY ? 0.0f : expf(pool.m[i] - mn);
        const float wv = expf(v - mn);
        pool.s[i] = pool.s[i] * old + wv;
        float* a = pool.acc + i * D;
        for (uint32_t d = 0; d < D; d++) st32(a + d, ld32(a + d) * old + wv * cell[d]);
        pool.m[i] = mn;
    }
    return true;
}

bool Model::confidence_logit(float* out) {
    if (!has_confidence || pos == 0) return false;
    const CactRec& gain = rec(i_conf + 1);
    const CactRec& query = rec(i_conf + 2);
    const CactRec& row_bias = rec(i_conf + 3);
    const CactRec& proj = rec(i_conf + 4);
    const CactRec& bias = rec(i_conf + 5);
    const uint32_t R = pool.rows * pool.k, Q = query.shape[0];
    const float inv = 1.0f / sqrtf((float)D);
    float* r = pbuf;          // D
    float* rr = mlp;          // D rotated
    float* uq = mlp + 1024;   // Q*R scores (<= 4*84)
    float* pooled = X;        // Q*D
    float* pr = xr;           // Q*D rotated
    if (Q * R > 1024 || Q * D > NR_MAX_CHUNK * lanes * D) return false;
    // r_i = rms_unit(acc_i / s_i) * gain_i, recomputed on demand (no R x D copy)
    auto make_r = [&](uint32_t i) {
        const float* a = pool.acc + i * D;
        const float is = 1.0f / pool.s[i];
        for (uint32_t d = 0; d < D; d++) r[d] = ld32(a + d) * is;
        op_rms_unit(r, D, r);
        float g;
        read_fp16(rd, gain, i, 1, &g);
        for (uint32_t d = 0; d < D; d++) r[d] *= g;
    };
    float qd[8];
    for (uint32_t i = 0; i < R; i++) {
        make_r(i);
        cq_rotate(r, D, query.in_pad(), rr);
        if (!matmul(query, rr, 1, qd, Q)) return false;
        for (uint32_t q = 0; q < Q; q++) {
            float rb;
            read_fp16(rd, row_bias, q * R + i, 1, &rb);
            uq[q * R + i] = qd[q] * inv + rb;
        }
    }
    for (uint32_t q = 0; q < Q; q++) op_softmax(uq + q * R, R);
    memset(pooled, 0, Q * D * sizeof(float));
    for (uint32_t i = 0; i < R; i++) {
        make_r(i);
        for (uint32_t q = 0; q < Q; q++) {
            const float wq = uq[q * R + i];
            for (uint32_t d = 0; d < D; d++) pooled[q * D + d] += wq * r[d];
        }
    }
    cq_rotate(pooled, Q * D, proj.in_pad(), pr);
    float logit;
    if (!matmul(proj, pr, 1, &logit, 1)) return false;
    float bv;
    read_fp16(rd, bias, 0, 1, &bv);
    *out = logit + bv;
    return true;
}

// ---------------------------------------------------------------------------
bool Model::forward(const uint16_t* toks, uint32_t n, float* h_last) {
    if (n == 0 || n > NR_MAX_CHUNK) return false;
    if (pos + n > ctx_cap) {
        nr_log("model: context full (%u + %u > %u)\n", (unsigned)pos, (unsigned)n,
               (unsigned)ctx_cap);
        return false;
    }
    uint64_t t0 = nr_micros();
    const uint32_t W = lanes * D;
    const uint32_t qkv_n = (H + KVH) * QK + KVH * VD;
    const uint32_t MIX = 2 * lanes + lanes * lanes;
    const float esc = sqrtf((float)D);

    // embedding gather -> all lanes; cell row 0
    for (uint32_t b = 0; b < n; b++) {
        float* x0 = u + b * D;
        if (!cq_row(rd, cact.cb(rec(i_emb).bits), rec(i_emb), toks[b], x0, &scratch)) return false;
        for (uint32_t d = 0; d < D; d++) x0[d] *= esc;
        for (uint32_t k = 0; k < lanes; k++) memcpy(X + b * W + k * D, x0, D * sizeof(float));
    }
    { uint64_t _t = nr_micros(); bool _ok = engram(toks, n); stats.engram_us += nr_micros() - _t; if (!_ok) return false; }
    for (uint32_t b = 0; b < n; b++) {
        // pool_update is per position; row 0 is the input embedding
        if (!pool_update(0, u + b * D)) return false;
    }

    for (uint32_t l = 0; l < L; l++) {
        const int site = cact.engram_site_of(l);
        // mHC: read gates from the normalised lanes
        for (uint32_t b = 0; b < n; b++) {
            float* nx = xr + b * W;
            op_rms_unit(X + b * W, W, nx);
            op_aq(nx, W, nx);
            for (uint32_t g = 0; g < W; g += 128) fwht128(nx + g);
        }
        const CactRec& ppre = rec(i_mhc + 6);
        const CactRec& ppost = rec(i_mhc + 7);
        const CactRec& pres = rec(i_mhc + 8);
        if (!matmul(ppre, xr, n, mix, MIX, l * lanes, lanes)) return false;
        if (!matmul(ppost, xr, n, mix + lanes, MIX, l * lanes, lanes)) return false;
        if (!matmul(pres, xr, n, mix + 2 * lanes, MIX, l * lanes * lanes, lanes * lanes))
            return false;
        const uint32_t lane = l % lanes;
        for (uint32_t b = 0; b < n; b++) {
            float* m = mix + b * MIX;
            float hpre[NR_MAX_LANES];
            for (uint32_t i = 0; i < lanes; i++) {
                const float off = i == lane ? 4.0f : -4.0f;
                hpre[i] = op_sigmoid(a_pre[l] * m[i] + b_pre[l * lanes + i] + off);
                const float poff = i == lane ? 0.0f : -4.0f;
                m[i] = 2.0f * op_sigmoid(a_post[l] * m[lanes + i] + b_post[l * lanes + i] + poff);
            }
            float* res = m + 2 * lanes;
            for (uint32_t i = 0; i < lanes * lanes; i++)
                res[i] = a_res[l] * res[i] + b_res[l * lanes * lanes + i];
            op_sinkhorn(res, lanes, 20);
            float* ub = u0 + b * D;
            memset(ub, 0, D * sizeof(float));
            for (uint32_t i = 0; i < lanes; i++) {
                const float* xi = X + b * W + i * D;
                for (uint32_t d = 0; d < D; d++) ub[d] += hpre[i] * xi[d];
            }
            // m[0..lanes) now holds hpost, m[2*lanes..) hres
        }

        // ---- block: engram injection, attention ----
        float* nin = pbuf;
        layer_param(l, T_NORM_IN, 0, D, nin);
        for (uint32_t b = 0; b < n; b++) {
            float* ub = u + b * D;
            memcpy(ub, u0 + b * D, D * sizeof(float));
            if (site >= 0) {
                const float* k = ek + site * NR_MAX_CHUNK * D + b * D;
                const float* v = ev + site * NR_MAX_CHUNK * D + b * D;
                float* ru = hb + b * D;
                op_rms_unit(ub, D, ru);
                float nk = 0, dot = 0;
                for (uint32_t d = 0; d < D; d++) nk += k[d] * k[d];
                const float ik = 1.0f / sqrtf(nk / (float)D + 1e-6f);
                for (uint32_t d = 0; d < D; d++) dot += ru[d] * k[d] * ik;
                const float alpha = op_sigmoid(dot / sqrtf((float)D));
                for (uint32_t d = 0; d < D; d++) ub[d] += alpha * v[d];
            }
            float* hn = hb + b * D;
            op_zcrms(ub, nin, D, hn);
            op_aq(hn, D, hn);
            cq_rotate(hn, D, D, xr + b * D);
        }
        const uint32_t Qn = H * QK, Kn = KVH * QK;
        if (!matmul(lrec(l, T_Q), xr, n, qkv, qkv_n)) return false;
        if (!matmul(lrec(l, T_K), xr, n, qkv + Qn, qkv_n)) return false;
        if (!matmul(lrec(l, T_V), xr, n, qkv + Qn + Kn, qkv_n)) return false;
        if (!matmul(lrec(l, T_GATE), xr, n, gate, H * VD)) return false;
        { uint64_t _t = nr_micros(); bool _ok = attention(l, n); stats.attn_us += nr_micros() - _t; if (!_ok) return false; }
        for (uint32_t b = 0; b < n; b++) {
            float* a = att + b * H * VD;
            op_aq(a, H * VD, a);
            cq_rotate(a, H * VD, lrec(l, T_OUT).in_pad(), xr + b * lrec(l, T_OUT).in_pad());
        }
        if (!matmul(lrec(l, T_OUT), xr, n, hb, D)) return false;

        // ---- block: post-attention, Hadamard MLP ----
        float gate_raw;
        layer_param(l, T_ATTN_GATE, 0, 1, &gate_raw);
        const float ag = op_sigmoid(gate_raw);
        for (uint32_t b = 0; b < n; b++) {
            float* ub = u + b * D;
            float* ho = hb + b * D;
            layer_param(l, T_POST_NORM, 0, D, pbuf);
            op_zcrms(ho, pbuf, D, ho);
            for (uint32_t d = 0; d < D; d++) ub[d] += ag * ho[d];
            layer_param(l, T_PRE_HADA, 0, D, pbuf);
            op_zcrms(ub, pbuf, D, ho);
            { uint64_t _t = nr_micros(); bool _ok = hada_mlp(l, ho); stats.mlp_us += nr_micros() - _t; if (!_ok) return false; }
            // y = block(u0) - u0
            const float* u0b = u0 + b * D;
            for (uint32_t d = 0; d < D; d++) ho[d] = ub[d] + ho[d] - u0b[d];
            // mHC combine: X = hres @ X + hpost * y
            const float* m = mix + b * MIX;
            const float* hres = m + 2 * lanes;
            float* Xb = X + b * W;
            for (uint32_t d = 0; d < D; d++) {
                float col[NR_MAX_LANES];
                for (uint32_t j = 0; j < lanes; j++) col[j] = Xb[j * D + d];
                for (uint32_t i = 0; i < lanes; i++) {
                    float s = m[i] * ho[d];
                    for (uint32_t j = 0; j < lanes; j++) s += hres[i * lanes + j] * col[j];
                    Xb[i * D + d] = s;
                }
            }
            // hidden cell = lane mean
            for (uint32_t d = 0; d < D; d++) {
                float s = 0;
                for (uint32_t i = 0; i < lanes; i++) s += Xb[i * D + d];
                ho[d] = s / (float)lanes;
            }
            if (!pool_update(l + 1, ho)) return false;
        }
    }

    // token history and position advance
    for (uint32_t b = 0; b < n; b++) {
        for (int j = 3; j > 0; j--) tok_hist[j] = tok_hist[j - 1];
        tok_hist[0] = toks[b];
    }
    pos += n;
    if (h_last) {
        const float* Xb = X + (n - 1) * W;
        for (uint32_t d = 0; d < D; d++) {
            float s = 0;
            for (uint32_t i = 0; i < lanes; i++) s += Xb[i * D + d];
            h_last[d] = s / (float)lanes;
        }
        read_fp16(rd, rec(i_final), 0, D, pbuf);
        op_zcrms(h_last, pbuf, D, h_last);
    }
    stats.tokens += n;
    stats.chunks++;
    stats.forward_us += nr_micros() - t0;
    return true;
}

// ---------------------------------------------------------------------------
bool Model::logits(const float* h, LogitSink* sink) {
    uint64_t t0 = nr_micros();
    const CactRec& e = rec(i_emb);
    float* hr = mlp;
    op_aq(h, D, pbuf);
    cq_rotate(pbuf, D, e.in_pad(), hr);
    float* out = mlp + 1024;  // tile of logits
    const uint32_t T = 256;
    for (uint32_t r = 0; r < out_vocab; r += T) {
        uint32_t k = out_vocab - r < T ? out_vocab - r : T;
        if (!matmul(e, hr, 1, out, k, r, k)) return false;
        sink->consume(r, out, k);
    }
    stats.logits_us += nr_micros() - t0;
    return true;
}

bool Model::logits_rows(const float* h, const uint16_t* ids, uint32_t n, float* out) {
    uint64_t t0 = nr_micros();
    const CactRec& e = rec(i_emb);
    float* hr = mlp;
    op_aq(h, D, pbuf);
    cq_rotate(pbuf, D, e.in_pad(), hr);
    for (uint32_t i = 0; i < n; i++)
        if (!matmul(e, hr, 1, out + i, 1, ids[i], 1)) return false;
    stats.logits_us += nr_micros() - t0;
    return true;
}

// ---------------------------------------------------------------------------
uint32_t Model::state_bytes() const {
    const uint32_t qkv_n = (H + KVH) * QK + KVH * VD;
    uint32_t b = 16 + sizeof(tok_hist);
    b += L * pos * (KVH * QK + KVH * VD + 2 * KVH * 4);
    b += L * (taps - 1) * qkv_n * 4;
    b += n_sites * ehist * D * 2;
    if (has_confidence) b += pool.rows * pool.k * (2 + D) * 4;
    return b;
}

bool Model::state_io(StateIO* io, bool saving) {
    const uint32_t qkv_n = (H + KVH) * QK + KVH * VD;
    uint32_t hdr[4] = {0x4E53545Au, pos, L, n_sites};
    if (!io->io(hdr, 16)) return false;
    if (!saving) {
        if (hdr[0] != 0x4E53545Au || hdr[2] != L || hdr[3] != n_sites || hdr[1] > ctx_cap)
            return false;
        pos = hdr[1];
    }
    bool ok = io->io(tok_hist, sizeof(tok_hist));
    for (uint32_t l = 0; ok && l < L; l++) {
        ok = io->io(kc[l], pos * KVH * QK) && io->io(vc[l], pos * KVH * VD) &&
             io->io(ks[l], pos * KVH * 4) && io->io(vs[l], pos * KVH * 4) &&
             io->io(qkv_hist[l], (taps - 1) * qkv_n * 4);
    }
    for (uint32_t s = 0; ok && s < n_sites; s++) ok = io->io(ev_hist[s], ehist * D * 2);
    if (ok && has_confidence) {
        const uint32_t R = pool.rows * pool.k;
        ok = io->io(pool.m, R * 4) && io->io(pool.s, R * 4);
        // acc may live in word-only IRAM: move it through a bounce buffer
        float bounce[64];
        for (uint32_t i = 0; ok && i < R * D; i += 64) {
            uint32_t n = R * D - i < 64 ? R * D - i : 64;
            if (saving) for (uint32_t j = 0; j < n; j++) bounce[j] = ld32(pool.acc + i + j);
            ok = io->io(bounce, n * 4);
            if (!saving) for (uint32_t j = 0; j < n; j++) st32(pool.acc + i + j, bounce[j]);
        }
    }
    return ok;
}
