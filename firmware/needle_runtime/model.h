// The Needle 3 forward pass (Laddered Simple Attention Network), streamed
// from a .cact archive: one layer's weights are read, used and dropped before
// the next is touched. Mirrors tools/ref_forward.py line for line; that file
// is pinned to the reference JAX model (tools/check_ref.py).
#pragma once
#include <stdint.h>

#include "cact_reader.h"
#include "tensor.h"

#ifndef NR_MAX_CHUNK
#define NR_MAX_CHUNK 4      // tokens per prefill chunk (weights read once per chunk)
#endif
#define NR_MAX_LAYERS 20
#define NR_MAX_SITES 5
#define NR_MAX_LANES 4

struct ModelStats {
    uint32_t tokens = 0;        // positions processed
    uint32_t chunks = 0;
    uint64_t forward_us = 0;
    uint64_t logits_us = 0;
    uint64_t engram_us = 0, attn_us = 0, mlp_us = 0;  // parts of forward_us
};

// Sequential byte stream for state snapshots (a file on SD or disk).
struct StateIO {
    virtual ~StateIO() {}
    virtual bool io(void* p, uint32_t n) = 0;   // write (saving) or read (loading) n bytes
};

// Consumer of the output head's logits, fed in row tiles so no 8192-float
// vector is ever held.
struct LogitSink {
    virtual ~LogitSink() {}
    virtual void consume(uint32_t first_id, const float* logits, uint32_t n) = 0;
};

// Probe-head pool (architecture.probe_pool) kept as a streaming softmax over
// positions, so the hidden cells of past tokens need not be stored.
struct ProbePool {
    uint32_t rows = 0, k = 0, D = 0;  // rows = L+1 cells per position, k probes
    float* m = nullptr;               // running max   [rows*k]
    float* s = nullptr;               // running sum   [rows*k]
    float* acc = nullptr;             // running sum of e^(score-m) * cell  [rows*k*D]
    bool init(uint32_t rows, uint32_t k, uint32_t D);
    void reset();
    void release();
};

class Model {
public:
    bool load(TensorReader* reader, uint32_t ctx_cap);
    void release();
    void reset();                        // forget all positions

    // Advance over n tokens (n <= NR_MAX_CHUNK). If h_last is given, it gets
    // the final-normalised hidden state of the last token.
    bool forward(const uint16_t* toks, uint32_t n, float* h_last);
    // Output head (tied embedding) over every vocabulary row, streamed.
    bool logits(const float* h, LogitSink* sink);
    // Output head for a subset of rows only (grammar-constrained steps).
    bool logits_rows(const float* h, const uint16_t* ids, uint32_t n, float* out);
    // Confidence head over every position seen since reset(); returns the logit.
    bool confidence_logit(float* out);

    // Snapshot of all per-position state (KV cache, conv histories, probe
    // pool), so a static prefix is computed once and restored per turn.
    // Streamed through `io` piece by piece: no second copy in RAM.
    uint32_t state_bytes() const;
    bool state_io(StateIO* io, bool saving);

    Cact cact;
    TensorReader* rd = nullptr;
    ModelStats stats;
    uint32_t pos = 0;                    // positions already in the cache
    uint32_t ctx_cap = 0;
    bool has_confidence = false;

    // geometry
    uint32_t L, D, H, KVH, QK, VD, lanes, taps, hada_n, vocab, out_vocab;
    uint32_t n_sites, n_tables, slots, sub, etaps, edil, n_orders;
    uint32_t window;   // attention window of local layers

private:
    // tensor directory positions
    uint32_t i_emb, i_layer0, i_mhc, i_perm, i_engram0, i_final, i_heads, i_conf;
    const CactRec& rec(uint32_t i) const { return cact.recs[i]; }
    const CactRec& lrec(uint32_t l, uint32_t k) const { return cact.recs[i_layer0 + 27 * l + k]; }

    bool matmul(const CactRec& t, const float* xr, uint32_t n, float* y, uint32_t ldy,
                uint32_t row0 = 0, uint32_t rows = 0);
    bool engram(const uint16_t* toks, uint32_t n);
    bool attention(uint32_t l, uint32_t n);
    bool hada_mlp(uint32_t l, float* x);   // in place, D in -> D out
    bool pool_update(uint32_t row, const float* cell);
    float* layer_param(uint32_t l, uint32_t k, uint32_t first, uint32_t count, float* dst);

    CqScratch scratch;

    // mHC small parameters, resident (L*... floats)
    float *a_pre = nullptr, *a_post = nullptr, *a_res = nullptr;
    float *b_pre = nullptr, *b_post = nullptr, *b_res = nullptr;
    uint16_t *perm1 = nullptr, *perm2 = nullptr;

    // per-position state
    int8_t* kc[NR_MAX_LAYERS] = {};     // [ctx][KVH*QK]
    int8_t* vc[NR_MAX_LAYERS] = {};     // [ctx][KVH*VD]
    float* ks[NR_MAX_LAYERS] = {};      // [ctx][KVH]
    float* vs[NR_MAX_LAYERS] = {};
    float* qkv_hist[NR_MAX_LAYERS] = {};     // ring [taps-1][Q+K+V] previous raw projections
    uint16_t* ev_hist[NR_MAX_SITES] = {};    // fp16 ring [ehist][D] raw engram values
    uint32_t ehist = 0;
    uint16_t tok_hist[4] = {};          // last tokens, newest first (engram hash)
    ProbePool pool;

    // work buffers (chunk-sized)
    float* X = nullptr;      // [n][lanes*D]  residual lanes
    float* xr = nullptr;     // [n][lanes*D]  rotated activation for the next matmul
    float* u0 = nullptr;     // [n][D] block input
    float* u = nullptr;      // [n][D]
    float* hb = nullptr;     // [n][D]
    float* qkv = nullptr;    // [n][Q+K+V]
    float* gate = nullptr;   // [n][H*VD]
    float* att = nullptr;    // [n][H*VD]
    float* ek = nullptr;     // [sites][n][D]
    float* ev = nullptr;     // [sites][n][D]
    float* mix = nullptr;    // [n][lanes + lanes + lanes*lanes]
    float* scores = nullptr; // [ctx]
    float* mlp = nullptr;    // 4 * hada_n scratch
    float* pbuf = nullptr;   // hada_n floats for streamed FP16 params
};
