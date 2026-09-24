#include "inference.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "nr_platform.h"
#include "ops.h"

// Streaming consumer of the full output head: argmax, log-sum-exp, and the
// logits of a small watched set of ids (the grammar's allowed tokens).
struct FullSink : LogitSink {
    float mx = -INFINITY, se = 0;
    float best = -INFINITY;
    uint32_t best_id = 0;
    const uint16_t* watch = nullptr;  // sorted ascending
    int n_watch = 0;
    float* watched = nullptr;
    void consume(uint32_t first, const float* lg, uint32_t n) override {
        for (uint32_t i = 0; i < n; i++) {
            const float v = lg[i];
            if (v > best) { best = v; best_id = first + i; }
            if (v > mx) { se = se * expf(mx - v) + 1.0f; mx = v; }
            else se += expf(v - mx);
        }
        for (int w = 0; w < n_watch; w++)
            if (watch[w] >= first && watch[w] < first + n) watched[w] = lg[watch[w] - first];
    }
    float lse() const { return mx + logf(se); }
};

static void sort_ids(uint16_t* a, int n) {
    for (int i = 1; i < n; i++)
        for (int j = i; j > 0 && a[j] < a[j - 1]; j--) { uint16_t t = a[j]; a[j] = a[j - 1]; a[j - 1] = t; }
}

bool NeedleSession::init(Model* m, Tokenizer* tk, const char* tools_json,
                         const char* const* names, int n_tools, const char* system,
                         int max_calls) {
    m_ = m;
    tk_ = tk;
    tools_json_ = tools_json;
    system_ = system;
    if (!gr_.init(names, n_tools, max_calls)) return false;
    return gr_.build_candidates(tk);
}

bool NeedleSession::feed(const uint16_t* t, uint32_t n, bool want_h) {
    for (uint32_t i = 0; i < n; i += NR_MAX_CHUNK) {
        uint32_t k = n - i < NR_MAX_CHUNK ? n - i : NR_MAX_CHUNK;
        const bool last = i + k == n;
        if (!m_->forward(t + i, k, last && want_h ? h_ : nullptr)) return false;
        if (opt_ && opt_->progress) opt_->progress(opt_->progress_ctx, 0, (int)m_->pos, nullptr);
    }
    return true;
}

bool NeedleSession::run_prefix() {
    static char text[2048];
    int w = 0;
    if (system_ && *system_)
        w = snprintf(text, sizeof(text), "<|im_start|>system\n%s<|im_end|>\n", system_);
    w += snprintf(text + w, sizeof(text) - w, "<|im_start|>user\n<tools>%s</tools>", tools_json_);
    if (w >= (int)sizeof(text)) return false;
    ids_[0] = TOK_BOS;
    int n = tk_->encode(text, ids_ + 1, (int)(sizeof(ids_) / 2) - 1);
    if (n < 0) return false;
    prefix_tokens = (uint32_t)n + 1;
    m_->reset();
    return feed(ids_, prefix_tokens, false);
}

// FILE-backed snapshot I/O lives with the platform (host/main or firmware).
bool NeedleSession::build_prefix(SnapshotStore* store) {
    store_ = store;
    have_snapshot_ = false;
    if (store_) {  // reuse a snapshot saved by an earlier boot
        StateIO* io = store_->open(false);
        if (io) {
            m_->reset();
            bool ok = io->io(&prefix_tokens, 4) && m_->state_io(io, false);
            store_->close(io);
            if (ok) { have_snapshot_ = true; return true; }
        }
    }
    if (!run_prefix()) return false;
    if (store_) {
        StateIO* io = store_->open(true);
        if (io) {
            bool ok = io->io(&prefix_tokens, 4) && m_->state_io(io, true);
            store_->close(io);
            have_snapshot_ = ok;
        }
    }
    return true;
}

bool NeedleSession::restore_prefix() {
    if (have_snapshot_) {
        StateIO* io = store_->open(false);
        if (!io) return false;
        m_->reset();
        uint32_t n;
        bool ok = io->io(&n, 4) && m_->state_io(io, false);
        store_->close(io);
        return ok;
    }
    return run_prefix();
}

bool NeedleSession::run(const char* query, const NeedleOptions& opt, NeedleResult* r) {
    struct Clear { const NeedleOptions** p; ~Clear() { *p = nullptr; } } clear{&opt_};
    opt_ = &opt;
    memset(r, 0, sizeof(*r));
    for (int i = 0; i < 4; i++) r->call_tool[i] = -1;
    gr_.max_calls = opt.max_calls;
    const uint64_t t0 = nr_micros();
    const uint64_t b0 = m_->rd->bytes_read;
    const uint32_t c0 = m_->rd->reads;
    { uint64_t _t = nr_micros(); if (!restore_prefix()) return false; r->restore_us = nr_micros() - _t; }
    r->prefix_tokens = prefix_tokens;

    // ---- the user turn, then the forced <think> -------------------------
    static char turn[512];
    int w = snprintf(turn, sizeof(turn), "\n%s<|im_end|>\n<|im_start|>assistant\n", query);
    if (w >= (int)sizeof(turn)) return false;
    uint64_t _te = nr_micros();
    int n = tk_->encode(turn, ids_, (int)(sizeof(ids_) / 2) - 1);
    r->encode_us = nr_micros() - _te;
    if (n < 0) return false;
    tk_->close_lookup();  // the surface table is only needed for encoding
    ids_[n++] = TOK_THINK;
    r->turn_tokens = (uint32_t)n;
    if (opt.progress) opt.progress(opt.progress_ctx, 1, n, "");
    if (!feed(ids_, (uint32_t)n, true)) return false;
    r->prefill_us = nr_micros() - t0;
    const uint64_t d0 = nr_micros();

    // ---- reasoning: greedy over the full vocabulary under a budget -------
    int rw = 0;
    bool closed = false;
    for (int i = 0; i < opt.think_budget; i++) {
        FullSink s;
        if (!m_->logits(h_, &s)) return false;
        uint16_t t = (uint16_t)s.best_id;
        if (opt.trace) nr_log("  think[%d] %u\n", i, (unsigned)t);
        if (t == TOK_THINK_END) {
            closed = true;
            break;
        }
        if (t == TOK_EOS || t == TOK_IM_END) break;
        rw += tk_->decode_append(t, r->reasoning + rw, (int)sizeof(r->reasoning) - 1 - rw);
        r->reasoning[rw] = 0;
        r->reasoning_tokens++;
        if (opt.progress) opt.progress(opt.progress_ctx, 2, i + 1, r->reasoning);
        r->generated++;
        if (!feed(&t, 1, true)) return false;
    }
    r->reasoning[rw] = 0;
    // trim the newline the model writes before </think>
    while (rw > 0 && (r->reasoning[rw - 1] == '\n' || r->reasoning[rw - 1] == ' '))
        r->reasoning[--rw] = 0;
    r->think_truncated = !closed;
    {
        const uint16_t close_seq[3] = {TOK_THINK_END, TOK_NEWLINE, TOK_TOOL_CALL};
        if (!feed(close_seq, 3, true)) return false;
    }

    // ---- the call, under the grammar ------------------------------------
    GrammarState gs;
    static uint16_t allowed[GR_MAX_CANDIDATES];
    static float lg[GR_MAX_CANDIDATES];
    double logp = 0;
    int cw = 0;
    for (int step = 0; step < 48; step++) {
        int na;
        if (gr_.accepts_end(gs)) {
            allowed[0] = TOK_TOOL_CALL_END;
            na = 1;
        } else {
            na = gr_.allowed(tk_, gs, allowed, GR_MAX_CANDIDATES);
        }
        if (na == 0) return false;
        sort_ids(allowed, na);
        int pick = 0;
        double lp = 0;
        if (opt.full_vocab_prob) {
            FullSink s;
            s.watch = allowed;
            s.n_watch = na;
            s.watched = lg;
            if (!m_->logits(h_, &s)) return false;
            for (int i = 1; i < na; i++) if (lg[i] > lg[pick]) pick = i;
            lp = lg[pick] - s.lse();
        } else if (na > 1) {
            if (!m_->logits_rows(h_, allowed, (uint32_t)na, lg)) return false;
            float mx = lg[0];
            for (int i = 1; i < na; i++) { if (lg[i] > lg[pick]) pick = i; if (lg[i] > mx) mx = lg[i]; }
            double se = 0;
            for (int i = 0; i < na; i++) se += exp(lg[i] - mx);
            lp = lg[pick] - mx - log(se);
        }
        logp += lp;
        uint16_t t = allowed[pick];
        if (opt.trace) nr_log("  call[%d] %u (%d allowed) p=%.4f\n", step, (unsigned)t, na, exp(lp));
        r->generated++;
        if (t == TOK_TOOL_CALL_END) {
            if (!feed(&t, 1, false)) return false;
            break;
        }
        char bytes[64];
        int bl = tk_->decode_append(t, bytes, sizeof(bytes));
        gr_.feed(&gs, bytes, bl);
        if (cw + bl < (int)sizeof(r->call_json)) { memcpy(r->call_json + cw, bytes, bl); cw += bl; }
        r->call_json[cw] = 0;
        if (opt.progress) opt.progress(opt.progress_ctx, 3, step + 1, r->call_json);
        if (opt.jump_forward && !m_->has_confidence) {
            char tail[64];
            int tl = gr_.forced_tail(gs, tail, sizeof(tail));
            if (tl >= 0 && cw + tl < (int)sizeof(r->call_json)) {
                gr_.feed(&gs, tail, tl);
                memcpy(r->call_json + cw, tail, tl);
                cw += tl;
                r->jumped_bytes = tl;
                if (opt.trace) nr_log("  jump-forward %d bytes\n", tl);
                if (opt.progress) opt.progress(opt.progress_ctx, 3, step + 1, r->call_json);
                break;
            }
        }
        if (!feed(&t, 1, true)) return false;
    }
    r->call_json[cw] = 0;
    r->n_calls = gs.calls;
    for (int i = 0; i < gs.calls && i < 4; i++) r->call_tool[i] = gs.chosen[i];
    r->call_prob = (float)exp(logp);
    r->decode_us = nr_micros() - d0;

    float hl = 0;
    if (m_->confidence_logit(&hl)) {
        r->head_logit = hl;
        const float head = op_sigmoid(hl);
        r->confidence = head < r->call_prob ? head : r->call_prob;
    } else {
        r->confidence = r->call_prob;
    }
    r->total_us = nr_micros() - t0;
    r->bytes_read = m_->rd->bytes_read - b0;
    r->reads = m_->rd->reads - c0;
    r->ok = true;
    return true;
}
