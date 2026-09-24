// One Needle turn on the device: static tool prefix (computed once, then
// restored from a snapshot), the user turn, forced <think>, greedy reasoning
// under a token budget, then the tool call decoded under the grammar, and the
// confidence head. Follows the wire format in "Porting Needle 3", docs/upstream/SOURCES.md.
#pragma once
#include <stdint.h>

#include "grammar.h"
#include "model.h"
#include "tokenizer.h"

enum {
    TOK_BOS = 2, TOK_EOS = 1, TOK_IM_END = 5, TOK_THINK = 6, TOK_THINK_END = 7,
    TOK_TOOL_CALL = 10, TOK_TOOL_CALL_END = 11, TOK_NEWLINE = 38
};

// Progress callback: phase 0 = after every forward position (step = position),
// 1 = turn prefill starts, 2 = reasoning, 3 = call.
// `text` is the reasoning or call text so far.
typedef void (*NeedleProgress)(void* ctx, int phase, int step, const char* text);

struct NeedleOptions {
    int think_budget = 24;        // reasoning tokens before </think> is forced
    int max_calls = 1;            // calls the grammar admits per turn
    bool full_vocab_prob = false; // call probability over the whole vocabulary
                                  // (default: renormalised over the grammar's
                                  // allowed tokens, which needs only those rows)
    bool trace = false;           // print every generated token
    bool jump_forward = true;     // once the grammar fixes the rest of the call, emit it
                                  // without running the network (skipped when a
                                  // confidence head needs every position)
    NeedleProgress progress = nullptr;
    void* progress_ctx = nullptr;
};

struct NeedleResult {
    bool ok = false;
    int n_calls = 0;
    int call_tool[4] = {-1, -1, -1, -1};
    char call_json[200] = {0};
    char reasoning[600] = {0};
    int reasoning_tokens = 0;
    int jumped_bytes = 0;         // call bytes emitted by jump-forward
    bool think_truncated = false;
    float call_prob = 0;          // probability of the decoded call tokens
    float head_logit = 0;         // confidence head
    float confidence = 0;         // min(sigmoid(head), call_prob)
    uint32_t turn_tokens = 0;     // turn tokens fed this time
    uint32_t prefix_tokens = 0;
    uint32_t generated = 0;
    uint64_t total_us = 0, prefill_us = 0, decode_us = 0, encode_us = 0, restore_us = 0;
    uint64_t bytes_read = 0;
    uint32_t reads = 0;
};

// Where the prefix snapshot lives: a file on the SD card or on the host disk.
struct SnapshotStore {
    virtual ~SnapshotStore() {}
    virtual StateIO* open(bool saving) = 0;   // nullptr when absent / unwritable
    virtual void close(StateIO* io) = 0;
};

class NeedleSession {
public:
    bool init(Model* m, Tokenizer* tk, const char* tools_json, const char* const* names,
              int n_tools, const char* system, int max_calls);
    // Tokenise and run the static prefix, then snapshot it (store may be null:
    // the prefix is then recomputed every turn).
    bool build_prefix(SnapshotStore* store);
    bool run(const char* query, const NeedleOptions& opt, NeedleResult* out);

    uint32_t prefix_tokens = 0;

private:
    bool restore_prefix();
    bool run_prefix();
    bool feed(const uint16_t* t, uint32_t n, bool want_h);
    Model* m_ = nullptr;
    Tokenizer* tk_ = nullptr;
    Grammar gr_;
    const char* tools_json_ = nullptr;
    const char* system_ = nullptr;
    SnapshotStore* store_ = nullptr;
    const NeedleOptions* opt_ = nullptr;  // during run(): for the per-position hook
    bool have_snapshot_ = false;
    uint16_t ids_[320];
    float h_[1024];
};
