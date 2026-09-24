// Constrained tool-call decoding: the smallest faithful slice of Needle's
// byte-level grammar. A byte automaton accepts exactly the minified JSON the
// engine emits for calls to the declared tools,
//     []                                             (refusal)
//     [{"name":"eat","arguments":{}}]                (one call)
//     [{"name":"eat","arguments":{}},{"name":...}]   (up to max_calls)
// and a token is allowed when every byte of its surface keeps the automaton
// alive. Arguments are a separate phase (ARGS) so typed argument grammars can
// be added without touching the name trie; v1 accepts only the empty object.
#pragma once
#include <stdint.h>

#include "tokenizer.h"

#define GR_MAX_TOOLS 8
#define GR_MAX_NAME 24
#define GR_MAX_CANDIDATES 768

struct GrammarState {
    uint8_t phase = 0;       // see grammar.cpp
    uint8_t lit = 0;         // position inside the current literal
    uint8_t name_len = 0;    // bytes of the name matched so far
    uint8_t calls = 0;       // completed calls
    uint16_t alive = 0;      // bitmask of tools whose name still matches
    int8_t chosen[4] = {-1, -1, -1, -1};  // tool index of each completed call
};

class Grammar {
public:
    // names: the tool names as the prompt declares them.
    bool init(const char* const* names, int n_tools, int max_calls);
    // Pre-scan the vocabulary once: keep only tokens whose bytes can occur in
    // any accepted string (pre-tokenising the static tool schemas). A plain
    // scan of every piece; no prompt or answer is looked at.
    bool build_candidates(Tokenizer* tk);

    bool feed(GrammarState* s, const char* bytes, int n) const;
    bool accepts_end(const GrammarState& s) const;  // a complete call list
    // If only one byte string can finish the list from state s, write it to
    // out and return its length (the rest is fixed by the grammar); else -1.
    int forced_tail(const GrammarState& s, char* out, int cap) const;
    // Allowed candidate tokens from state s -> ids; returns the count.
    int allowed(Tokenizer* tk, const GrammarState& s, uint16_t* ids, int cap);

    int n_tools = 0;
    int max_calls = 1;
    char names[GR_MAX_TOOLS][GR_MAX_NAME];
    uint8_t name_lens[GR_MAX_TOOLS];
    uint16_t cand[GR_MAX_CANDIDATES];
    int n_cand = 0;
};
