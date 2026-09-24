#include "grammar.h"

#include <stdio.h>
#include <string.h>

// Phases of the call-list automaton
enum {
    P_OPEN,       // expect '['
    P_FIRST,      // after '[': '{' starts a call, ']' ends an empty list
    P_NAME_LIT,   // inside  {"name":"
    P_NAME,       // tool-name bytes, or '"' when a full name matched
    P_ARGS_LIT,   // inside  ,"arguments":
    P_ARGS,       // the arguments value; v1: {}
    P_CLOSE,      // expect '}' closing the call
    P_NEXT,       // ',' then '{' for another call, or ']'
    P_COMMA,      // after ',': expect '{'
    P_DONE        // list closed
};

static const char NAME_LIT[] = "\"name\":\"";          // after '{'
static const char ARGS_LIT[] = ",\"arguments\":";
static const char ARGS_EMPTY[] = "{}";

bool Grammar::init(const char* const* nm, int n, int mc) {
    if (n > GR_MAX_TOOLS) return false;
    n_tools = n;
    max_calls = mc < 1 ? 1 : mc > 4 ? 4 : mc;
    for (int i = 0; i < n; i++) {
        size_t l = strlen(nm[i]);
        if (l >= GR_MAX_NAME) return false;
        memcpy(names[i], nm[i], l + 1);
        name_lens[i] = (uint8_t)l;
    }
    return true;
}

bool Grammar::feed(GrammarState* s, const char* bytes, int n) const {
    for (int i = 0; i < n; i++) {
        const char c = bytes[i];
        switch (s->phase) {
            case P_OPEN:
                if (c != '[') return false;
                s->phase = P_FIRST;
                break;
            case P_FIRST:
                if (c == ']') { s->phase = P_DONE; break; }
                if (c != '{') return false;
                s->phase = P_NAME_LIT; s->lit = 0;
                break;
            case P_COMMA:
                if (c != '{') return false;
                s->phase = P_NAME_LIT; s->lit = 0;
                break;
            case P_NAME_LIT:
                if (c != NAME_LIT[s->lit]) return false;
                if (!NAME_LIT[++s->lit]) {
                    s->phase = P_NAME; s->name_len = 0;
                    s->alive = (uint16_t)((1u << n_tools) - 1);
                }
                break;
            case P_NAME: {
                if (c == '"') {
                    int hit = -1;
                    for (int t = 0; t < n_tools; t++)
                        if ((s->alive >> t & 1) && name_lens[t] == s->name_len) hit = t;
                    if (hit < 0) return false;
                    s->chosen[s->calls] = (int8_t)hit;
                    s->phase = P_ARGS_LIT; s->lit = 0;
                    break;
                }
                uint16_t a = 0;
                for (int t = 0; t < n_tools; t++)
                    if ((s->alive >> t & 1) && s->name_len < name_lens[t] &&
                        names[t][s->name_len] == c)
                        a |= (uint16_t)(1u << t);
                if (!a) return false;
                s->alive = a;
                s->name_len++;
                break;
            }
            case P_ARGS_LIT:
                if (c != ARGS_LIT[s->lit]) return false;
                if (!ARGS_LIT[++s->lit]) { s->phase = P_ARGS; s->lit = 0; }
                break;
            case P_ARGS:  // zero-argument tools: the empty object only
                if (c != ARGS_EMPTY[s->lit]) return false;
                if (!ARGS_EMPTY[++s->lit]) s->phase = P_CLOSE;
                break;
            case P_CLOSE:
                if (c != '}') return false;
                s->calls++;
                s->phase = P_NEXT;
                break;
            case P_NEXT:
                if (c == ']') { s->phase = P_DONE; break; }
                if (c != ',' || s->calls >= max_calls) return false;
                s->phase = P_COMMA;
                break;
            default:
                return false;
        }
    }
    return true;
}

bool Grammar::accepts_end(const GrammarState& s) const { return s.phase == P_DONE; }

// A token can only ever be allowed if its surface occurs inside some accepted
// string. Accepted lists repeat one call shape, so every substring of any of
// them already occurs in a list of at most two calls: "[]" plus the n^2
// two-call lists cover them all.
static int call_text(char* dst, const char* name) {
    return sprintf(dst, "{%s%s\"%s%s}", NAME_LIT, name, ARGS_LIT, ARGS_EMPTY);
}

bool Grammar::build_candidates(Tokenizer* tk) {
    n_cand = 0;
    char buf[64], uni[160];
    for (uint32_t id = 0; id < tk->n; id++) {
        uint8_t ty = tk->type((uint16_t)id);
        if (ty == 2 || ty == 1 || ty == 3) continue;  // control, unknown, markers
        int len = tk->decode_append((uint16_t)id, buf, sizeof(buf) - 1);
        if (len <= 0) continue;
        buf[len] = 0;
        bool ok = strstr("[]", buf) != nullptr;
        for (int a = 0; a < n_tools && !ok; a++)
            for (int b = 0; b < n_tools && !ok; b++) {
                int w = 0;
                uni[w++] = '[';
                w += call_text(uni + w, names[a]);
                uni[w++] = ',';
                w += call_text(uni + w, names[b]);
                uni[w++] = ']';
                uni[w] = 0;
                ok = memchr(buf, 0, len) == nullptr && strstr(uni, buf) != nullptr;
            }
        if (!ok) continue;
        if (n_cand >= GR_MAX_CANDIDATES) return false;
        cand[n_cand++] = (uint16_t)id;
    }
    return true;
}

int Grammar::allowed(Tokenizer* tk, const GrammarState& s, uint16_t* ids, int cap) {
    int n = 0;
    char buf[64];
    for (int i = 0; i < n_cand && n < cap; i++) {
        int len = tk->decode_append(cand[i], buf, sizeof(buf));
        GrammarState t = s;
        if (feed(&t, buf, len)) ids[n++] = cand[i];
    }
    return n;
}

int Grammar::forced_tail(const GrammarState& s0, char* out, int cap) const {
    GrammarState s = s0;
    int n = 0;
    auto put = [&](char c) {
        if (n >= cap || !feed(&s, &c, 1)) return false;
        out[n++] = c;
        return true;
    };
    while (s.phase != P_DONE) {
        char c;
        switch (s.phase) {
            case P_NAME_LIT: c = NAME_LIT[s.lit]; break;
            case P_ARGS_LIT: c = ARGS_LIT[s.lit]; break;
            case P_ARGS: c = ARGS_EMPTY[s.lit]; break;
            case P_CLOSE: c = '}'; break;
            case P_COMMA: c = '{'; break;
            case P_NEXT:
                if (s.calls < max_calls) return -1;  // ',' or ']': a real choice
                c = ']';
                break;
            case P_NAME: {  // forced only when one name is still alive and it is unfinished or done
                int alive = 0, t = -1;
                for (int i = 0; i < n_tools; i++) if (s.alive >> i & 1) { alive++; t = i; }
                if (alive != 1) return -1;
                c = s.name_len < name_lens[t] ? names[t][s.name_len] : '"';
                break;
            }
            default:
                return -1;  // P_OPEN / P_FIRST: '[' then a call or ']'
        }
        if (!put(c)) return -1;
    }
    return n;
}
