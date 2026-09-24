#include "tokenizer.h"

#include <math.h>
#include <string.h>

#include "nr_platform.h"

enum { TK_NORMAL = 0, TK_UNKNOWN = 1, TK_CONTROL = 2, TK_USER_DEFINED = 3, TK_BYTE = 4 };
static const uint32_t HDR = 24;  // <IIIIIBBH
static const uint32_t REC = 7;   // <fBH

static uint32_t fnv(const char* s, int n) {
    uint32_t h = 2166136261u;
    for (int i = 0; i < n; i++) h = (h ^ (uint8_t)s[i]) * 16777619u;
    return h;
}

bool Tokenizer::load(TensorReader* rd, const CactRec& blob) {
    rd_ = rd;
    base_ = blob.offset;
    uint8_t h[HDR];
    if (!rd->read(base_, h, HDR)) return false;
    memcpy(&n, h, 4);
    memcpy(&pad_id, h + 4, 4);
    memcpy(&eos_id, h + 8, 4);
    memcpy(&bos_id, h + 12, 4);
    memcpy(&unk_id, h + 16, 4);
    add_dummy = h[20];
    byte_fallback = h[21];
    coarse_ = (uint32_t*)nr_alloc(((n + 15) / 16) * 4, "tokenizer coarse index");
    if (!coarse_) return false;
    memset(byte_id_, 0xFF, sizeof(byte_id_));
    n_markers_ = 0;
    // One sequential pass: coarse offsets, byte pieces, markers.
    uint32_t off = HDR;
    uint8_t r[REC + 32];
    for (uint32_t i = 0; i < n; i++) {
        if (i % 16 == 0) coarse_[i / 16] = off;
        if (!rd->read(base_ + off, r, REC)) return false;
        uint16_t len;
        memcpy(&len, r + 5, 2);
        uint8_t t = r[4];
        if (t == TK_BYTE || t == TK_USER_DEFINED) {
            char s[32];
            if (len >= sizeof(s) || !rd->read(base_ + off + REC, s, len)) return false;
            s[len] = 0;
            if (t == TK_BYTE) {  // "<0xAB>"
                unsigned v = 0;
                for (int k = 3; k < 5; k++) {
                    char c = s[k];
                    v = v * 16 + (c >= 'A' ? (c & 0xDF) - 'A' + 10 : c - '0');
                }
                byte_id_[v & 0xFF] = (uint16_t)i;
            } else if (n_markers_ < 32 && len < 16) {
                memcpy(marker_str_[n_markers_], s, len + 1);
                marker_len_[n_markers_] = (uint8_t)len;
                markers_[n_markers_++] = (uint16_t)i;
            }
        }
        off += REC + len;
    }
    // longest first, stable (RefTokenizer sorts by len, reverse=True)
    for (int i = 1; i < n_markers_; i++)
        for (int j = i; j > 0 && marker_len_[j] > marker_len_[j - 1]; j--) {
            uint16_t m = markers_[j]; markers_[j] = markers_[j - 1]; markers_[j - 1] = m;
            uint8_t l = marker_len_[j]; marker_len_[j] = marker_len_[j - 1]; marker_len_[j - 1] = l;
            char tmp[16];
            memcpy(tmp, marker_str_[j], 16);
            memcpy(marker_str_[j], marker_str_[j - 1], 16);
            memcpy(marker_str_[j - 1], tmp, 16);
        }
    return true;
}

void Tokenizer::release() {
    close_lookup();
    nr_free(coarse_);
    coarse_ = nullptr;
}

bool Tokenizer::rec_at(uint16_t id, uint32_t* off, float* score, uint8_t* type, uint16_t* len) {
    if (id >= n) return false;
    uint32_t o;
    uint16_t i;
    if (cache_id_ != 0xFFFF && cache_id_ <= id && id - cache_id_ < 16) {
        o = cache_off_;
        i = cache_id_;
    } else {
        o = coarse_[id / 16];
        i = (uint16_t)(id / 16 * 16);
    }
    uint8_t r[REC];
    for (;;) {
        if (!rd_->read(base_ + o, r, REC)) return false;
        uint16_t l;
        memcpy(&l, r + 5, 2);
        if (i == id) {
            *off = o;
            if (score) memcpy(score, r, 4);
            if (type) *type = r[4];
            *len = l;
            cache_id_ = id;
            cache_off_ = o;
            return true;
        }
        o += REC + l;
        i++;
    }
}

int Tokenizer::piece(uint16_t id, char* dst, int cap) {
    uint32_t off;
    uint16_t len;
    if (!rec_at(id, &off, nullptr, nullptr, &len) || len >= cap) return -1;
    if (!rd_->read(base_ + off + REC, dst, len)) return -1;
    dst[len] = 0;
    return len;
}

uint8_t Tokenizer::type(uint16_t id) {
    uint32_t off; uint16_t len; uint8_t t = 0;
    rec_at(id, &off, nullptr, &t, &len);
    return t;
}

float Tokenizer::score(uint16_t id) {
    uint32_t off; uint16_t len; float s = 0;
    rec_at(id, &off, &s, nullptr, &len);
    return s;
}

bool Tokenizer::open_lookup() {
    if (table_ || table_rd_) return true;
    table_size_ = 1;
    while (table_size_ < n * 2) table_size_ <<= 1;
    table_ = (uint16_t*)nr_alloc(table_size_ * 2, "tokenizer hash");
    if (!table_) return false;
    uint32_t off = HDR;
    uint8_t r[REC];
    char s[64];
    for (uint32_t i = 0; i < n; i++) {
        if (!rd_->read(base_ + off, r, REC)) return false;
        uint16_t len;
        memcpy(&len, r + 5, 2);
        if (len < sizeof(s) && rd_->read(base_ + off + REC, s, len)) {
            // later duplicates win, as in the reference dict
            uint32_t hsh = fnv(s, len) & (table_size_ - 1);
            for (;;) {
                uint16_t e = table_[hsh];
                if (e == 0) { table_[hsh] = (uint16_t)(i + 1); break; }
                char t[64];
                int tl = piece((uint16_t)(e - 1), t, sizeof(t));
                if (tl == len && memcmp(t, s, len) == 0) { table_[hsh] = (uint16_t)(i + 1); break; }
                hsh = (hsh + 1) & (table_size_ - 1);
            }
        }
        off += REC + len;
    }
    return true;
}

void Tokenizer::close_lookup() {
    nr_free(table_);
    table_ = nullptr;
}

void Tokenizer::use_stored_table(TensorReader* r, uint32_t size_entries) {
    close_lookup();
    table_rd_ = r;
    table_size_ = size_entries;
}

int32_t Tokenizer::lookup(const char* s, int len) {
    if ((!table_ && !table_rd_) || len <= 0 || len >= 64) return -1;
    uint32_t hsh = fnv(s, len) & (table_size_ - 1);
    char t[64];
    for (;;) {
        uint16_t e = entry(hsh);
        if (e == 0) return -1;
        int tl = piece((uint16_t)(e - 1), t, sizeof(t));
        if (tl == len && memcmp(t, s, len) == 0) return e - 1;
        hsh = (hsh + 1) & (table_size_ - 1);
    }
}

// UTF-8 code point length at s
static int cp_len(const char* s) {
    uint8_t c = (uint8_t)s[0];
    return c < 0x80 ? 1 : c < 0xE0 ? 2 : c < 0xF0 ? 3 : 4;
}

// BPE over one marker-free segment (RefTokenizer._bpe).
static int bpe(Tokenizer* tk, const char* seg, int seglen, uint16_t* out, int cap,
               const uint16_t* byte_id) {
    enum { MAXS = 640 };
    static uint16_t st[MAXS], ln[MAXS];  // symbol start / length in seg
    static float ps[MAXS];               // score of pair (j, j+1), -inf if not a piece
    int ns = 0;
    for (int i = 0; i < seglen; ns++) {
        if (ns >= MAXS) return -1;
        int l = cp_len(seg + i);
        st[ns] = (uint16_t)i;
        ln[ns] = (uint16_t)l;
        i += l;
    }
    auto pair_score = [&](int j) -> float {
        int32_t id = tk->lookup(seg + st[j], ln[j] + ln[j + 1]);
        return id >= 0 ? tk->score((uint16_t)id) : -INFINITY;
    };
    for (int j = 0; j < ns - 1; j++) ps[j] = pair_score(j);
    // Same choice as the reference loop (highest score, leftmost on ties);
    // only the two pairs touching a merge are re-scored.
    while (ns > 1) {
        int bj = -1;
        for (int j = 0; j < ns - 1; j++)
            if (ps[j] != -INFINITY && (bj < 0 || ps[j] > ps[bj])) bj = j;
        if (bj < 0) break;
        ln[bj] = (uint16_t)(ln[bj] + ln[bj + 1]);
        for (int j = bj + 1; j < ns - 1; j++) { st[j] = st[j + 1]; ln[j] = ln[j + 1]; }
        for (int j = bj + 1; j < ns - 2; j++) ps[j] = ps[j + 1];
        ns--;
        if (bj > 0) ps[bj - 1] = pair_score(bj - 1);
        if (bj < ns - 1) ps[bj] = pair_score(bj);
    }
    int n = 0;
    for (int j = 0; j < ns; j++) {
        int32_t id = tk->lookup(seg + st[j], ln[j]);
        if (id >= 0) {
            if (n >= cap) return -1;
            out[n++] = (uint16_t)id;
        } else if (tk->byte_fallback) {
            for (int k = 0; k < ln[j]; k++) {
                if (n >= cap) return -1;
                out[n++] = byte_id[(uint8_t)seg[st[j] + k]];
            }
        } else {
            if (n >= cap) return -1;
            out[n++] = (uint16_t)tk->unk_id;
        }
    }
    return n;
}

int Tokenizer::encode(const char* text, uint16_t* out, int cap) {
    if (!table_ && !table_rd_ && !open_lookup()) return -1;
    // escape spaces to U+2581 into a working buffer
    static char esc[1024];
    int e = 0;
    if (add_dummy) { memcpy(esc, "\xE2\x96\x81", 3); e = 3; }
    for (const char* p = text; *p; p++) {
        if (e + 4 >= (int)sizeof(esc)) return -1;
        if (*p == ' ') { memcpy(esc + e, "\xE2\x96\x81", 3); e += 3; }
        else esc[e++] = *p;
    }
    int n_out = 0, seg0 = 0, i = 0;
    while (i < e) {
        int hit = -1;
        for (int m = 0; m < n_markers_; m++)
            if (i + marker_len_[m] <= e && memcmp(esc + i, marker_str_[m], marker_len_[m]) == 0) {
                hit = m;
                break;
            }
        if (hit >= 0) {
            int k = bpe(this, esc + seg0, i - seg0, out + n_out, cap - n_out, byte_id_);
            if (k < 0 || n_out + k >= cap) return -1;
            n_out += k;
            out[n_out++] = markers_[hit];
            i += marker_len_[hit];
            seg0 = i;
        } else {
            i += cp_len(esc + i);
        }
    }
    int k = bpe(this, esc + seg0, e - seg0, out + n_out, cap - n_out, byte_id_);
    if (k < 0) return -1;
    return n_out + k;
}

int Tokenizer::decode_append(uint16_t id, char* dst, int cap) {
    char s[64];
    uint8_t t = type(id);
    if (t == TK_CONTROL || t == TK_UNKNOWN) return 0;
    int len = piece(id, s, sizeof(s));
    if (len < 0) return 0;
    if (t == TK_BYTE) {
        unsigned v = 0;
        for (int k = 3; k < 5; k++) {
            char c = s[k];
            v = v * 16 + (c >= 'A' ? (c & 0xDF) - 'A' + 10 : c - '0');
        }
        if (cap < 1) return 0;
        dst[0] = (char)v;
        return 1;
    }
    int w = 0;
    for (int i = 0; i < len && w < cap; ) {
        if (i + 2 < len && (uint8_t)s[i] == 0xE2 && (uint8_t)s[i + 1] == 0x96 &&
            (uint8_t)s[i + 2] == 0x81) {
            dst[w++] = ' ';
            i += 3;
        } else {
            dst[w++] = s[i++];
        }
    }
    return w;
}
