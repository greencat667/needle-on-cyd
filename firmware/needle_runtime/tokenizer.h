// SentencePiece BPE from the archive's RAW tokenizer blob, read in place.
// Port of needle.model.export.RefTokenizer (the reference encoder the format
// specifies). Only a coarse offset index lives in RAM; the piece table stays
// in storage. The surface->id hash table is built for an encode() and can be
// dropped afterwards.
#pragma once
#include <stdint.h>

#include "cact_reader.h"

class Tokenizer {
public:
    bool load(TensorReader* rd, const CactRec& blob);
    void release();

    // Build / drop the surface lookup table (needed by encode only).
    bool open_lookup();
    void close_lookup();
    // Keep the table in storage instead of RAM: write the built table out,
    // then serve lookups from `r` (a file on SD) at 2 bytes per probe.
    uint32_t table_bytes() const { return table_size_ * 2; }
    const uint16_t* table() const { return table_; }
    void use_stored_table(TensorReader* r, uint32_t size_entries);

    // Encode UTF-8 text; returns the token count or -1 (out full / error).
    int encode(const char* text, uint16_t* out, int cap);
    // Surface bytes of a piece (UTF-8, with U+2581 for spaces). Returns length.
    int piece(uint16_t id, char* dst, int cap);
    // Append the decoded text of one token to dst (spaces restored, control
    // pieces dropped, byte pieces as raw bytes). Returns bytes written.
    int decode_append(uint16_t id, char* dst, int cap);
    uint8_t type(uint16_t id);
    float score(uint16_t id);
    int32_t lookup(const char* s, int len);   // id of an exact surface, or -1

    uint32_t n = 0;
    uint32_t pad_id = 0, eos_id = 1, bos_id = 2, unk_id = 3;
    bool add_dummy = false, byte_fallback = false;

private:
    bool rec_at(uint16_t id, uint32_t* off, float* score, uint8_t* type, uint16_t* len);
    TensorReader* rd_ = nullptr;
    uint32_t base_ = 0;           // blob offset in the archive
    uint32_t* coarse_ = nullptr;  // offset of every 16th record, relative to base_
    uint16_t* table_ = nullptr;   // open-addressing surface hash -> id+1
    uint32_t table_size_ = 0;
    TensorReader* table_rd_ = nullptr;  // stored table (when table_ is null)
    uint16_t entry(uint32_t h) {
        if (table_) return table_[h];
        uint16_t e = 0;
        if (table_rd_) table_rd_->read(h * 2, &e, 2);
        return e;
    }
    // markers (USER_DEFINED pieces), longest first
    uint16_t markers_[32];
    uint8_t marker_len_[32];
    char marker_str_[32][16];
    int n_markers_ = 0;
    uint16_t byte_id_[256];
    // one-entry record cache (piece reads cluster on the same ids)
    uint16_t cache_id_ = 0xFFFF;
    uint32_t cache_off_ = 0;
};
