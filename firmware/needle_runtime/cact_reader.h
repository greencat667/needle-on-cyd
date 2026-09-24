// .cact archive access: a replaceable byte source (TensorReader) and the
// parsed header + nameless tensor directory ("The .cact Format", docs/upstream/SOURCES.md,
// vendor/needle/needle/model/export.py). Nothing here loads a tensor; the
// runtime reads each weight from its absolute offset when it needs it.
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

// ---------------------------------------------------------------------------
// Storage backends
// ---------------------------------------------------------------------------
class TensorReader {
public:
    virtual ~TensorReader() {}
    virtual bool read(uint32_t offset, void* dst, size_t len) = 0;
    // Bytes [offset, offset+len) somewhere in memory: in `buf` (cap bytes,
    // 4-aligned) or in a buffer the reader already holds. Backends that can
    // transfer straight into buf without a copy override it.
    virtual const uint8_t* view(uint32_t offset, uint32_t len, uint8_t* buf, uint32_t cap) {
        return len <= cap && read(offset, buf, len) ? buf : nullptr;
    }
    uint64_t bytes_read = 0;
    uint32_t reads = 0;
    uint64_t read_us = 0;   // time spent inside read()
};

// stdio file: the desktop build, and the ESP32 SD card through the FAT VFS.
class FileReader : public TensorReader {
public:
    bool open(const char* path, uint32_t block = 8192);
    void close();
    uint32_t size() const { return size_; }
    bool read(uint32_t offset, void* dst, size_t len) override;
    const uint8_t* view(uint32_t offset, uint32_t len, uint8_t* buf, uint32_t cap) override;
    ~FileReader() override { close(); }
    uint32_t card_reads = 0;      // block transfers actually issued
    uint64_t card_bytes = 0;
private:
    struct Way { uint8_t* buf = nullptr; uint32_t off = 0xFFFFFFFFu, len = 0, used = 0; };
    Way* fill(uint32_t at);
    int fd_ = -1;
    uint32_t size_ = 0;
    uint32_t pos_ = 0xFFFFFFFFu;  // current file position (skip redundant seeks)
    Way way_[2];                  // aligned block buffers, 2-way LRU
    uint32_t block_ = 4096, tick_ = 0;
};

// A window into another reader starting at `base` (a file with a header).
class OffsetReader : public TensorReader {
public:
    OffsetReader(TensorReader* r = nullptr, uint32_t base = 0) : r_(r), base_(base) {}
    void set(TensorReader* r, uint32_t base) { r_ = r; base_ = base; }
    bool read(uint32_t offset, void* dst, size_t len) override {
        bytes_read += len;
        reads++;
        return r_->read(base_ + offset, dst, len);
    }
private:
    TensorReader* r_;
    uint32_t base_;
};

// Reads a byte range of the archive from a second source (the ESP32's own
// flash partition holding a verbatim copy of those bytes) and everything else
// from the base reader. The copy is byte-identical to the file.
class OverlayReader : public TensorReader {
public:
    OverlayReader(TensorReader* base) : base_(base) {}
    // Bytes [start, start+len) of the archive are served by `src` at src_offset + (off - start).
    bool add(uint32_t start, uint32_t len, TensorReader* src, uint32_t src_offset);
    bool read(uint32_t offset, void* dst, size_t len) override;
    const uint8_t* view(uint32_t offset, uint32_t len, uint8_t* buf, uint32_t cap) override;
    uint64_t overlay_bytes = 0;
    static const int MAX_RANGES = 96;
    int ranges() const { return n_; }
private:
    struct Range { uint32_t start, len, src_off; TensorReader* src; };
    int find(uint32_t offset) const;
    TensorReader* base_;
    Range ranges_[MAX_RANGES];
    int n_ = 0;
};

// ---------------------------------------------------------------------------
// Archive structure
// ---------------------------------------------------------------------------
enum { CACT_FP16 = 1, CACT_FP32 = 2, CACT_CQ = 3, CACT_RAW = 4 };
#define CACT_TAG 0x05E12A84u

struct CactRec {
    uint8_t dtype, ndim;
    uint32_t shape[4];
    uint32_t offset;   // absolute (archives are < 4 GB)
    uint32_t nbytes;
    uint16_t group;
    uint8_t bits;
    // CQ helpers
    uint32_t rows() const { return shape[0]; }
    uint32_t cols() const { return ndim > 1 ? shape[1] : 1; }
    uint32_t in_pad() const { return group ? (cols() + group - 1) / group * group : cols(); }
    uint32_t packed_row_bytes() const { return in_pad() * bits / 8; }
    uint32_t groups_per_row() const { return in_pad() / group; }
    uint32_t norms_offset() const { return offset + rows() * packed_row_bytes(); }
};

struct CactHeader {
    uint32_t tag, num_tensors, codebook_len, kv_window, kv_bits;
    uint32_t vocab, out_vocab, d_model, num_heads, num_kv_heads, num_layers;
    uint32_t qk_head_dim, v_head_dim, max_seq_len, hada_n, mhc_lanes;
    uint32_t sliding_window, global_mask_lo, global_mask_hi, qkv_conv_taps;
    uint32_t engram_slots, engram_sub_dim, num_engram_tables, engram_conv_taps;
    uint32_t engram_conv_dilation, engram_seed_heads, num_engram_orders;
    uint32_t engram_orders[4];
    uint32_t num_engram_sites;
    uint32_t engram_sites[16];
    float rope_theta;
};

struct Cact {
    CactHeader h;
    float codebook[28];     // cb2[4] | cb3[8] | cb4[16], already divided by sqrt(group)
    CactRec* recs = nullptr;
    uint32_t n_recs = 0;

    bool parse(TensorReader* r);
    void release();
    const float* cb(int bits) const {
        return bits == 2 ? codebook : bits == 3 ? codebook + 4 : codebook + 12;
    }
    bool is_global(uint32_t layer) const {
        return layer < 32 ? (h.global_mask_lo >> layer) & 1 : (h.global_mask_hi >> (layer - 32)) & 1;
    }
    int engram_site_of(uint32_t layer) const {
        for (uint32_t s = 0; s < h.num_engram_sites; s++)
            if (h.engram_sites[s] == layer) return (int)s;
        return -1;
    }
};
