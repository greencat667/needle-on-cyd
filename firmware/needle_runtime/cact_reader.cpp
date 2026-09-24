#include "cact_reader.h"

#include <fcntl.h>
#include <string.h>
#include <unistd.h>

#include "nr_platform.h"

// Every card read is a whole number of 512 B sectors at a sector-aligned file
// position into a 4-byte-aligned (DMA-capable) block buffer, so FATFS hands
// the SD driver multi-sector transfers straight into it. Unaligned buffers
// make the driver fall back to one bounced sector per command (measured on
// the CYD: 0.11 MB/s). The block doubles as a one-entry cache.
static const uint32_t SECTOR = 512;

bool FileReader::open(const char* path, uint32_t block) {
    close();
    // POSIX I/O, not stdio: newlib's FILE layer hands the VFS tiny pieces
    // (measured 0.11 MB/s through fread vs 2.4 MB/s raw on the CYD).
    fd_ = ::open(path, O_RDONLY);
    if (fd_ < 0) return false;
    off_t end = ::lseek(fd_, 0, SEEK_END);
    if (end < 0) { close(); return false; }
    size_ = (uint32_t)end;
    ::lseek(fd_, 0, SEEK_SET);
    pos_ = 0;
    // two ways of half the budget each: a matrix tile streams through one
    // while its norms (a separate region of the same tensor) sit in the other
    block_ = (block / 2) & ~(SECTOR - 1);
    if (block_ < SECTOR) block_ = SECTOR;
    for (int w = 0; w < 2; w++) {
        way_[w].buf = (uint8_t*)nr_alloc(block_, "sd block buffer");
        way_[w].off = 0xFFFFFFFFu;
        way_[w].len = 0;
        way_[w].used = 0;
        if (!way_[w].buf) return false;
    }
    return true;
}

void FileReader::close() {
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1;
    for (int w = 0; w < 2; w++) { nr_free(way_[w].buf); way_[w].buf = nullptr; }
}

FileReader::Way* FileReader::fill(uint32_t at) {
    Way* w = way_[0].used <= way_[1].used ? &way_[0] : &way_[1];  // least recently used
    const uint32_t start = at & ~(SECTOR - 1);
    uint32_t n = size_ - start < block_ ? size_ - start : block_;
    if (start != pos_ && ::lseek(fd_, start, SEEK_SET) != (off_t)start) return nullptr;
    size_t got = 0;
    while (got < n) {
        ssize_t r = ::read(fd_, w->buf + got, n - got);
        if (r <= 0) break;
        got += (size_t)r;
    }
    pos_ = start + (uint32_t)got;
    w->off = start;
    w->len = (uint32_t)got;
    card_reads++;
    card_bytes += got;
    return got == n ? w : nullptr;
}

bool FileReader::read(uint32_t offset, void* dst, size_t len) {
    if (fd_ < 0 || (uint64_t)offset + len > size_) return false;
    uint64_t t0 = nr_micros();
    uint8_t* out = (uint8_t*)dst;
    size_t left = len;
    uint32_t off = offset;
    while (left) {
        Way* w = nullptr;
        for (int i = 0; i < 2; i++)
            if (off >= way_[i].off && off < way_[i].off + way_[i].len) w = &way_[i];
        if (!w && !(w = fill(off))) return false;
        w->used = ++tick_;
        uint32_t k = w->off + w->len - off;
        if (k > left) k = (uint32_t)left;
        memcpy(out, w->buf + (off - w->off), k);
        out += k;
        off += k;
        left -= k;
    }
    read_us += nr_micros() - t0;
    bytes_read += len;
    reads++;
    return true;
}

// A tile of rows: served from a cached block when it is there, else one
// sector-aligned read of the covering sectors straight into buf.
const uint8_t* FileReader::view(uint32_t offset, uint32_t len, uint8_t* buf, uint32_t cap) {
    if (fd_ < 0 || (uint64_t)offset + len > size_) return nullptr;
    for (int i = 0; i < 2; i++)
        if (offset >= way_[i].off && offset + len <= way_[i].off + way_[i].len) {
            way_[i].used = ++tick_;
            bytes_read += len;
            reads++;
            return way_[i].buf + (offset - way_[i].off);
        }
    const uint32_t start = offset & ~(SECTOR - 1);
    uint32_t end = (offset + len + SECTOR - 1) & ~(SECTOR - 1);
    if (end > size_) end = size_;
    if (end - start > cap || ((uintptr_t)buf & 3)) return TensorReader::view(offset, len, buf, cap);
    uint64_t t0 = nr_micros();
    if (start != pos_ && ::lseek(fd_, start, SEEK_SET) != (off_t)start) return nullptr;
    uint32_t n = end - start, got = 0;
    while (got < n) {
        ssize_t r = ::read(fd_, buf + got, n - got);
        if (r <= 0) return nullptr;
        got += (uint32_t)r;
    }
    pos_ = end;
    card_reads++;
    card_bytes += n;
    read_us += nr_micros() - t0;
    bytes_read += len;
    reads++;
    return buf + (offset - start);
}

bool OverlayReader::add(uint32_t start, uint32_t len, TensorReader* src, uint32_t src_offset) {
    if (n_ >= MAX_RANGES) return false;
    int i = n_++;  // keep sorted by start (insertion; ranges do not overlap)
    while (i > 0 && ranges_[i - 1].start > start) { ranges_[i] = ranges_[i - 1]; i--; }
    ranges_[i] = {start, len, src_offset, src};
    return true;
}

// Index of the last range starting at or before offset, or -1.
int OverlayReader::find(uint32_t offset) const {
    int lo = 0, hi = n_ - 1, best = -1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (ranges_[mid].start <= offset) { best = mid; lo = mid + 1; }
        else hi = mid - 1;
    }
    return best;
}

bool OverlayReader::read(uint32_t offset, void* dst, size_t len) {
    uint8_t* out = (uint8_t*)dst;
    uint64_t t0 = nr_micros();
    bytes_read += len;
    reads++;
    while (len) {
        const int i = find(offset);
        const Range* hit = (i >= 0 && offset < ranges_[i].start + ranges_[i].len) ? &ranges_[i] : nullptr;
        const uint32_t next = i + 1 < n_ ? ranges_[i + 1].start : 0xFFFFFFFFu;
        size_t chunk;
        bool ok;
        if (hit) {
            chunk = hit->start + hit->len - offset;
            if (chunk > len) chunk = len;
            ok = hit->src->read(hit->src_off + (offset - hit->start), out, chunk);
            overlay_bytes += chunk;
        } else {
            chunk = len;
            if ((uint64_t)offset + chunk > next) chunk = next - offset;
            ok = base_->read(offset, out, chunk);
        }
        if (!ok) return false;
        out += chunk;
        offset += (uint32_t)chunk;
        len -= chunk;
    }
    read_us += nr_micros() - t0;
    return true;
}

const uint8_t* OverlayReader::view(uint32_t offset, uint32_t len, uint8_t* buf, uint32_t cap) {
    const int i = find(offset);
    bytes_read += len;
    reads++;
    if (i >= 0 && offset + len <= ranges_[i].start + ranges_[i].len) {
        const Range& r = ranges_[i];
        overlay_bytes += len;
        return r.src->view(r.src_off + (offset - r.start), len, buf, cap);
    }
    const bool touches = (i >= 0 && offset < ranges_[i].start + ranges_[i].len) ||
                         (i + 1 < n_ && offset + len > ranges_[i + 1].start);
    if (touches) return TensorReader::view(offset, len, buf, cap);  // straddles: copy path
    return base_->view(offset, len, buf, cap);
}

// Header: 48 u32 + one f32 (rope_theta); then codebook_len f32; then records.
bool Cact::parse(TensorReader* r) {
    uint32_t raw[49];
    if (!r->read(0, raw, sizeof(raw))) return false;
    memcpy(&h, raw, sizeof(raw));
    static_assert(sizeof(CactHeader) == 196, "header layout");
    if (h.tag != CACT_TAG) {
        nr_log("cact: bad tag %08x (Needle 2 is 0x05E12A83)\n", (unsigned)h.tag);
        return false;
    }
    if (h.codebook_len != 28) return false;
    if (!r->read(196, codebook, sizeof(codebook))) return false;
    n_recs = h.num_tensors;
    recs = (CactRec*)nr_alloc(sizeof(CactRec) * n_recs, "cact directory");
    if (!recs) return false;
    uint32_t off = 196 + h.codebook_len * 4;
    uint8_t rec[44];
    for (uint32_t i = 0; i < n_recs; i++, off += 44) {
        if (!r->read(off, rec, 44)) return false;
        CactRec& t = recs[i];
        t.dtype = rec[0];
        t.ndim = rec[1];
        memcpy(t.shape, rec + 4, 16);
        uint64_t o, n;
        memcpy(&o, rec + 20, 8);
        memcpy(&n, rec + 28, 8);
        uint32_t group, bits;
        memcpy(&group, rec + 36, 4);
        memcpy(&bits, rec + 40, 4);
        t.offset = (uint32_t)o;
        t.nbytes = (uint32_t)n;
        t.group = (uint16_t)group;
        t.bits = (uint8_t)bits;
    }
    return true;
}

void Cact::release() {
    nr_free(recs);
    recs = nullptr;
    n_recs = 0;
}
