#!/usr/bin/env python3
"""Slice the shipped 2-bit needle3.cact to an N-layer ladder rung, byte-exact.

`needle build --layers N` re-quantises the half-precision master to 4 bits.
The shipped archive is the 2-bit post-trained model, so this tool keeps its
bytes as they are and only drops the blocks the ladder rule does not select:
the same selection `ladder_slice` / `ladder_config` make in
needle/model/architecture.py, applied to the positional tensor order that tag
0x05E12A84 promises ("Porting Needle 3", docs/upstream/SOURCES.md).

No weight is decoded or re-encoded. Every surviving tensor blob is copied
verbatim, except row subsets of the layer-stacked tensors (mHC, probe-head
rows), which are copied row by row, still without touching any value.

    python tools/slice_cact.py models/needle3.cact 2 models/needle3-L2.cact
"""
import struct
import sys

TAG = 0x05E12A84
HDR = "<48If"
REC = "<BBHIIIIQQII"
REC_SIZE = struct.calcsize(REC)
FP16, FP32, CQ, RAW = 1, 2, 3, 4
PER_LAYER = 27
ALIGN = 64


def ladder_order(n):
    """Endpoint-preserving bisection (architecture._ladder_layer_order)."""
    if n == 1:
        return (0,)
    selected = [0, n - 1]
    order = list(selected)
    while len(order) < n:
        selected.sort()
        _gap, left, right = max(
            ((r - l, l, r) for l, r in zip(selected, selected[1:]) if r - l > 1),
            key=lambda item: (item[0], -item[1]))
        c = (left + right) // 2
        selected.append(c)
        order.append(c)
    return tuple(order)


def ladder_layers(n, depth):
    return tuple(sorted(ladder_order(n)[:depth]))


class T:
    def __init__(self, dtype, shape, blob, group=0, bits=0):
        self.dtype, self.shape, self.blob = dtype, tuple(shape), blob
        self.group, self.bits = group, bits


def read(path):
    raw = open(path, "rb").read()
    hdr = list(struct.unpack_from(HDR, raw, 0))
    assert hdr[0] == TAG, "not a Needle 3 archive"
    ncb = hdr[2]
    cb = raw[196:196 + ncb * 4]
    off = 196 + ncb * 4
    ts = []
    for _ in range(hdr[1]):
        r = struct.unpack_from(REC, raw, off)
        off += REC_SIZE
        dtype, ndim = r[0], r[1]
        ts.append(T(dtype, r[3:3 + ndim], raw[r[7]:r[7] + r[8]], r[9], r[10]))
    return hdr, cb, ts


def elem_rows(t, rows, axis=0):
    """Row subset of an FP16/FP32 tensor along `axis`, bytes copied verbatim."""
    esz = 2 if t.dtype == FP16 else 4
    shape = list(t.shape)
    outer = 1
    for s in shape[:axis]:
        outer *= s
    inner = esz
    for s in shape[axis + 1:]:
        inner *= s
    n = shape[axis]
    out = bytearray()
    for o in range(outer):
        base = o * n * inner
        for r in rows:
            out += t.blob[base + r * inner: base + (r + 1) * inner]
    shape[axis] = len(rows)
    return T(t.dtype, shape, bytes(out))


def cq_rows(t, rows):
    """Row subset of a CQ matrix: packed index rows, then their norm rows."""
    out_rows, in_dim = t.shape
    in_pad = (in_dim + t.group - 1) // t.group * t.group
    prow = in_pad * t.bits // 8
    nrow = in_pad // t.group * 2
    packed = t.blob[:out_rows * prow]
    norms = t.blob[out_rows * prow:]
    assert len(norms) == out_rows * nrow
    b = b"".join(packed[r * prow:(r + 1) * prow] for r in rows)
    b += b"".join(norms[r * nrow:(r + 1) * nrow] for r in rows)
    return T(CQ, (len(rows), in_dim), b, t.group, t.bits)


def slice_archive(hdr, ts, depth):
    L = hdr[10]
    lanes = hdr[15]
    sel = ladder_layers(L, depth)
    remap = {l: i for i, l in enumerate(sel)}
    n_sites = hdr[31]
    sites = hdr[32:32 + n_sites]

    i = 0
    out = [ts[i]]; i += 1                                  # embedding
    for layer in range(L):
        block = ts[i:i + PER_LAYER]; i += PER_LAYER
        if layer in remap:
            out += block
    mhc = ts[i:i + 9]; i += 9
    for t in mhc[:6]:                                      # a_*, b_* (L, ...)
        out.append(elem_rows(t, sel, 0))
    phi_rows = [l * lanes + k for l in sel for k in range(lanes)]
    out.append(cq_rows(mhc[6], phi_rows))                  # phi_pre  (L*lanes)
    out.append(cq_rows(mhc[7], phi_rows))                  # phi_post (L*lanes)
    res_rows = [l * lanes * lanes + k for l in sel for k in range(lanes * lanes)]
    out.append(cq_rows(mhc[8], res_rows))                  # phi_res  (L*lanes^2)
    out += ts[i:i + 2]; i += 2                             # hada_p1, hada_p2
    new_sites = []
    for s in sites:
        site = ts[i:i + 4]; i += 4
        if s in remap:
            out += site
            new_sites.append(remap[s])
    out.append(ts[i]); i += 1                              # final_norm

    extra = len(ts) - i - 1
    if extra:
        manifest = ts[i]; i += 1
        out.append(manifest)
        codes = struct.unpack(f"<{manifest.shape[0]}e", manifest.blob)
        cell_rows = [0] + [l + 1 for l in sel]             # embedding, blocks
        for code in codes:
            probes, gain, query, row_bias, proj, bias = ts[i:i + 6]; i += 6
            k = gain.shape[1]
            out.append(cq_rows(probes, [r * k + j for r in cell_rows for j in range(k)]))
            out.append(elem_rows(gain, cell_rows, 0))
            out.append(query)
            out.append(elem_rows(row_bias, cell_rows, 1))
            out += [proj, bias]
            if int(code) == 3:                             # router calibration
                out.append(ts[i]); i += 1
    assert ts[i].dtype == RAW and i == len(ts) - 1
    out.append(ts[i])

    hdr = list(hdr)
    hdr[1] = len(out)
    hdr[10] = depth
    gmask = hdr[17] | (hdr[18] << 32)
    g = 0
    for l in range(L):
        if gmask >> l & 1 and l in remap:
            g |= 1 << remap[l]
    hdr[17], hdr[18] = g & 0xFFFFFFFF, g >> 32
    hdr[31] = len(new_sites)
    hdr[32:48] = (new_sites + [0] * 16)[:16]
    return hdr, out, sel


def write(path, hdr, cb, ts):
    head = struct.pack(HDR, *hdr) + cb
    pos = len(head) + len(ts) * REC_SIZE
    offs = []
    for t in ts:
        pos = (pos + ALIGN - 1) & ~(ALIGN - 1)
        offs.append(pos)
        pos += len(t.blob)
    d = b"".join(struct.pack(REC, t.dtype, len(t.shape), 0,
                             *(list(t.shape) + [0, 0, 0, 0])[:4], o, len(t.blob),
                             t.group, t.bits) for t, o in zip(ts, offs))
    buf = bytearray(head + d)
    for t, o in zip(ts, offs):
        buf += b"\x00" * (o - len(buf))
        buf += t.blob
    open(path, "wb").write(buf)
    return len(buf)


def main():
    src, depth, dst = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    hdr, cb, ts = read(src)
    hdr2, out, sel = slice_archive(hdr, ts, depth)
    n = write(dst, hdr2, cb, out)
    print(f"{dst}: blocks {sel}, {len(out)} tensors, {n:,} bytes, "
          f"engram sites {hdr2[32:32 + hdr2[31]]}")


if __name__ == "__main__":
    main()
