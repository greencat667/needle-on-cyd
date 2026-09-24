#!/usr/bin/env python3
"""Lay out the SD card and the flash overlay for one Needle archive.

  sdcard/needle/needle3.cact   the archive, unchanged
  sdcard/needle/label.txt      what the screen calls the model
  sdcard/needle/tools.json     the minified tool list (the firmware has a default)
  build/needle_overlay.bin     partition image for `needle` (0x90000): a
                               header, then verbatim copies of the archive's
                               embedding, tokenizer, FP16 vectors and mHC maps

The firmware builds its tokenizer index and prefix snapshot on the device at
first boot and keeps them on the card; nothing here is computed for it.

    python tools/prepare_sd.py models/needle3-L2.cact --label "2 layers, base" \
        [--card /Volumes/NEEDLE] [--image build/sd.img]
Then flash the overlay (the app itself is flashed by idf.py):
    python -m esptool --chip esp32 write_flash 0x90000 build/needle_overlay.bin
"""
import argparse, os, shutil, struct, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
HDR, REC = "<48If", "<BBHIIIIQQII"
PART_SIZE = 0x370000
PART_OFFSET = 0x90000
MAGIC = 0x464C444E  # 'NDLF'


def directory(path):
    raw = open(path, "rb").read()
    h = struct.unpack_from(HDR, raw, 0)
    assert h[0] == 0x05E12A84, "not a Needle 3 archive"
    off = 196 + h[2] * 4
    recs = []
    for _ in range(h[1]):
        recs.append(struct.unpack_from(REC, raw, off))
        off += 44
    fnv = 2166136261
    for b in raw[:196 + h[2] * 4 + h[1] * 44]:
        fnv = ((fnv ^ b) * 16777619) & 0xFFFFFFFF
    return raw, h, recs, fnv


def overlay_tensors(recs, n_layers):
    """Which tensors go to flash: the embedding, the tokenizer, every
    non-CQ tensor (norms, taps, diagonals, Monarch factors, gates, perms) and
    the mHC maps; i.e. everything except the large CQ matrices."""
    mhc0 = 1 + 27 * n_layers
    pick = [0, len(recs) - 1]
    for i, r in enumerate(recs[1:-1], start=1):
        if r[0] != 3 or mhc0 <= i < mhc0 + 9:
            pick.append(i)
    return sorted(set(pick))


def overlay(raw, recs, dir_hash, n_layers):
    idx = overlay_tensors(recs, n_layers)
    ranges = [(recs[i][7], recs[i][8]) for i in idx]
    table = []
    pos = (20 + 12 * len(ranges) + 1023) & ~1023  # header area
    for start, n in ranges:
        pos = (pos + 63) & ~63
        table.append((start, n, pos))
        pos += n
    if pos > PART_SIZE:
        raise SystemExit(f"overlay {pos} B does not fit the {PART_SIZE} B partition")
    head = struct.pack("<5I", MAGIC, 1, len(raw), dir_hash, len(table))
    for t in table:
        head += struct.pack("<3I", *t)
    data = bytearray(head)
    for (start, n), (_, _, p) in zip(ranges, table):
        data += b"\xff" * (p - len(data))
        data += raw[start:start + n]
    return bytes(data), table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--label", default=None)
    ap.add_argument("--profile", choices=("creature", "needle"), default="creature",
                    help="creature: the autonomous pet (/creature); needle: the four-action tap-to-set demo (/needle)")
    ap.add_argument("--tools", default=None, help="tool list (default: the profile's own)")
    ap.add_argument("--card", default=os.path.join(ROOT, "sdcard"), help="SD card root (a mounted card or a folder)")
    ap.add_argument("--image", default=None, help="also build a FAT image of the card (QEMU)")
    ap.add_argument("--image-mb", type=int, default=32)
    a = ap.parse_args()

    raw, h, recs, dir_hash = directory(a.model)
    layers = h[10]
    d = os.path.join(a.card, a.profile)
    os.makedirs(d, exist_ok=True)
    tools = a.tools or os.path.join(ROOT, "bench", "creature_tools.min.json" if a.profile == "creature" else "tools.min.json")
    for stale in ("tok.idx", "prefix.snap"):  # built by the device for its archive
        p = os.path.join(d, stale)
        if os.path.exists(p):
            os.remove(p)
    shutil.copyfile(a.model, os.path.join(d, "needle3.cact"))
    label = a.label or f"{layers} layers"
    open(os.path.join(d, "label.txt"), "w").write(label + "\n")
    shutil.copyfile(tools, os.path.join(d, "tools.json"))
    open(os.path.join(a.card, "profile.txt"), "w").write(a.profile + "\n")  # the firmware boots into this one
    blob, table = overlay(raw, recs, dir_hash, layers)
    os.makedirs(os.path.join(ROOT, "build"), exist_ok=True)
    out = os.path.join(ROOT, "build", "needle_overlay.bin")
    open(out, "wb").write(blob)
    print(f"card     {d}: profile '{a.profile}', needle3.cact {len(raw):,} B ({layers} layers), label '{label}'")
    print(f"overlay  {out}: {len(blob):,} B, archive hash {dir_hash:08x}")
    print(f"         {len(table)} tensors, {sum(t[1] for t in table):,} B")
    if a.image:  # raw FAT16 image of the card for QEMU's sd-card (mtools)
        if not shutil.which("mformat"):
            raise SystemExit("needs mtools (brew install mtools)")
        with open(a.image, "wb") as f:
            f.truncate(a.image_mb * 1024 * 1024)
        subprocess.check_call(["mformat", "-i", a.image, "-F" if a.image_mb > 2048 else "-v", "NEEDLE", "::"])
        subprocess.check_call(["mmd", "-i", a.image, f"::/{a.profile}"])
        for name in sorted(os.listdir(d)):
            subprocess.check_call(["mcopy", "-i", a.image, os.path.join(d, name), f"::/{a.profile}/{name}"])
        subprocess.check_call(["mcopy", "-i", a.image, os.path.join(a.card, "profile.txt"), "::/profile.txt"])
        print(f"image    {a.image}: {a.image_mb} MB FAT")
    print("flash    python -m esptool --chip esp32 write_flash 0x90000 build/needle_overlay.bin")


if __name__ == "__main__":
    main()
