# Feasibility: Needle 3 on a stock Cheap Yellow Display

Target: ESP32-2432S028R. ESP32-D0WD-V3 (rev 3.1), 2 × Xtensa LX6 at 240 MHz,
520 KB SRAM, 4 MB flash, **no PSRAM**, ILI9341 320×240, XPT2046 touch, microSD.

Every number below was read from the archives' own directories or measured,
on the Mac or on the board itself. Where a figure is an estimate it says so.

## 1. What Needle 3 is, as far as a port cares

Source of truth: `vendor/needle` (commit in `docs/upstream/needle-commit.txt`; links in `docs/upstream/SOURCES.md`),
`needle/model/architecture.py` (the reference model), `needle/model/export.py`
(the `.cact` writer), and the Cactus posts listed in `docs/upstream/SOURCES.md`.

- Laddered Simple Attention Network: width 768, 12 query heads over 2 KV heads
  (q/k 48, v 64), causal 3-tap convs on q/k/v, a Monarch "Hadamard MLP"
  (32×32 Kronecker factors, n = 1024) instead of an FFN, 4 multi-lane (mHC)
  residual lanes mixed by a Sinkhorn matrix, and **engram** n-gram tables read
  by hash gather at layers 3/7/11/15/19. Vocabulary 8192; the output head is
  the tied embedding.
- Every depth 2–20 is a trained model (endpoint-preserving bisection:
  depth 2 = blocks 0 and 19).
- Weights are Cactus Quants: groups of 128 along each row, rotated by a
  Walsh–Hadamard matrix, stored as an fp16 norm plus 2-bit (CQ2) or 4-bit
  (CQ4) Lloyd–Max indices. A kernel rotates the activation once per group
  and never forms a float weight.
- The shipped `needle3.cact` is the 2-bit post-trained model. Locally,
  `needle build` can only write 4-bit (`_cq_pack` refuses other widths).

## 2. The questions in the brief

1. **Smallest exportable network.** 2 layers. `needle build --layers 2` gives a
   4-bit 13.34 MB file. The official 2-bit numerics at depth 2 are obtained by
   slicing the shipped archive byte for byte (`tools/slice_cact.py`: the
   depth-20 slice is identical to the shipped file; depth-2/4/8 headers and
   tensor shapes equal `needle build`'s): **8,606,036 bytes, 79 tensors,
   25.4 M parameters** (14.2 M of them in the engram tables).
2. **Is `--layers 2` supported?** Yes by `needle build`. The engine's
   `--depth N` flag is accepted but **ignored**: identical output, confidence
   and memory at depths 2, 4, 8 and 20. The only way to run a small rung on
   the shipped engine is a sliced file.
3. **Exact size of the weights** (bytes, from the directories):

   | component | L2 2-bit (base) | L2 4-bit (tuned) | L4 2-bit | L8 2-bit | L20 2-bit |
   |---|---:|---:|---:|---:|---:|
   | embedding / output head (CQ4) | 3,244,032 | 3,244,032 | 3,244,032 | 3,244,032 | 3,244,032 |
   | blocks | 1,074,692 | 1,971,716 | 2,149,384 | 4,298,768 | 10,746,920 |
   | mHC + Hadamard perms | 84,332 | 84,332 | 160,472 | 312,752 | 769,592 |
   | engram tables | 3,760,128 | 7,299,072 | 3,760,128 | 7,520,256 | 18,800,640 |
   | engram projections + taps | 319,488 | 614,400 | 319,488 | 638,976 | 1,597,440 |
   | final norm + confidence head | 9,580 | 1,536 | 12,828 | 19,324 | 38,812 |
   | tokenizer (RAW) | 109,140 | 109,140 | 109,140 | 109,140 | 109,140 |
   | **file** | **8,606,036** | **13,328,276** | 9,762,644 | 16,155,796 | 35,335,380 |
   | parameters | 25.4 M | 25.4 M | 29.3 M | 52.2 M | 121.0 M |

4.–5. **RAM during inference, by part**: see `memory-budget.md`. The runtime
   holds about 190 KB at peak on the board (the KV cache is the largest item).
6. **Memory-mapped or streamed?** Streamed. The format is positional,
   layer-major and 64-byte aligned, and every matrix is `[out, in]`, so a row
   tile of any matrix is one contiguous read. Nothing needs mapping.
7. **Individual tensors from offsets?** Yes: the directory gives absolute
   offsets, and CQ rows split cleanly (packed indices, then per-row norms).
8. **Hot path** (per token, 2 layers): five attention projections per layer
   (1.79 M weights), the engram key/value projections (1.18 M), the mHC gate
   matrices, the Monarch MLP (small), attention over the cache, and the
   output head (8192 × 768 = 6.3 M weights) on every decoding step. Reading
   the weights costs more than computing with them.
9. **Reusable C/C++?** No. The engine is a closed binary (`libneedle.a`,
   arm64/x86 only). The runtime here is new C++ written against the Python
   reference and the format documentation.
10. **Architecture-specific assumptions.** The shipped kernels rely on NEON
    `sdot`. The format itself is little-endian and portable. ESP32-specific
    traps found on the way: IRAM only allows 32-bit integer loads and stores
    (FPU loads there fault), and stdio `fread` from the SD card is about 20×
    slower than POSIX `read`.

## 3. Memory budget (stock CYD, measured on the board)

| Component | Flash / SD storage | Runtime RAM |
|---|---:|---:|
| runtime code | 430 KB app image (512 KB partition) | ~60 KB static DRAM (.data + .bss, incl. 5 KB LCD strip, 4 KB kernel table) |
| tokenizer | 109 KB (in the flash partition) + 32 KB index on SD | 2 KB coarse offset index |
| model weights | 8.6 MB on SD; a verbatim copy of the embedding, tokenizer, every FP16 vector and the mHC maps (59 tensors, 3.57 MB) in the memory-mapped `needle` flash partition | 8 KB tile buffer + 8 KB SD block cache; weights are never held |
| activations | – | ~55 KB (lanes, rotated activation, block I/O, q/k/v, MLP, gates) |
| KV cache + conv/engram state | prefix snapshot on SD (built on device) | 77 KB KV (161 positions × 2 layers, int8 + scales) + 28 KB histories |
| confidence head pool | – | 37 KB, placed in spare IRAM (base model only) |
| grammar | – | 1.5 KB candidate list (static) |
| display / UI | 34 KB fonts in flash | 5 KB strip buffer (no framebuffer) |
| stack / heap reserve | – | 12 KB main task stack; 8–10 KB heap left (tuned 2-bit) |
| **TOTAL** | **8.6 MB SD + 3.57 MB flash overlay + 0.43 MB app** | **~196 KB heap at peak + ~60 KB static (+37 KB IRAM for the base model's head)** |

Boot heap on this board: 249 KB free with a largest block of 118 KB, split
across 5 regions. Every runtime buffer is ≤ 41 KB, so fragmentation is not a
problem; the total is.

## 4. Verdict

**Possible, and running.** See `../README.md` for the full result. The
blocking resource was never compute (about 0.2 GMAC per decision). It was
SRAM for the per-position state, then SD throughput. The remaining
bottleneck is SD bytes per token.
