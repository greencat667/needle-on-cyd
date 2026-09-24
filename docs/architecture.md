# Architecture

```
 touch / serial                      ESP32 (stock CYD, 240 MHz, 520 KB SRAM, no PSRAM)
      │
      ▼
 app (main.cpp) ── creature state ─► state_text()  "I'm hungry, tired and bored."
      │                                   │
      │                                   ▼
      │                       NeedleSession::run           (inference.cpp)
      │                         1. restore prefix snapshot (SD)  ◄── built on this board, first boot
      │                         2. tokenise turn (BPE from the archive's own tokenizer)
      │                         3. forward the turn, force <think>
      │                         4. greedy reasoning, full-vocab head, budget 12
      │                         5. force </think>\n<tool_call>
      │                         6. call under the byte grammar (+ jump-forward)
      │                         7. confidence = min(head, p(call))
      │                                   │
      │                                   ▼
      │                         Model::forward (model.cpp), one token, 2 blocks:
      │                           embedding gather ─ engram hash/gather ─ mHC gates
      │                           ─ attention (conv taps, RoPE, int8 KV) ─ Hadamard MLP
      │                           ─ Sinkhorn lane mix ─ probe pool (confidence)
      │                                   │  cq_matmul / cq_row (tensor.cpp)
      │                                   ▼
      │                         OverlayReader ── embedding + tokenizer ─► flash partition `needle`
      │                                   └─────── everything else ─────► FileReader (SD, POSIX)
      ▼
 ui (ILI9341 regions, 4-bit AA fonts, no framebuffer)  +  RGB LED
```

## Storage

- `.cact` stays unmodified on the card at `/needle/needle3.cact`.
- `tools/prepare_sd.py` copies the archive's embedding (3,244,032 B) and
  tokenizer (109,140 B) byte for byte into the 3.25 MB `needle` flash
  partition, behind a header naming the archive (size + FNV hash of header
  and directory). At boot the firmware checks the header and spot-checks both
  ends of each range against the card. On any mismatch it reads everything
  from SD.
- `TensorReader` is the only storage interface. Backends: `FileReader` (POSIX
  file; sector-aligned transfers, 2 × 4 KB LRU block cache, zero-copy
  `view()` of whole row tiles), `PartitionReader` (esp_partition_read),
  `OverlayReader` (routes byte ranges), `OffsetReader`.
- Files the device creates on the card: `tok.idx` (the tokenizer surface
  hash, 32 KB) and `prefix.snap` (KV cache, conv and engram histories and
  probe pool after the tool prefix). Both are keyed to the archive and the
  tool list, and rebuilt if either changes.

## Streaming

- One layer's weights are touched in order, tile by tile (8 KB of packed rows
  plus their norms), and dropped. No weight is ever resident.
- CQ kernel: the activation is int8 fake-quantised (as the reference does),
  then each 128-group is rotated with a fast Walsh–Hadamard transform. For
  each weight row, activations are summed into buckets by index (4 buckets at
  2 bits, 16 at 4 bits), then one short dot with the codebook and a
  multiply by the group norm. No float weights are formed.
- The output head is computed only for the rows the grammar allows during
  the call, and in full (streamed, logits consumed tile by tile) during
  reasoning.

## Faithfulness checks

| layer | check | result |
|---|---|---|
| NumPy oracle vs JAX reference (`tools/check_ref.py`) | all-position logits, confidence logit | cosine ≥ 0.999998, top-1 100%, Δconf 0.004 |
| C runtime vs oracle (`tools/check_host.py`, `tests/test_golden.py`) | tokenizer ids, logits, reasoning text, confidence | identical ids; cosine ≥ 0.99999; top-1 100%; identical 12-token reasoning; Δconf < 1e-4 |
| probe pool vs engine | `needle_embed` of the shipped engine vs mine | cosine 0.99999 |
| C runtime vs engine | "I need a rest" on the 2-bit L2 rung | identical reasoning text and tool call |
| board vs host | same prompt, same model | identical call and reasoning (see README) |

The engine's `confidence` number is not reproduced. The shipped engine
re-scores an id sequence built inside `generate_turn` that could not be
identified from the outside. The port reports Needle's documented definition
instead.
