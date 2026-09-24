# Memory budget

Every runtime allocation goes through `nr_alloc` (`firmware/needle_runtime/nr_platform.cpp`),
which logs it. The table lists the allocations at the board's configuration:
2 layers, context 161 positions (107-token tool prefix + up to 24 turn tokens +
12 reasoning + 3 forced + 14 call), one token per forward pass
(`NR_MAX_CHUNK=1`). Host: `NR_LOG_ALLOC=1 host/needle_host ... --ctx 161`.

| buffer | bytes | notes |
|---|---:|---|
| KV cache V (int8) | 41,216 | 161 × 2 kv heads × 64 × 2 layers |
| KV cache K (int8) | 30,912 | 161 × 2 × 48 × 2 |
| KV scales (f32) | 5,152 | one per head per position |
| probe pool accumulator | 36,864 | confidence head, streaming softmax over positions; **in IRAM**; base model only |
| engram value history (fp16) | 15,360 | ring of 10 positions (dilated conv taps at t, t-3, t-6, t-9) |
| q/k/v conv history (f32) | 12,800 | 2 previous positions per layer; kept f32 because fp16 flipped a top-1 (checked against the oracle) |
| residual lanes | 12,288 | 4 lanes × 768 f32 |
| rotated activation | 12,288 | the Hadamard-rotated input of the next matmul (up to 3072 wide) |
| Hadamard MLP scratch | 12,288 | z, cond, and the two 32×32 factors as raw fp16 |
| block I/O | 9,216 | u0, u, h |
| CQ tile buffer | 8,192 | one row tile of the current matrix, filled by one aligned SD read |
| SD block cache | 8,192 | two 4 KB ways; matrix norms, FP16 vectors |
| param stream / perms / q,k,v, gates, engram k/v, misc | ~27,000 | |
| **runtime total** | **≈198 KB DRAM** (+37 KB IRAM for the base model) | measured on the board: `runtime live 197,916 B` (tuned) |

Transient, before the model loads: the tokenizer's surface index (32 KB) is
built once on the device, saved to the SD card and freed. Lookups then read
2 bytes per probe from the card.

Not in the heap: 55 KB static DRAM (IDF, FATFS, drivers, a 5 KB LCD strip,
the kernels' staging arrays) and the 12 KB main-task stack.

## What did not fit, and what was done about it

| attempt | result | fix |
|---|---|---|
| 4-token prefill chunks | OOM: residual lanes 49 KB + rotated activation 49 KB | one token per pass (weights re-read per token) |
| chunk 1, fp32 everything, 183-position cache | OOM by 20 KB | context sized to the prompt; MLP factors as raw fp16; engram history fp16 |
| probe pool in DRAM | OOM by 16 KB | 37 KB into the 68 KB of spare IRAM, touched only via 32-bit integer loads and stores |

Heap left at the ready screen: 13 KB (tuned model, no head), 30–33 KB (base
model, head in IRAM).
