# Benchmark results (24 Sept 2026)

Four tools (eat / sleep / explore / rest), 24 clear-cut states (`bench/cases.json`),
rendered to text by the device's template ("I'm not hungry, tired and bored.").
Accepted answers allow sleep or rest when energy is very low.

## Shipped engine on the Mac (reference), base Needle 3

| model | size | "words" template | "numbers" | "scores" | engine peak RAM |
|---|---:|---:|---:|---:|---:|
| L2, 2-bit (sliced) | 8.6 MB | 12/24 | 11/24 | 12/24 | 25.9 MB |
| L4, 2-bit | 9.8 MB | 6/24 | 9/24 | 10/24 | 28.8 MB |
| L8, 2-bit | 16.2 MB | 12/24 | 12/24 | 13/24 | 44.4 MB |
| L20, 2-bit (shipped) | 35.3 MB | 8/24 | 12/24 | 9/24 | 91.5 MB |

Base Needle maps explicit requests to tools ("I'm hungry" → eat works at 20
layers); it was not trained to infer a decision from a state description, at
any depth. Engine latency on the Mac: 60–160 ms per decision.

## Fine-tuned rung (Needle's own LoRA loop, applied to the 2-layer rung)

`models/needle3-L2-creature-bal.cact`: 2 layers, CQ4 (local fine-tuning is
4-bit), 13.3 MB, no confidence head (Needle drops it for local tunes). 2,000
class-balanced sentences from six paraphrase families. The device template
and the 24 benchmark states were held out of training.

| runner | words (device template) | numbers | scores |
|---|---:|---:|---:|
| shipped engine, Mac | 24/24 | 13/24 | 6/24 |
| this runtime, Mac host build | 24/24 | – | – |
| **this runtime, on the CYD** | **24/24** | – | – |

On the CYD all 24 results were identical to the host build: same call, same
reasoning text (`bench/host_tuned_24.json` vs `device_bench_tuned_2026-09-24.log`).

Board figures over the 24 decisions: 66.9–80.3 s (mean 71.0 s); 8–10
reasoning tokens; 130 MB read per decision (about 99 MB SD, 31 MB flash);
runtime heap 197,916 B; minimum free heap 13,692 B.

## 2-bit fine-tune, after the speed work (24 Sept, evening)

`models/needle3-L2-creature-2bit-r64.cact` (8.6 MB): LoRA (rank 64) trained on
the shipped 2-bit rung through 2-bit numerics. Every tensor except the 10
attention matrices is byte-identical to the official 2-bit L2 slice; those
10 are re-packed at CQ2. No confidence head.

| runner | words (device template) | held-out training-style sentences |
|---|---:|---:|
| shipped engine, Mac | 24/24 | – |
| this runtime, Mac host | 24/24 | 191/200 (package's scorer) |
| **this runtime, on the CYD** | **24/24**, all identical to the host | – |

Board: **mean 37.2 s per decision** (34.6–41.6 s); 88 MB read per decision
(about 48 MB SD, 40 MB memory-mapped flash); runtime heap 195,012 B;
minimum free heap 10,436 B. Raw log: `device_bench_2bit_r64_2026-09-24.log`.

| step | per decision | note |
|---|---:|---|
| first working run (base 2-bit, stdio reads) | 160 s | |
| POSIX aligned reads, zero-copy tiles | 55 s | base 2-bit |
| tuned 4-bit model (more SD bytes) | 71 s | 24/24 |
| tuned 2-bit + LUT kernels + mapped flash overlay | **37 s** | 24/24 |
