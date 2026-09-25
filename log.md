# 080 - Needle on CYD: log

> Append-only, newest at the bottom. Goal: Cactus Needle 3 running entirely on a stock ESP32-2432S028R ("Cheap Yellow Display"), no PSRAM, weights on microSD. Full brief in the first session's prompt; repo is a local git repo (`git log`).

## 2026-09-24 — first session: study, port, and working on the real board

**Done**
- Studied Needle 3 (vendor/needle @ `docs/upstream/needle-commit.txt`, plus the cact-format / porting / confidence blog posts saved in `docs/upstream/`). Smallest rung is 2 layers (blocks 0 and 19, one engram site).
- `tools/slice_cact.py`: byte-exact ladder slicer of the shipped **2-bit** archive (the depth-20 slice is byte-identical to the shipped file; headers match `needle build`). 2-layer 2-bit = **8.6 MB**.
- Found: the engine CLI's `--depth` flag is silently ignored (identical outputs at every depth). `needle build --layers N` re-quantises from the master to **4-bit** (L2 = 13.3 MB).
- NumPy token-at-a-time oracle (`tools/ref_forward.py`) pinned to the JAX reference: logits cosine ≥ 0.999998, top-1 100%, confidence-head logit matches.
- Portable C++ runtime (`firmware/needle_runtime/`): cact reader with replaceable storage, CQ2/CQ4 kernels that never materialise weights, streamed layers, engram, mHC lanes, Hadamard MLP, int8 KV, streaming probe-pool confidence head, SentencePiece BPE from the archive's RAW blob, byte-level tool-call grammar, jump-forward. Host build (`host/`) matches the oracle: cosine ≥ 0.99999, top-1 100%, same tokenizer ids; for "I need a rest" it reproduces the engine's reasoning text token for token.
- Benchmark (`bench/cases.json`, 24 states, 4 tools): base Needle at every depth scores 25–54%, **including the full 20-layer model**; it maps explicit requests to tools, not state readouts to decisions. Fine-tuned the 2-layer rung itself with Needle's own LoRA loop (`tools/finetune_rung.py`, class-balanced synthetic data, device template held out): **24/24** on engine and on the port (`models/needle3-L2-creature-bal.cact`, 4-bit, no confidence head).
- ESP-IDF 5.3.1 firmware (414 KB): SD over SPI, `needle` flash partition holding a verbatim copy of the embedding (tied output head) + tokenizer, UI (Space Grotesk / JetBrains Mono, OFL), XPT2046 touch, RGB LED, serial console (`say`, `set`, `bench`, `stats`, `sdbench`, `sdraw`, `colours`).
- **On the real CYD** (ESP32-D0WD-V3, 4 MB flash, no PSRAM): genuine inference works. Base 2-bit L2: "I need a rest" → `rest()`, 55 s. Tuned L2 from the touchscreen: "I'm not hungry, tired and bored." → reasoning "'tired' -> rest" → `rest()`, 68–72 s. Peak ~198 KB DRAM (+37 KB IRAM for the base model's probe pool), 13 KB heap free with the tuned model.

**Decided / learned (why)**
- Embedding + tokenizer in internal flash: the output head is read every decode step; everything else streams from SD.
- Confidence: engine's exact number isn't reproduced (it re-scores an id sequence built inside `generate_turn`); the port uses the documented definition min(head, renormalised call prob). My probe-pool maths matches the engine's `needle_embed` to cosine 0.99999.
- SD speed was the whole story: stdio `fread` on ESP-IDF fed the VFS tiny pieces (0.11 MB/s); POSIX `open/read` + sector-aligned reads + a 2-way 4 KB block cache + zero-copy tile views reach the card's 2.2–2.4 MB/s. 160 s → 55 s per decision.
- ESP32 IRAM heap only takes 32-bit **integer** loads/stores (FPU loads fault): the probe pool goes through `ld32/st32`.
- This board's panel: ILI9341_2 init sequence, MADCTL 0x00 (RGB, no swap), no inversion. Touch calibration X 285–3562, Y 441–3799 (both chosen on the board from test frames and corner taps).

**Next**
- Run the on-device 24-case benchmark (`bench` over serial, ~28 min).
- Speed: under 60 s (fewer SD bytes per token; kernel LUTs; dual-core prefetch if RAM allows).
- Write the deliverable docs: `docs/feasibility.md`, `memory-budget.md`, `architecture.md`, golden files in `tests/golden/`, README, the RESULT report. Stretch: the autonomous creature screen.

## 2026-09-24 — on-device benchmark and docs

**Done**
- On-device benchmark of the tuned 2-layer rung on the CYD: **24/24 correct**, and all 24 identical to the host build (call and reasoning text). Mean 71.0 s per decision (66.9–80.3 s), heap minimum 13.7 KB free. Results in `bench/RESULTS.md`; raw serial log in `bench/device_bench_tuned_2026-09-24.log`.
- Docs: `docs/feasibility.md` (exact byte breakdown for L2/L4/L8/L20 and the memory table), `docs/memory-budget.md`, `docs/architecture.md`.
- Golden files `tests/golden/*.json` from the NumPy oracle (base 2-bit L2); `tests/test_golden.py`: the C runtime passes all four (identical tokens, top-10, confidence and 12-token reasoning).

**Next**
- README with the RESULT block; build/flash/serial instructions.
- Speed: under 60 s (SD bytes per token is the bottleneck).
- Stretch: the autonomous creature screen.

## 2026-09-24 (evening) — speed work: 71 s → ~35 s per decision

**Done**
- Profiled on the board (`[prof]` line: engram / attention / MLP / output head). SD bytes were ~70% of the time; the 2-bit kernel's per-weight `switch` was most of the rest.
- **2-bit fine-tune** (`tools/finetune_rung_2bit.py`): LoRA trained on top of the shipped 2-bit rung through 2-bit numerics, exported with every untouched tensor byte-identical to `needle3-L2.cact` and only the 10 attention matrices re-packed at CQ2. Found that CQ re-encoding is not idempotent (codebook points aren't unit length, so the stored norm shrinks ~6%); the encoder now stores the least-squares group norm (same format and reconstruction, exact on already-quantised groups). Rank 64, 10 epochs: 191/200 held-out, **24/24** on the device template (engine and port). File 8.6 MB instead of 13.3 MB; SD bytes per token 2.75 MB → 1.39 MB.
- Kernels: 2-bit via a 256×4 byte→codebook table and 4 independent multiply-adds; 4-bit via direct codebook loads. Compute 14.7 s → ~3.7 s. Oracle checks unchanged.
- Flash: the `needle` partition is now memory-mapped (zero-copy output head), grown to 3.44 MB (app slot 704 → 512 KB) and holds 59 tensors: embedding, tokenizer, every FP16/FP32 vector and the mHC maps (3.57 MB). Engram row gathers are single-sector reads.
- Serial file push (`tools/push_file.py`, `recv` command, CRC32 per 512 B chunk, 460800 baud: 8.6 MB in 5 min). No more card swaps.
- Bug fixed: the prefix snapshot was keyed on the directory only, so two fine-tunes of the same rung shared it. Now keyed on a 256-point sample of the weight bytes too.
- Result: "I'm not hungry, tired and bored." → `rest()` in **34.6 s** (was 71 s; 160 s at the first working run). Free heap at the ready screen is down to ~8–9 KB.

- Board benchmark with the 2-bit model: **24/24**, all identical to the host build, mean **37.2 s** (34.6–41.6), min free heap 10.4 KB (`bench/RESULTS.md`).

**Next**
- README + RESULT block; memory docs updated for the new partition map.
- Stretch: the creature screen.

## 2026-09-24 (night) — README, RESULT, the creature

**Done**
- `README.md` with the RESULT block, what is genuinely on the device, build/flash/serial instructions.
- The autonomous creature (stretch goal): a second 2-bit fine-tune of the same rung (`models/needle3-L2-creature4-2bit.cact`, four stats, tools eat/sleep/explore/play; 220/240 held-out, **24/24** on its benchmark on engine and port; 15/15 on the board before the run was stopped for UI work). Profiles on the card (`/needle`, `/creature`, `/sdcard/profile.txt`), switched from the debug screen. World sim in `firmware/app/creature.cpp`.
- Creature screen: a scene renderer that composes the panel in 8-row strips in the LCD driver's own DMA buffer (no framebuffer, no flicker), ~8 fps: breathing, blinking, a ball on a real bouncing arc with squash and shadow, chewing and a bitten apple, drifting z's, wandering with a magnifier, glancing while thinking. Footer changed from a slogan to facts ("on-device · Wi-Fi off · #8 in 47 s"). Previews rendered on the Mac with the firmware's own UI code (`host/ui_preview.cpp`, `docs/screens/`).
- Heap at the creature's ready screen: 8.6 KB free.

**Next**
- Check the animation on the real board (done: it looks right).
- Full creature benchmark on the board (15/15 so far).

## 2026-09-24 — explainer video
- Built `video/`: Chatterbox narration (calm default voice, sentences re-rolled until Whisper hears them as written), YuE2 instrumental via ChorusServer (take C chosen; B may carry faint vocals), Blender board shots (orbit, push-in) with the firmware's own creature frames on the screen, Remotion scenes timed from `audio/timing.json`, burnt-in captions, music ducked under voice, loudness −16 LUFS.
- Output: `video/out/needle-on-cyd.mp4` (1080p30, 2:13). Whisper transcript of the final mix matches the script.
- Next: user review of the video; push the `public` branch (plus the later master commits) to greencat667/needle-on-cyd; put the Medium draft in with the repo link.
- Later the same day: captions removed from the video (they'll be added on YouTube); `video/out/needle-on-cyd.en.srt` generated from the narration timing for upload. New "So what?" scene 9 (narration added with `video/tts/add_scene.py`, other sentences untouched): the creature as a stand-in for devices that turn a situation into an action, with three example devices shown as ideas, not built. Video now 2:41. Matching "So what?" section added to the blog draft.
- Published: https://github.com/greencat667/needle-on-cyd (public, Apache-2.0; `public` branch pushed as `main`, noreply identity, three commits). Board photos had GPS in their EXIF and were stripped before publishing.
- README second pass: leads with the creature, a "Using it" section with the real touch targets (header → debug; switch button; the tap-to-set demo), stale claims fixed. `prepare_sd.py --profile creature|needle` now writes either profile and `profile.txt`; checked that one overlay serves both fine-tunes (byte-identical).
- YouTube: draft GVSNnOFtazo filled in (title, description with chapters and repo link, tags, Science & Technology, not for kids, no paid promotion, AI use: Yes, English (UK)) and the SRT uploaded with timing. Left private at the Visibility step for Christian to publish. Metadata kept in `video/youtube.md`.
- Next: Christian publishes the video; then add the video link to the README and the blog; Medium draft still waiting on his go-ahead.

## 2026-09-25 — Medium draft
- Blog gained a process diagram (`docs/blog/img/diagram-loop.png`, drawn by `tools/make_blog_diagram.py` in the charts' style, labels taken from the firmware) showing the loop: world drifts → it describes itself → Needle decides on the ESP32 → the action runs. Drawn in code rather than ChatGPT so the labels are exact.
- Medium draft created (unpublished): https://medium.com/p/a3c8ec0a824b/edit. Pasted as HTML; Medium re-hosted all five images from the public repo.
- Next: Christian reviews and publishes the Medium post and the YouTube video; then cross-link them (video link in README and blog, post link in the video description).
