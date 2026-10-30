# BUILD_NOTES.md

Flag verification and measurements for this repo. The batch app's notes, which this file
cites as *batch §N*, are at `https://github.com/NGJHD/Meeting-Summarizer/blob/main/BUILD_NOTES.md`.

No meeting content appears here: recordings are described by length and kind only.

---

## 1. Binaries and flags (CLAUDE.md §1)

Verified 2026-10-30 against the binaries in `bin/`.

**`whisper-server.exe`** was missing from `bin\whisper-cuda\`. It is in the pinned upstream
zip (`whisper-cublas-12.4.0-bin-x64.zip`, build `b4938`) alongside the `whisper-cli.exe`
already shipped, so it shares the same DLLs. The copy in `bin\` arrived without it (not
investigated which step dropped it). `DOWNLOAD_MODELS.bat` must make sure it is kept.

| CLAUDE.md says | Shipped build | Notes |
|---|---|---|
| `-m`, `-l en`, `-t`, `--host`, `--port` | present | |
| `-dev N` | present | CUDA device index (see GPU ordering below) |
| `--convert` | present | not used — writes temp files |
| request field `prompt` | accepted | also `--prompt` / `--carry-initial-prompt` at launch |
| `response_format=verbose_json` | works | segments carry `no_speech_prob`, `avg_logprob` |
| — | `-nth 0.60`, `-lpt -1.00` | built-in no-speech / logprob thresholds, on by default |
| — | `--vad` family | server-side VAD exists; we segment ourselves (§5) and do not use it |

**`llama-server.exe`** (`b10852`): `--device`, `--spec-type draft-mtp`, `--reasoning-budget`,
`-ub`, `--flash-attn on`, `--no-webui`, `--override-tensor` all present.

**`ffmpeg.exe`**: dshow demuxer has `-audio_buffer_size` (ms). Device capture is not yet
tested — no microphone available.

**GPU ordering differs between tools.** On the dev machine, CUDA (`llama-server
--list-devices`, `whisper-server -dev`) orders `CUDA0 = RTX 3080`, `CUDA1 = RTX 5060 Ti`,
while `nvidia-smi` lists the 5060 Ti first. Never map one index onto the other; match by
name.

**`bin\` must be on PATH** for both servers: `cudart64_12.dll` and the cuBLAS DLLs sit in
`bin\`, not in the engine folders. Without it llama-server fails with
`invalid device: CUDA1` — the CUDA backend never loaded, so no CUDA devices exist. That
error is misleading. `run.bat` already sets PATH; any harness must too.

**Silero VAD v5 ONNX** (`snakers4/silero-vad` tag `v5.1.2`,
`src/silero_vad/data/silero_vad.onnx`, 2,327,524 bytes): inputs `input [N, 576]`
(512 new samples + 64 context), `state [2, N, 128]`, `sr` int64 scalar. 0.46 ms CPU per
32 ms frame on one thread — 1.4% of one core.

---

## 2. Spike: whisper + LLM resident together (CLAUDE.md §15 step 2)

`tools/spike.py`. Replays a recording at real time (`ffmpeg -re`) through Silero VAD →
whisper-server → transcript, firing a block summary at llama-server **every 120 s of
audio** (2.5× the real 300 s, to collect more contention samples). Thinking off,
`prompts/block.txt`, ctx 8192, q8_0 KV. VRAM sampled from nvidia-smi once a second.

Input: 12 minutes (from minute 10) of a 1h09m in-room recording with mixed English and
Mandarin. 110 utterances, 3 forced cuts at 15 s, 6 blocks per run.

**Dev machine:** RTX 5060 Ti 16 GB (compute 12.0) + RTX 3080 10 GB. Idle desktop holds
~500 MiB on the 5060 Ti.

### Results

| | A: one GPU, IQ3_XXS + MTP | B: two GPUs, IQ4_XS + MTP | C: one GPU, IQ4_XS, no MTP |
|---|---|---|---|
| whisper on | 5060 Ti | **3080** | 5060 Ti |
| LLM on | 5060 Ti | 5060 Ti | 5060 Ti |
| 5060 Ti peak VRAM | 14,197 / 16,311 MiB | 14,930 | **15,968** |
| headroom on 5060 Ti | 2.1 GiB | 1.3 GiB | **0.33 GiB** |
| latency, LLM idle (median / max) | 1.06 / 1.44 s | 1.04 / 1.24 s | 1.06 / 1.46 s |
| latency, LLM busy (median / max) | 1.50 / 1.81 s | 0.96 / 1.29 s | 1.65 / 2.25 s |
| whisper inference, idle (median) | 0.45 s | 0.44 s | 0.46 s |
| whisper inference, LLM busy (median) | 1.08 s | 0.46 s | 1.17 s |
| block call wall time | 5.3–8.3 s | 4.6–6.9 s | 7.9–15.4 s |
| generation | 38–47 tok/s | 43–46 tok/s | 21–26 tok/s |
| prompt processing | 560–610 tok/s | 600–680 tok/s | 650–720 tok/s |
| MTP draft acceptance | 48–68% | 57–65% | — |
| whisper queue depth, max | 0 | 0 | 0 |

Latency is measured from the last speech frame heard to text returned, so it includes
the 600 ms of silence that closes an utterance. **Text is on screen about 0.45 s after the
utterance is closed.**

### Conclusions

1. **The design holds on one 16 GB card.** Sharing the GPU roughly doubles whisper
   inference time while a block is generating (0.45 → 1.1 s), but each block takes only
   5–8 s every 5 minutes, and worst-case latency stayed under 2 s (A) against a 5 s
   target. CLAUDE.md §8's GPU-contention fallbacks were not needed.
2. **The default should be A** — IQ3_XXS with MTP, ctx 8192, on a single 16 GB card. MTP
   fits beside whisper with 2.1 GiB to spare.
3. **IQ4_XS beside whisper on one 16 GB card (C) technically fits, and should not be
   offered.** 0.33 GiB of headroom is below the §16 test 8 floor of 0.5 GiB. A browser or
   other GPU app opening during a meeting would push llama.cpp into paging. MTP does not
   fit, so it is also half the speed of A.
4. **With a second GPU, B is strictly better.** It gives High Quality at full MTP speed
   and whisper never contends with the LLM. CLAUDE.md §0.4 assumes one card. Whether to
   support this layout is an open question for the operator. If yes: whisper on the card
   with less VRAM, the LLM on the largest one, chosen by name, not index.
   **Decided 2026-10-30: not supported.** The app targets one 16 GB card so it behaves the
   same wherever it is deployed. On a multi-GPU machine both servers go to the largest card
   (CLAUDE.md §8).
5. **`WHISPER_RESERVE_GIB` ≈ 1.9** — large-v3-turbo resident with CUDA context and
   buffers, measured on the 3080 in B (2,255 − 379 MiB baseline).
6. **Blocks are cheap.** ~1,000–1,250 prompt tokens and 150–350 output tokens per
   2-minute block. A real 5-minute block is ~2.5× the prompt; prefill at ~600 tok/s adds
   about 3 s. Expect ~8–12 s per real block in A.

### First-use JIT — a real startup cost

The first spike run's **first whisper call took 59 s**; every later call took ~0.45 s.
The pinned CUDA 12.4 build has no native kernels for compute 12.0 (Blackwell), so the
driver JIT-compiles PTX on first use. In later runs the first call took 0.59 s: the driver
caches the result in its own ComputeCache, across processes.

Consequences for the app:

- **Warm up whisper before declaring the session live:** one inference on a short noise
  clip, included in *"Loading models…"*. Without this the first utterance of the first
  meeting is a minute late and everything behind it queues (queue depth reached 7).
- **The first run on a new machine or after a driver update may take a minute longer to
  load.** Say so in the loading message rather than letting it look hung.
- llama-server did not show the same delay. It loaded in 5–8 s from a warm disk cache.

### Not yet measured

- Live microphone capture via dshow, including `-audio_buffer_size`.
- A real 300 s block cadence over 3 hours (§16 test 7) — memory growth and drift.
- IQ3_XXS vs IQ4_XS note quality on these blocks. Both runs' notes exist locally and
  were not compared.
