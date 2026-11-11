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
investigated which step dropped it). `DOWNLOAD_MODELS.bat` now looks for
`whisper-server.exe` specifically, so a copy made from the batch app fetches it.

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

**`ffmpeg.exe`**: dshow demuxer has `-audio_buffer_size` (ms). Capture from a real USB mic
verified in §3.

**GPU ordering differs between tools.** On the dev machine, CUDA (`llama-server
--list-devices`, `whisper-server -dev`) orders `CUDA0 = RTX 3080`, `CUDA1 = RTX 5060 Ti`,
while `nvidia-smi` lists the 5060 Ti first. Never map one index onto the other. The app
therefore uses neither index: it pins each GPU child to the chosen card with
`CUDA_VISIBLE_DEVICES=<uuid>` (CLAUDE.md §8).

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

### Measured later

- Live microphone capture via dshow: §3.
- A real 300 s block cadence over 3h43m: §6.
- Still not compared: IQ3_XXS against IQ4_XS note quality on the same blocks.

---

## 3. Live microphone (2026-11-11)

**Device:** a USB mic, `Microphone (BOYA CM30)` — the only dshow audio device. Its USB
instance path is `USB\VID_4D43&PID_3033&MI_00\9&3A2E8903&0&0000`. The `9&…&0&0000` form is
Windows-generated and port-derived: **the device has no serial number**. On another port it
gets a new instance and a new endpoint GUID (the dshow alternative name
`@device_cm_{…}\wave_{96CBD551-…}`), and often a renamed friendly name such as
`Microphone (2- BOYA CM30)`. CLAUDE.md §4 matches on the friendly name with that prefix
stripped, and never stores the GUID.

The mic also exposes a playback endpoint, `Speakers (BOYA CM30)` (its headphone jack).
Windows did **not** make it the default output; the default stayed
`Speakers (High Definition Audio Device)`.

**dshow capture works as specified:** `-f dshow -audio_buffer_size 50 -i audio="<name>"
-ac 1 -ar 16000 -f s16le pipe:1`, with no stderr output.

**Level check** (meeting audio from the room speaker, mic beside it, Windows mic level
100%):

| | RMS | dBFS |
|---|---|---|
| room, silent | 62 | −54.5 |
| room, playback | 245 | −42.5 (peak 3,432) |
| signal over silence | | **12 dB** |

The first attempt measured 1.3 dB: the default speaker was muted in Windows. It was
unmuted by request.

### Live vs replay — timing

10 minutes, same stretch as §2, real 300 s block cadence, IQ3_XXS + MTP, one GPU.

| | replay | live mic |
|---|---|---|
| utterances | 89 | 83 |
| latency median / worst | 1.1 / 3.1 s | **1.1 / 2.2 s** |
| peak VRAM (5060 Ti) | 14,755 MiB | 14,746 MiB |
| CJK characters in output | 0 | 0 |

**Capture adds no latency.** Whisper's warm-up took **61.8 s** again on the first launch
in 11 days: the driver's JIT cache does not persist reliably, so the warm-up (§8) can be
needed on any launch, not only a machine's first.

### Transcript quality — short utterances are the problem

No ground truth exists for this audio. The reference is **batch-style `whisper-cli`**
over the same 10 minutes (one call, whisper's own 30 s windows, `--vad`) — the method the
operator is satisfied with in the batch app. Word error rate against that reference:

| method | WER vs batch reference | utterances | median length |
|---|---|---|---|
| live mic, 0.6 s silence / 15 s cut | 68.1% | 83 | 3.2 s |
| replay, 0.6 s / 15 s | 55.1% | 89 | 3.8 s |
| replay, 0.6 s / 15 s + context carry | 54.3% | 89 | 3.8 s |
| replay, 1.0 s / 15 s + carry | 53.0% | 65 | 4.8 s |
| replay, 1.0 s / 30 s + carry | 50.0% | 57 | 5.4 s |
| **batch `whisper-cli` per 5-minute block** | **21.1%** | — | 300 s |

- The audio is hard: mixed-language, overlapping speech. Translations of the Mandarin
  vary between any two passes, so part of every WER here is wording, not loss.
- **Utterance-sized pieces are the main loss** (~55%), and the room acoustics add ~13
  points on top (68%). Tuning the segmenter recovers 5 points at most.
- **Transcribing a whole 5-minute block at once comes within 21%** of the reference.
  It took 10.0 s per block with `whisper-cli`, including the model load that a resident
  `whisper-server` avoids.
- Hallucinated fillers ("Thank you.") appear in the live output, as §6 expected.

This points at a second pass: re-transcribe each finished block batch-style, from audio
held in memory only. Approved by the operator; see §4 and CLAUDE.md §6.1.

---

## 4. The refine pass — how to re-transcribe a block (2026-11-11)

Same 10 minutes and the same batch-style reference as §3. All runs use `whisper-cli`,
large-v3-turbo, `-l en` and `--vad` with `ggml-silero-v5.1.2.bin`.

| method | WER vs reference | wall time |
|---|---|---|
| whole 300 s block, GPU | **21.1%** | 10.0 s incl. model load |
| 60 s chunks, fixed cuts, GPU | 42.2% | — |
| 60 s chunks + previous 200 chars as `--prompt`, GPU | 38.4% | — |
| whole 300 s block, **CPU** (`-ng`), 8 threads | 31.2% (both blocks) | **122 s** |
| whole 300 s block, CPU, 12 threads | (as above) | **106 s** |
| CPU blocks vs GPU blocks, against each other | 22.9% | |

- **Chunking loses most of the gain.** Prompt carry recovers 4 points of the 21 lost. The
  long window is the point; it cannot be approximated with short ones.
- **~20–25% is the noise floor on this audio.** Two equally valid batch passes, CPU and
  GPU, differ by 22.9%: the numerics differ slightly, and the Mandarin translation is
  re-worded. The reference was made on the GPU, which is why GPU blocks score closer to it
  — not evidence that CPU blocks are worse. Read every WER in §3–4 against that floor.
- **CPU refine keeps up:** 106–122 s per 300 s block is 35–41% of real time, on a machine
  with 24 logical cores. A 16-thread machine should still manage at 8 threads.
- Chosen: **a second whisper-server, CPU-only, whole blocks** (CLAUDE.md §6.1). The GPU
  instance would serve one request at a time and stall live lines 5–10 s per block. A
  second GPU instance needs ~1.9 GiB that §2 shows is not there.

### Refine output needs a punctuated prompt (2026-11-11)

Whole-block transcription with whisper's VAD came back all lower case with no
punctuation, in 2-3 s fragments: on a 5-minute block, 0 sentence marks and 6%
capitalised words. A punctuated opening line passed as the prompt
(`asr.REFINE_PRIMER`) restores it with the same words: 104 sentence marks, 94
commas, 25% capitalised. The refine pass also merges whisper's segments into
sentence-length lines (`session._sentence_lines`): a line ends at a sentence end
once it is 60+ characters, at a pause of 2 s or more, or at about 300 characters.

---

## 5. The built app, end to end (2026-11-11)

All runs replay a recording of an in-room meeting through `capture.replay_file`, on the
dev machine (RTX 5060 Ti 16 GB chosen over the RTX 3080 by VRAM, as CLAUDE.md §8 requires).

**12-minute replay, 2-minute blocks** (to exercise the refine pass and notes six times):

| | |
|---|---|
| models loaded | whisper GPU 2.8 s, whisper CPU 2.3 s, llama-server 8 s; warm-up 0.6 s |
| utterances | 111 |
| latency, last speech frame -> text on the page | median 1.09 s, p90 1.31 s, max 4.12 s |
| refine pass (CPU, 12 threads) | 40-63 s per ~2-minute block |
| notes per block | 5.4-9.8 s, 26-33 tok/s |
| peak VRAM, 5060 Ti | 14,774 MiB (whisper GPU + IQ3_XXS + MTP) |
| VRAM on the 3080 | 0 -- the CPU refine server sees no GPU (`CUDA_VISIBLE_DEVICES=""`) |
| stop when the replay ran out | automatic; partial block refined and noted |
| meeting text outside `output\<meeting>\` | none -- grep of the app folder, `temp\` included |

**Full summary of that meeting, forced multi-tier** (`summary.target_tokens` 800,
`group_reduce_threshold` 2):

| | |
|---|---|
| model | IQ4_XS, ctx 32768, fully on the GPU, MTP; ready in 17 s after whisper and IQ3 were unloaded |
| path | 266 lines -> 11 chunks -> group tier 1 (3 groups) -> tier 2 (1) -> final |
| map calls | 13-31 s each, 27-33 tok/s |
| final call, thinking on | 143 s; 1,660-token document, 5,681 tokens including thinking |
| after | llama-server stopped, VRAM back to 438 MiB, no processes left |
| output | topic sections in paragraphs, no bullets or labels; clock times cited |

Two group calls hit `max_group_output_tokens` (2,500). That is an artefact of the forced
800-token chunks producing many notes per group; at the default 10,000-token chunks a
group holds far less. Worth watching on a real 3-hour meeting.

**Shutdown.** Stopping the server process left no `whisper-server.exe`,
`llama-server.exe` or `ffmpeg.exe` running.

**`run.bat`** was launched for real: preflight passed, the server came up on 8000 and the
browser opened. Its desktop shortcut had an empty version stamp (`?v=`), a bug carried
over from the batch app, where a `for /f` command beginning with a quoted path loses its
quotes. Fixed by prefixing `call`.

Which acceptance tests have run, and their results, is tracked in `TESTING.md`.

---

## 6. Long session, Port mode, and the summary settings (2026-11-11)

### 3h43m live session (CLAUDE.md §16 test 7)

A single 3h43m recording of a public meeting held online, in English, replayed at real
time through the whole app with the default settings (5-minute blocks, IQ3_XXS + MTP).
1,374 transcript lines, 45 blocks.

| | hour 1 | hour 2 | hour 3 | hour 4 (43 min) |
|---|---|---|---|---|
| utterances | 464 | 434 | 433 | 369 |
| latency median | 1.04 s | 1.03 s | 1.05 s | 1.04 s |
| latency p90 | 1.25 s | 1.22 s | 1.23 s | 1.22 s |
| latency max | 2.07 s | 2.51 s | 2.00 s | 2.18 s |

- **All 45 blocks** got notes; no refine fallbacks, no queue build-up.
- Refine pass: up to 149 s per 5-minute block (of the 300 s available). Notes 4-9 s per
  block, the same in the last hour as the first.
- **App server memory flat** at ~100 MB. Both whisper servers flat. **VRAM flat**,
  14.3-14.5 GiB.
- **llama-server's system RAM grew 11.6 -> 19.0 GB.** This llama.cpp build caches past
  prompts in RAM, 8 GiB by default (`--cache-ram`). Every prompt here is new -- each block,
  chunk and reduce -- so it was pure waste. **Fixed: `--cache-ram 0`** on our own server.
  Measured afterwards: 14.5 GB peak across a whole 3h43m full summary.

### Port mode (test 14)

Against the operator's own launcher -- IQ4_XS across both cards, 163,840 context.

- Attached, and read the model and its context from `/props`.
- **Whisper fitted beside it with 292 MiB to spare** on the 16 GB card. Works, but too
  thin for comfort: anything else taking VRAM mid-meeting could stop transcription. A
  smaller-context launch would leave more.
- Notes written by that server with thinking off (130-150 tokens, 7-9 s).
- Full summary through it: map 40 s, final 82 s with thinking on, `max_tokens` 24,000
  (spare context on an external server, as designed).
- **Never killed:** healthy after Stop and after the summary.

### The map notes were being truncated

The batch app's `max_map_output_tokens: 1500` was carried over. On this meeting **every
map call ended at exactly 1,500 tokens**. The batch app's own logs show the same on every
real meeting ("map calls run 1,500 tokens"), where it was read as their natural length;
nothing there measured it as a cap.

Same meeting, two full summaries, checked against its official minutes (37 agenda lines,
25 recorded motions). Coverage counts the minutes' own words for each agenda item in the
summary:

| | cap 1,500 | cap 2,500 |
|---|---|---|
| map call lengths | 1,500 x 6 (all at the cap) | 1,877-2,500 (3 of 6 at the cap) |
| final prompt | 9,820 tokens | 14,978 tokens (fits: + 12,000 + 2,000 < 32,768) |
| topic sections / words | 14 / 4,607 | 16 / 4,895 |
| decision wording (approved, carried, adopted...) | 24 | 36 |
| agenda items in the back half of the meeting | thin | roughly doubled |
| total time | 13 min | 17 min |

What the cap cuts is the **last headings** of each chunk's notes, not the end of the chunk in
time: the notes are written heading by heading (Themes, then Outcomes, then Loose Ends),
and generation simply stops at the cap. So Outcomes -- the decisions -- were the first
casualty, which is what the decision count shows. In the batch app's own cache for a
1h09m meeting, two of its three notes stopped inside Positions and Themes respectively, so
Outcomes and Speaker Evidence never reached the final reduce for two-thirds of the
meeting. Every topic of substance was present at 1,500 too, so the cap costs detail and
decisions rather than whole topics.

**Then 4,500, where the cap stops binding.** Re-running the batch app's summarising stage
on two real meetings' transcripts (1h09m, 2h12m): at 2,500 the longest notes still stopped
early; at 4,500 every note finished on its own, at 585-3,937 tokens -- the notes' natural
length. No repeated sentences and no rise in hedging; the 2h12m summary recorded the most
decisions of the three settings (32, against 26 at 2,500 and 18 at 1,500) and dropped a
bulleted "open items" list that 2,500 had added. About 25 s more per meeting. One run each,
so the counts carry sampling noise; completeness does not. Full table in the batch app's
BUILD_NOTES 9ay. **Default is now 4,500** in both apps.

Fuller notes mean the final prompt outgrows the 32k context from about six chunks
(~4 hours), where `_fit_final` adds a consolidation tier -- a few minutes, by design.

### Incidental passes

- **Test 23:** a 24-second session was removed at Stop ("under 40 words, not kept").
- **Test 19, partly:** delete through the app's API removed a meeting's folder.

---

## 7. Microphone dropouts (2026-11-11)

**Unplugging the USB mic does not end ffmpeg's dshow capture.** The process stays alive and
silent: no exit, nothing on stderr, no more audio on the pipe. Found by the operator
unplugging the mic mid-meeting: the session ran on for two minutes having received 3 s of
audio, still showing *Listening*.

So capture watches the pipe instead (CLAUDE.md §4): a connected mic delivers frames
continuously, silence included, so **3 s with no frames** means the device is gone (6 s
grace before the first audio). The meeting stays open, the same mic is looked for every
2 s by name, and the gap is filled with silence from the last frame received.

Tested by freezing the session's ffmpeg process (`NtSuspendProcess`), which reproduces
exactly what an unplug does to the pipe:

| | |
|---|---|
| stall detected, message shown | 3.6-4.1 s after the freeze |
| resumed after audio was available again | 2 s (the next device-list check) |
| frozen ffmpeg left behind | none -- killed by the watchdog |
| audio clock vs wall clock over the test | 18.0 s against 19.7 s |

The first version counted the gap from when the stall was *noticed*, losing the ~3 s
before it (15 s against 19.7 s). Counting from the last frame received fixed it. The
remaining ~1.7 s is the mic's own start-up, once at Start and once at the reconnect.

A real unplug-and-replug with the hardware is still to be confirmed (`TESTING.md`).
