# CLAUDE.md — Offline Real-Time Meeting Transcriber with Rolling Notes

You are building a **fully offline Windows desktop application** that listens to a
microphone in a meeting room, shows a live English transcript as people speak, and keeps a
column of short notes summarising each completed 5-minute block of the meeting. Every
meeting is kept, as text, in a history sidebar (§11.1). One button stops listening and
writes a full summary of the meeting (§7.1).

This repository began as a copy of the batch app at
`https://github.com/NGJHD/Meeting-Summarizer` (a recording goes in, minutes come out). About
two thirds of that code was removed; hardware detection, the llama-server lifecycle,
config, cleanup, the updater and the packaging carried over and were cut down to fit.

**Status (2026-11-11): built, and verified on replayed recordings and one live microphone
check.** §15 marks each build step's state. §16 lists the acceptance tests; the ones that
need a person, a real meeting or a clean machine are still open, and `BUILD_NOTES.md` §5
records what has and has not been run.

Read this entire file before writing any code. Every design decision here was made
deliberately. If you think one is wrong, say so explicitly in your response — do not
silently substitute your own approach.

Where this file says *measured in the batch app*, the evidence is in that repo's
`BUILD_NOTES.md`. It is not copied here. Start a fresh `BUILD_NOTES.md` in this repo for
everything measured in this one.

---

## 0. NON-NEGOTIABLE CONSTRAINTS

1. **No internet access at runtime.** Not for models, fonts, CDN scripts or telemetry. The
   app must work with the network adapter physically disabled. The only exception is the
   *Check for updates* button in the About overlay. It is pressed by hand and never runs on
   startup, on a timer or in the background.
2. **No installation steps.** The user unzips a folder and double-clicks `run.bat`.
3. **No PyTorch**, and nothing that pulls it in: no transformers, no HuggingFace hub
   client, no WhisperX, no faster-whisper. All heavy inference happens in pre-built native
   binaries, called by subprocess or HTTP. onnxruntime is acceptable — it is a plain wheel
   with no torch in its tree.
4. **NVIDIA only.** Windows 11, an NVIDIA GPU with 16 GB of VRAM, 32 GB of system RAM. The
   batch app's Vulkan, CPU and unified-memory paths are **out of scope** and are removed as
   the code is touched (§13). Doing both models in real time on anything weaker is not a
   goal.
5. **One user, one room, one microphone.** No accounts, no network viewers, no multiple
   sessions at once.
6. **Text only, never audio.** Every meeting's transcript and 5-minute notes are saved
   automatically, **as they happen**, in the app's own `output\` folder (§11.1). So is its
   full summary, if one is generated, and a copy of the summary goes wherever the user
   picks in a Save dialog (§7.1). **Audio is never written to disk**, not even
   temporarily. See §10.

   (History of this rule: the first version of this spec wrote nothing at all. The
   operator changed it on 2026-11-11 to keep a history.)

### Scope

- **In-room meetings only.** The microphone is the only audio source. Capturing system
  audio from an online call (WASAPI loopback, virtual cables) is out of scope.
- **The transcript is English.** Speakers mix English and Mandarin. Whisper with its
  language forced to English writes the Mandarin as English as well. The operator has used
  this in the batch app and is satisfied with it. It is a translation, not a verbatim
  transcript — names and figures said in Mandarin may come out paraphrased — and that is
  accepted. **Do not "fix" it** by switching to language auto-detection or a Chinese-capable
  recogniser: that would produce Chinese characters, which is not what is wanted.
- **No speaker labels.** No diarization, live or afterwards.
- **One generated document, on request only:** a full summary covering *what was
  discussed and what was concluded*, by topic, with no speaker attribution (§7.1). It can
  be generated at the end of a meeting or later, from a past meeting in the history. No
  minutes, no action-item list, no other export.
- **Recording length: up to about 3 hours of continuous listening.** Nothing may assume a
  length: not buffer sizes, not list growth, not the number of blocks. Memory must stay
  flat over a long session apart from the transcript text itself.

---

## 1. VERIFY BEFORE YOU CODE

llama.cpp and whisper.cpp rename and remove flags often. Before writing the orchestration
code, run each of these against the exact binaries in `bin/` and read the output:

```
bin\llama-cuda\llama-server.exe --help
bin\whisper-cuda\whisper-server.exe --help
bin\ffmpeg.exe -hide_banner -list_devices true -f dshow -i dummy
```

**`whisper-server.exe` comes from the pinned upstream zip** (`whisper-cublas-12.4.0-bin-x64.zip`,
build `b4938`); the batch app had kept only `whisper-cli.exe`. `DOWNLOAD_MODELS.bat` now checks
for `whisper-server.exe` specifically. Do not substitute `whisper-cli.exe` per utterance:
reloading a 1.6 GB model for every sentence defeats the point.

For every flag and request parameter this document names, confirm it exists under that name
in that build. If it is absent or renamed, use the equivalent and record the change in
`BUILD_NOTES.md`. Do not silently drop a flag, and do not guess.

If a binary or model file is missing, stop and report which one.

---

## 2. ARCHITECTURE

```
Microphone
 └─> ffmpeg (dshow)            → 16 kHz mono s16le PCM on a pipe. Never on disk.
      ├─> level meter          → RMS, ~5 Hz, to the UI
      └─> VAD segmenter (CPU)  → utterances (§5)
           └─> whisper-server (GPU, resident)   → English text per utterance (§6)
                ├─> transcript column           (SSE, a second or two behind speech)
                └─> block buffer: every 5 min of audio (in memory only)
                     └─> whisper-server (CPU, resident) → refined block text (§6.1)
                          └─> llama-server (GPU, resident, thinking off) → block notes (§7)
                          └─> notes column      (SSE)
```

Four long-lived children: `ffmpeg.exe`, two `whisper-server.exe` (GPU for live lines,
CPU for the refine pass, §6.1) and `llama-server.exe`. The
Python layer is an **orchestrator only**. It reads a pipe, runs a tiny VAD model, and makes
HTTP calls. It performs no recognition or generation itself.

**Language: Python** — the standalone relocatable CPython in `runtime\` with vendored
wheels. The frontend is plain HTML/CSS/JS with no build step, no npm and no framework.

**Backend: FastAPI + uvicorn**, bound to `127.0.0.1` only. Never `0.0.0.0`.

### 2.1 The one inversion from the batch app

The batch app's rule was *"Whisper and the LLM must never be resident in VRAM at the same
time."* **This app inverts that rule.** Both servers are resident for the whole session,
because transcription cannot stop while a block is summarised. Two consequences follow, and
everything in §8 exists because of them:

- **VRAM is shared.** The LLM is sized to leave room for whisper.
- **GPU time is shared.** While a block summary is generating, whisper's latency rises.
  The transcript must never *wait* on the LLM — separate threads, separate HTTP
  connections, no shared lock — but it may run slower for those few seconds.

---

## 3. DIRECTORY LAYOUT

```
MeetingTranscriber\
  run.bat
  DOWNLOAD_MODELS.bat
  config.json
  BUILD_NOTES.md              <- start fresh; flag verification + measurements
  runtime\                    <- standalone CPython + fastapi, uvicorn, numpy, onnxruntime
  bin\
    ffmpeg.exe
    cudart64_12.dll cublas64_12.dll cublasLt64_12.dll
    llama-cuda\llama-server.exe ...
    whisper-cuda\whisper-server.exe ...
  models\
    ggml-large-v3-turbo.bin
    silero_vad.onnx                     <- for our own segmenter (§5)
    ggml-silero-v5.1.2.bin              <- whisper's own VAD, refine pass only (§6.1)
    Qwen3.8-27B-UD-IQ3_XXS.gguf         <- default (§8)
    Qwen3.8-27B-UD-IQ4_XS.gguf          <- optional; only where it fits beside whisper
  prompts\
    block.txt
    summary_map.txt  summary_group.txt  summary_final.txt
  web\
    index.html  app.js  style.css
  server\
    main.py           <- FastAPI app, routes, SSE                  (kept, cut down)
    session.py        <- session state + lifecycle; replaces jobs.py's Job
    jobs.py           <- process cleanup, kill-on-close            (kept, cut down)
    capture.py        <- ffmpeg dshow → PCM pipe, device list, level meter   (new)
    vad.py            <- silero under onnxruntime → utterances               (new)
    asr.py            <- whisper-server lifecycle + client                   (new)
    notes.py          <- the 5-minute block loop                             (new)
    summary.py        <- Generate Full Summary: save dialog, model swap, files (new)
    history.py        <- output\ meeting folders: append, list, rename, delete (new)
    chunker.py reduce.py <- restored from the batch app, speakers stripped (§7.1)
    llm.py            <- llama-server lifecycle + chat calls       (kept, mostly as is)
    hardware.py       <- NVIDIA detection, VRAM, model choice     (kept, cut down)
    config.py         <- config load/merge/write                  (kept, new schema)
    open_browser.py   version.py   updater.py                     (kept)
  temp\               <- job.log and the running lock only. Never transcript content.
  output\             <- one folder per meeting: transcript, notes, summary (§11.1)
```

`server/audio.py` (the batch ffmpeg converter) survives only as a source of patterns for
`capture.py`, then goes. Do not keep code alive for a pipeline that no longer exists.

---

## 4. CAPTURE

```
bin\ffmpeg.exe -hide_banner -loglevel error ^
  -f dshow -audio_buffer_size 50 -i audio="<device name>" ^
  -ac 1 -ar 16000 -f s16le pipe:1
```

- **Read stdout in fixed blocks** (e.g. 512 samples = 32 ms, matching the VAD frame) on a
  dedicated thread. Never let the pipe back up: if the consumer stalls, ffmpeg's buffer
  fills, dshow drops audio, and the gap shows up later as words missing for no visible
  reason.
- `-audio_buffer_size` is in milliseconds. dshow's default is around 500 ms, which adds
  pure latency for nothing. Verify the option exists in the shipped build (§1).
- **Device choice.** List devices with `-list_devices true -f dshow -i dummy` and parse
  stderr for audio devices. dshow has no "default device" alias, so the UI shows a
  microphone dropdown (§11). The choice is remembered in `capture.device` (by name), the
  same way the batch app remembered the Port box. Re-enumerate every time the dropdown is
  opened and at every Start, never once at startup: the mic may be plugged in after the
  app has started.
- **The same mic on another USB port is a different device to Windows.** A USB mic with
  no serial number (the operator's has none — its instance path is port-derived) gets a
  new device instance on each port, with a new endpoint GUID. Windows also often renames
  the second instance, e.g. `Microphone (2- BOYA CM30)`. So:
  - **Never store the GUID** (dshow's `@device_cm_…\wave_{…}` alternative name). Store
    the friendly name.
  - **Match in this order:**
    1. the exact remembered name;
    2. the same name with Windows' instance prefix removed — `(\d+- )` just inside the
       parenthesis — compared case-insensitively, if exactly one present device matches;
    3. the Windows default recording device;
    4. the first audio device.

    Steps 1–2 are silent. When step 2 matches, update `capture.device` to the current
    name. Steps 3–4 show *"Couldn't find "<remembered>". Using "<chosen>" — change it
    above if that's wrong."*, and do not overwrite the remembered name until the user
    picks one.
  - With no audio device at all: Start is disabled and the status says *"No microphone
    found. Plug one in."*
- **Level meter.** Compute RMS per block and publish at ~5 Hz. This is the user's only way
  to know the right microphone is selected and actually hearing the room, before anything
  has been transcribed.
- **Mic disappears mid-session** (USB unplugged): ffmpeg exits. Stop the session cleanly,
  keep what is on screen, and say *"The microphone was disconnected."* Do not restart
  silently on another device.
- **Replay mode, for development only.** `capture.replay_file` (default `""`), when set,
  replaces the dshow input with `-re -i "<file>"`: the same pipe, paced at real time, from
  a recording. It is how every latency and long-session test in §16 is run reproducibly
  without holding a meeting. It never appears in the UI.

---

## 5. VOICE ACTIVITY DETECTION AND SEGMENTATION

Whisper hallucinates on silence and noise — fabricated sentences, *"Thank you."*, repeated
lines. In the batch app `--vad` was mandatory for exactly this reason. Live, it matters
more: a meeting room is mostly not speech at any given moment.

Run **Silero VAD v5** (`models/silero_vad.onnx`) under onnxruntime in Python: 512-sample
frames at 16 kHz with recurrent state carried between frames. It is about 2 MB and costs
microseconds per frame, so it runs in-process. (sherpa-onnx's `VoiceActivityDetector` wraps
the same model and would also work, but sherpa holds the GIL through its native calls. That
caused a deadlock in the batch app, so plain onnxruntime is preferred.) The `ggml-silero`
file the batch app used is ggml format and cannot be loaded by onnxruntime.

**An utterance** is a run of speech frames, padded and cut as follows. All values live in
`config.json` → `vad`:

| Setting | Default | Meaning |
|---|---|---|
| `threshold` | 0.5 | speech probability to count a frame as speech |
| `min_silence_ms` | 600 | silence that ends an utterance |
| `pad_ms` | 200 | audio kept before and after the speech |
| `min_speech_ms` | 250 | shorter than this is discarded |
| `max_utterance_s` | 15 | forced cut for a speaker who never pauses |

- `min_silence_ms` is the main latency knob. Lower makes the transcript snappier and
  fragments sentences; higher reads better and lags more.
- **The forced cut** at `max_utterance_s` happens at the lowest-speech-probability frame in
  the last 3 s, not at exactly 15.000 s. Cutting mid-word sends half a word to each side.
- **Do not raise `min_speech_ms` to suppress noise.** A one-word "No." can change what was
  decided. Hallucination is handled after recognition (§6), where the text is visible.
- Each utterance carries its start time in **session time** (seconds since Start, from the
  sample count, not the wall clock — the sample count cannot drift).
- **Every time shown to the user is real clock time, 12-hour** — `2:34:05 PM` on transcript
  lines, `2:30 – 2:35 PM` on a note, `2:34 PM` in the full summary — computed as the
  meeting's start (the first captured sample) plus session time. Never "time since Start"
  (operator's choice, 2026-11-11: this is a real-time app). Session time stays internal.

---

## 6. RECOGNITION — whisper-server

One `whisper-server.exe` for the whole session, model loaded once:

```
bin\whisper-cuda\whisper-server.exe ^
  -m models\ggml-large-v3-turbo.bin ^
  -l en ^
  -t 4 ^
  --host 127.0.0.1 --port 8081
```

Verify every flag (§1). Do not pass `--convert`: it shells out to ffmpeg and writes temp
files. We send clean 16 kHz WAV already.

**Per utterance:** wrap the PCM in a WAV header **in memory** and POST it to `/inference` as
multipart, with `response_format=verbose_json`, `language=en`, `temperature=0`. No temp
file.

- **`language=en` is what translates the Mandarin** (§0 scope). Do not set `translate`; it
  is a different task and large-v3-turbo was not trained for it.
- **Context carry.** If the server accepts a `prompt` field, pass the last ~200 characters
  of transcript. It helps names and terms stay consistent across utterances. It can also
  start a repetition loop, so it is behind `asr.carry_context` (default `true`), and the
  hallucination filter below runs regardless.
- **One request in flight.** Utterances queue in order. If the queue grows past
  `asr.max_queue` (default 4), whisper is not keeping up. Log it and show *"Transcription
  is falling behind"*. Never drop audio silently.

**Hallucination filter.** Drop a segment when either holds:

- `no_speech_prob > 0.6` and `avg_logprob < -1.0` (if the server reports them — verify);
- the utterance is under 2 s and its whole text is a known filler hallucination (*"Thank
  you."*, *"Thanks for watching."*, *"you"*, a lone *"."*). Keep the list in
  `config.json` → `asr.drop_phrases`.

Also collapse a segment that repeats the same phrase three or more times in a row. Log
every drop with its timestamp to `temp\job.log` — **the timestamp, not the text** (§10).

**Latency target:** text appears within **2 s of the speaker pausing**. That is
`min_silence_ms` (0.6 s) plus turbo inference on ≤15 s of audio (a few hundred ms on a
16 GB card), plus transport. Measure it (§16); do not assume it.

### 6.1 The refine pass — each finished block, transcribed again whole

Live lines are fast and rough. Whisper translates 3–5 s pieces far worse than the long
windows of a batch pass. Measured against batch-style transcription of the same audio
(`BUILD_NOTES.md` §3–4), live utterances differ by 50–68%, a whole 5-minute block by
21–31%, and two equally valid batch passes (CPU vs GPU) by 23% from each other. So block
refining reaches the noise floor, and the live lines do not. The operator approved a second
pass on 2026-11-11.

- **When each 5-minute block closes**, its audio is transcribed again, **whole, in one
  request**. Not in 60 s chunks: chunks measured 42%, or 38% with the previous text as a
  prompt, which loses most of the gain.
- **On a second whisper-server running on the CPU** (`-ng`), on its own port
  (`asr.refine_port`, 8082), launched with `--vad --vad-model models\ggml-silero-v5.1.2.bin`
  (batch quality used whisper's own VAD) and `-t asr.refine_threads`. The GPU instance
  serves live utterances only, so live lines never wait behind a refine.
  - Why not the GPU instance: it serves one request at a time, so a 300 s refine would
    stall live lines for 5–10 s every 5 minutes.
  - Why not a second GPU instance: +1.9 GiB, which does not fit beside the LLM.
  - Measured: 106 s per 300 s block at 12 threads, 122 s at 8. `refine_threads: "auto"`
    means half the logical cores, at least 4.
  - Verify `-ng` takes no VRAM. The CUDA backend DLL still loads; if it creates a context
    anyway, measure it against §8's headroom.
- **Block audio** runs from the block's first utterance start to its last utterance end,
  padded. Utterance boundaries are silences, so a block never splits a word. Utterances
  belong to the block their start falls in (§7).
- **Audio is held in memory only, and bounded:** the block being captured plus the one
  being refined, about 10 minutes, ~19 MB. It is released the moment its refine returns
  or fails. Never on disk (§10).
- **Refined lines replace the block's live lines** — on screen, in `transcript.md` and in
  the session. Their timestamps come from whisper's segment times, offset to session time.
  The §6 hallucination filter runs on them too.
- **The block's notes wait for the refined text**, so they are built from the better
  transcript. They land about 2 minutes after the block closes.
- **If a refine fails, or is still running when the next block closes,** keep the live
  lines for that block, log it and move on. Never queue a second refine behind a slow one:
  falling behind must degrade to rough text, not to growing delay.
- **Stop and Generate Full Summary** refine the final partial block before whisper is
  unloaded, while its audio is still in memory. A past meeting's audio is gone, so it
  keeps whatever was saved.
- **On screen:** live lines are shown in a muted grey until refined. The swap must not move
  what the reader is looking at: anchor the scroll position to the first line visible
  below the replaced range, so the column doesn't jump.
- **On disk:** `transcript.md` is rewritten atomically after each refine, with refined
  blocks first and the current block's live lines after them. Live lines are still appended
  as they arrive between refines, so a crash loses nothing (§11.1).
- **`asr.refine: false`** turns the whole pass off, giving live lines only.

**Out of scope for v1:** word-by-word partial text while someone is still speaking. That
needs re-decoding a sliding window several times a second, which is a different design with
its own GPU cost. Sentence-level updates are the agreed behaviour.

---

## 7. THE ROLLING NOTES — 5-MINUTE BLOCKS

The right-hand column shows what happened recently, as short notes, without the user having
to read the transcript.

**Blocks, not a sliding window.** Every 5 minutes of session time (`notes.block_s`, 300),
the utterances whose *start* falls in that block are summarised **once**, and the result
never changes. **Every block stays on screen, fully expanded, for the whole session.** Each
new block is **appended at the bottom**, and the column auto-scrolls with the same rule as
the transcript (§11). Nothing is ever collapsed or removed until the next Start. This was
the operator's explicit choice (2026-11-11), replacing an earlier newest-on-top design that
collapsed older blocks.

Why not re-summarise the last 15 minutes every minute: the notes would rewrite themselves
on every refresh, bullet points would reshuffle under the reader's eyes, and the same 15
minutes would cost the GPU three times over. Blocks are stable, append-only and cheap.

**Each call:**

- **Input:** the block's transcript lines with their clock times, and the
  **previous block's notes** as context, so the model can tell a continuing topic from a
  new one. Not the whole session: the prompt must stay the same size at hour three as at
  minute five.
- **Thinking off.** `chat_template_kwargs: {"enable_thinking": false}`. The batch app's map
  stage ran thinking-off over ~10,000-token chunks — ten times a block — and produced good
  notes. Thinking earned its cost only in cross-meeting synthesis, which this app does not
  do. Still strip `<think>…</think>` from every response (`llm.strip_thinking`).
- **Sampling:** the batch app's thinking-off set — `temperature 0.7, top_p 0.8, top_k 20,
  min_p 0, presence_penalty 1.5`. `max_tokens` 600.
- **Output:** see §9. Terse Markdown, rendered in the column.

**Edge cases:**

- **Near-silent block:** fewer than `notes.min_words` (40) words → no LLM call; show *"Little
  was said."*
- **Stop pressed:** summarise the partial final block if it has at least `min_words`, so the
  last minutes are not lost from the column.
- **A call fails:** retry once. Then show *"Notes for this block couldn't be written."* in
  that block's slot and carry on. A failed block must never stop transcription or later
  blocks.
- **Falling behind:** at most one call in flight and one queued. A third pending block
  means something is badly wrong — log it. At ~1,000 tokens in and ~300 out this should
  never happen (§8).
- **Pending state:** while a block is waiting or generating, show it as a placeholder with
  its time range, and in the column header show *"Next notes at 00:15:00"*.

### 7.1 Generate Full Summary

A button **below the transcript column**, labelled **Generate Full Summary**, with an ⓘ
info tooltip beside it: *"Stops listening and writes a summary of the whole meeting — the
topics discussed and what was concluded on each. It's kept with this meeting in the
history, and you can save a copy anywhere. Takes several minutes for a long meeting.
Listening can't be resumed afterwards."* (For a past meeting, the tooltip omits the
listening sentences.)

It is shown on the live meeting, and on any past meeting that has no summary yet, when the
transcript has at least `notes.min_words` words. Once a summary exists, the button is
replaced by the **Full Summary** tab (§11).

**On the live meeting, this ends the meeting.** It is final: no resuming, no appending.
The next Start begins a new, empty session.

**Flow — the user can back out until a location is chosen:**

1. **Confirm** — an in-page modal, not a browser `confirm()`: *"This will stop listening
   and generate a full summary of the meeting so far. It can take several minutes, and
   listening can't be resumed. Continue?"* — **Cancel** / **Continue**. Cancel changes
   nothing.
2. **Save dialog** — a native Windows *Save As* dialog for the **copy**, opened **by the
   server** (a PowerShell `System.Windows.Forms.SaveFileDialog`, owned by a hidden `TopMost`
   form so it appears in front of the browser). Filter `Markdown (*.md)`. Defaults: the
   user's **Desktop** folder, and the file name `<meeting name> - summary.md`. Overwrite
   prompt on. Cancel here also changes nothing, and listening continues.

   Why the server, not the browser: the summary takes minutes. A server-side path is
   still written if the tab is closed meanwhile. A browser-side handle
   (`showSaveFilePicker`) is lost with the tab. Verify the dialog actually comes to the
   front on Windows 11; a dialog hidden behind the browser looks like a hang.
3. **From here it is final.**
   - Live meeting only: stop capture, flush the open utterance through whisper, and
     abandon any pending 5-minute block. The transcript is already on disk (§11.1).
   - Unload whisper and the realtime llama-server.
   - Start llama-server for the summary (below) and run the summary.
   - Write `summary.md` into the meeting's history folder, then write the copy to the
     chosen path.
   - The **Full Summary** tab opens, showing the summary, with **Open file** (the copy)
     and **Open folder** buttons.

**Files.** The summary is written in two places: the meeting's history folder (the
record) and the user's chosen path (the copy). **No transcript is written beside the
copy**: the transcript lives in the history. The 5-minute notes stay in the history only.
If writing the copy fails (for example, the folder has gone), keep the history copy and say
*"Saved in the history, but not to <path>."*

**How the summary is made — the full transcript, High Quality.** Not from the 5-minute
notes: those are terse, lossy and from the 3-bit model.

- Once whisper is unloaded, the whole card is free, so the summary runs on **High Quality
  (`UD-IQ4_XS`) at `ctx_size` 32768 with MTP**. On a 16 GB card it fits whole (measured in
  the batch app). If IQ4_XS is absent, use IQ3_XXS and say so in the log. With Port, use
  the external server and load nothing.
- **The batch app's tested path, restored rather than reinvented.** Its `chunker.py`
  (`/tokenize`-based, ~10k-token chunks on utterance boundaries, 400-token overlap) and
  `reduce.py` (map → group reduce when chunks exceed 8 → final reduce, recursive), with
  their context guards. Take them from `NGJHD/Meeting-Summarizer` at commit `1559ecf` and
  strip speakers, minutes mode and calibration.
- **Map notes get 2,500 tokens, not the batch app's 1,500.** At 1,500 every map call hit
  the cap and lost the end of its chunk; at 2,500 a 3h43m meeting's summary recorded half
  as many decisions again (`BUILD_NOTES.md` §6).
- **Sampling as in the batch app:** map and group-reduce run with thinking off. The final
  reduce runs with thinking on, `reasoning_effort` medium, `--reasoning-budget 4000`,
  `max_tokens` 12000, and the thinking-off retry on an empty `length` finish.
- **Prompts:** `prompts/summary_map.txt`, `summary_group.txt` and `summary_final.txt`,
  adapted from the batch app's with all speaker-label rules removed. The map notes keep
  **Themes**, **Outcomes** and **Loose Ends** only.

**The document:**

```
# Meeting summary — <YYYY-MM-DD HH:MM>
*Duration: <h m>*

## <Topic>
<what was discussed, in paragraphs, ending with a sentence on how it ended:
 "They agreed at around 2:34 PM to…", or that it was left open and what remained>

## <Topic>
...
```

**Only topics, and only prose.** No executive summary, no participants, no separate
decisions or action-item list, no speaker attribution. **No bullets, numbered lists,
tables or bold labels** such as "Conclusion:" — sentences and paragraphs under one heading
per topic (operator's choices, 2026-11-11). Organise by theme, not in time order. Every
topic ends by saying how it concluded, or that it was left open.
Base everything on the transcript, flag ambiguity, and never invent a decision. The
machine-translation caveat from §9 applies.

**Progress.** A full-width panel replaces the Start/Stop controls. It shows:
*"Saving transcript…"* → *"Loading the High Quality model (up to 2 minutes)…"* →
*"Reading the meeting: part i of N"* → *"Writing the summary…"*. Use the batch app's
asymptotic per-call progress (`1 − e^(−elapsed/expected)`), not `tokens/max_tokens`.

**Cancel** is available throughout. It aborts the LLM work and unloads the model. The
meeting stays in the history with its transcript and notes, the button is enabled again
(it now works as for a past meeting), and Start stays available for a new meeting. Failure
behaves like Cancel, plus *"The summary couldn't be written. The meeting's transcript is
safe in the history."*

---

## 8. THE LLM AND VRAM

The llama-server launch, health polling, `/props` context read, streaming chat, retries,
`strip_thinking` and the external-server (Port) path are all in `server/llm.py` and carry
over largely unchanged. What changes is sizing.

**Default model: Low Quality, `Qwen3.8-27B-UD-IQ3_XXS`, thinking off, `ctx_size` 8192.**

- **Why 8192:** a block is ~5 min × ~200 tokens/min ≈ 1,000 tokens, plus the previous
  block's notes, the prompt and 600 output tokens. That is about 2,500 tokens. 32k (the
  batch default) is pure KV-cache waste, and VRAM is the scarce thing here.
- **Why Low Quality:** measured in the batch app, IQ3_XXS sat at 11.69 GiB resident with
  everything on a 16 GB card at 32k context. At 8k it is less. That leaves room for
  whisper turbo, which needs about 2 GiB including CUDA context and buffers — **measure
  this; do not trust the estimate**. High Quality (IQ4_XS, 13.27 GiB of weights) does not
  fit beside whisper on a 16 GB card without offloading FFN blocks to the CPU, which
  costs about 2.4× in generation speed.
- **Known trade-off:** below 4 bits, models start mangling proper nouns and digits. For
  glanceable notes over a transcript that is already partly a translation, that is
  accepted. The full summary (§7.1) does need 4-bit, and gets it: whisper is unloaded by
  then, so IQ4_XS has the whole card.

**`hardware.py` must account for whisper.** Add a measured `WHISPER_RESERVE_GIB` and
subtract it from VRAM before `choose_key`, `offload_regex` and the decision on whether
`--spec-type draft-mtp` fits. High Quality is chosen automatically only if it fits *whole*,
beside whisper, with its KV cache. On a 16 GB card that should resolve to Low Quality. On a
24 GB card it may resolve to High. `llm.model` set explicitly still wins.

**MTP.** `--spec-type draft-mtp` (the model's own multi-token-prediction layer) costs about
700 MiB and was worth about 50% generation speed in the batch app. Enable it only if it
still fits with whisper resident. Measure both ways, since outputs here are short.

**No prompt cache in RAM.** Our llama-server runs with `--cache-ram 0`. This build caches
past prompts in system RAM, 8 GiB by default, and none of ours repeat: over a 3h43m
meeting it grew the process from 11.6 to 19 GB for nothing (`BUILD_NOTES.md` §6).

**Reasoning budget.** Thinking is off, so `--reasoning-budget` is irrelevant on our own
server. Leave the code path in place: the Port option below may face a server whose
template ignores `enable_thinking`.

**Port (external server).** The Model dropdown's third entry, kept from the batch app. It
sends the LLM calls to a llama-server the user already runs, at `127.0.0.1:<port>`. That
server is not ours: never started or killed by us, and its VRAM is not ours to budget.
**If it lives on the same GPU, whisper may not fit.** Detect it: if `whisper-server` fails
to load or `nvidia-smi` shows less free memory than `WHISPER_RESERVE_GIB`, say *"Not enough
video memory for transcription alongside the model on port N."* Never let the app hang.

**Measured (`BUILD_NOTES.md` §2):** on a single 16 GB card, IQ3_XXS + MTP + whisper peaks
at 13.9 GiB. Worst-case utterance latency while a block generates is 1.8 s, and a block
takes 5–8 s. IQ4_XS also squeezes in beside whisper, but with only 0.33 GiB to spare and
no room for MTP, so it is not offered on one 16 GB card.

**One card, even when there are several.** The target is a single 16 GB card. Where a
machine has more than one NVIDIA GPU, put **both** servers on the one with the most VRAM —
`--device` for llama-server, `-dev` for whisper-server — and never split them across cards.
The operator decided this on 2026-10-30 so that the app behaves the same wherever it is
deployed. A two-card split measured better (`BUILD_NOTES.md` §2, layout B), but it is not
supported. Choose the card **by name and VRAM, never by index**: CUDA and `nvidia-smi`
number the cards differently, and both servers default to CUDA device 0, which on the dev
machine is the 10 GB card.

**Lifecycle.** Start loads whisper-server and llama-server in parallel, with the message
*"Loading models (up to a minute)…"*. **Then warm whisper up** with one inference on a
short noise clip before going live. On a GPU the CUDA build has no native kernels for
(compute 12.0 against the CUDA 12.4 build), the first inference JIT-compiles for about a
minute. Without a warm-up, the first utterance of the first meeting arrives a minute late
and everything queues behind it. Stop keeps both loaded for `llm.idle_timeout_s`
(300), so pressing Start again is instant. After that, both exit and the VRAM is released.
Closing the app kills both, always (§12).

**GPU contention.** While a block summary generates (~10–20 s every 5 minutes), whisper
shares the GPU. Expect utterance latency to rise during that window. Measure the worst case
(§16). If it exceeds ~5 s, the fallbacks in order are: a smaller `ubatch-size` for the LLM
so whisper gets scheduled between batches; then whisper on fewer, larger requests; and
only then whisper on the CPU. Do not reach for the CPU first — turbo on the CPU is
borderline for real time.

---

## 9. PROMPT — `prompts/block.txt`

Plain text, loaded at session start, placeholders filled with `str.replace` (the prompts
contain literal braces): `{{TRANSCRIPT}}`, `{{PREVIOUS_NOTES}}`, `{{TIME_START}}`,
`{{TIME_END}}`.

Rules the prompt must carry:

- Base everything strictly on the transcript. Never infer decisions, numbers, deadlines or
  owners that were not stated.
- The transcript is machine-generated and partly machine-translated from Mandarin. Treat
  inconsistent spellings of a term as one thing. Do not comment on the translation.
- There are no speaker labels. Attribute a point to a person only when the transcript names
  them saying it.
- Flag ambiguity and disagreement rather than smoothing it over.
- The previous block's notes are context only. Do not repeat them. Say *"continued"* where a
  topic carries over.
- **Plain and short.** Read at a glance during a meeting. No preamble, no closing line.

Output shape: **one to three short paragraphs of plain English, about 60–120 words** —
what was discussed and what came of it (agreed, promised, left open). **No timestamps,
no headings, no bullets, no bold labels.** Operator's choice, 2026-11-11, replacing an
earlier bulleted format with timestamps: the note is read at a glance to catch up, and
the transcript beside it already carries the times.

---

## 10. PRIVACY — TEXT IN ONE PLACE, AUDIO NOWHERE

- **Audio** goes from the pipe into memory and is released once its utterance has been
  recognised. The VAD keeps at most `max_utterance_s` plus padding. **Audio never touches
  disk**, not even in `temp\`.
- **Meeting text lives in exactly two places:** the meeting's folder under `output\`
  (§11.1), and the summary copy at the path the user chose (§7.1). Nowhere else.
- **`temp\job.log`** records timings, queue depths, token counts, latencies, drop counts
  and errors. **Never transcript text, never note text, never prompt or response bodies.**
  A log is not where the user looks for, or deletes, meeting content. Deleting a meeting
  from the history must actually delete it, and a copy left in a log would defeat that.
- Diagnostic info copied by the user follows the same rule.
- Nothing is written to `temp\` during a full summary: chunks, map notes and prompts stay
  in memory. The §17 rule against writing under `%USERPROFILE%` covers the app's own state.
  It does not cover a file the user explicitly saves to their Desktop.
- **Never put anything from a real meeting into anything published** — commit messages,
  release notes, `BUILD_NOTES.md`, prompts, code comments or test fixtures. That means names
  of people, organisations, products, systems, projects and sites; acronyms; figures; quoted
  lines; and anything a model wrote from one. When a finding needs an example, describe it
  structurally or invent a neutral one and do not present it as real. Before every commit or
  release, re-read the message and the diff for anything that came from a meeting.

---

## 11. USER INTERFACE

Single page, no framework, no bundler, no external assets. **Look: a deep ink-blue,
teal and violet gradient with frosted-glass panes** (`backdrop-filter`), Segoe UI
Variable (ships with Windows 11, so no web font), sentence-case labels — the operator
asked for something sleeker than a stock Windows form (2026-11-11). The live transcript
is the one loud element: large, light type. It sits on a desktop monitor for hours, so
it must be readable from a chair away and must not flicker.

**Top bar:**

1. **Microphone** dropdown — dshow devices. Remembered in `capture.device`.
2. **Model** dropdown — Low Quality / High Quality / Port, as in the batch app. Defaults to
   detection (§8), except that Port is remembered.
3. **Start / Stop** button — one button, two states. Disabled while models load.
4. **Status** — *Loading models…* / *Listening* / *Stopped* / an error sentence, plus elapsed
   session time.
5. **Level meter** — a glowing line along the top edge of the transcript pane, driven by
   §4's RMS. It is live whenever a microphone is selected, including before Start, so the
   user can check the mic first.
6. **About** — version, repository, *Check for updates* (kept from the batch app).

**Body: two columns.**

- **Left, ~65%: transcript.** One line per utterance: `(2:34:05 PM) text`. It auto-scrolls
  while the user is at the bottom. If they scroll up, auto-scroll stops and a *Jump to live*
  button appears. New lines must not shift what the user is reading.
- **Tabs on the transcript column: Transcript | Full Summary.** The Full Summary tab
  exists only once a summary has been generated for the meeting on screen, and shows it
  rendered. Without one, **Generate Full Summary** with its ⓘ tooltip (§7.1) sits below
  the transcript.
- **Right, ~35%: notes.** Blocks as in §7: appended at the bottom, all expanded, never
  removed. The pending block is shown as a placeholder at the bottom. Auto-scroll and
  *Jump to live* work as in the transcript column.

**After Stop, and after a full summary,** the meeting stays on screen until the next
Start, or until another meeting is picked from the history. There is no other save or
export (§0.6).

### 11.1 History sidebar

A collapsible sidebar on the **left**, listing every meeting in `output\`, newest first.

- **Entry label:** the meeting's name, which defaults to its start time, plus its
  duration — e.g. *"11 Nov 2026, 14:30 · 1h 52m"*. A ✓ marks meetings that have a full
  summary. The live meeting sits at the top with a ● and a running duration.
- **Clicking an entry** loads that meeting's transcript, notes and (if any) full summary
  into the two columns, **read-only**.
- **Rename** (pencil icon, inline edit) changes the display name only. The folder name
  never changes, so nothing can break by renaming.
- **Delete** (bin icon) asks in-page *"Delete this meeting's transcript, notes and summary?
  This can't be undone."* and removes the folder. The live meeting cannot be deleted.
- **Browsing while live is allowed.** Listening, transcription and notes carry on in the
  background, and a bar across the top reads **"● Live — back to meeting"**. Clicking the
  live entry or the bar returns to it. Start/Stop always act on the live meeting, never on
  the one being viewed.
- **Short sessions are not kept.** At Stop, a meeting with fewer than `notes.min_words`
  transcribed words has its folder removed and never appears in the list, so a quick mic
  check leaves nothing behind.
- **No automatic clean-up.** It is text; a year of daily meetings is a few tens of MB.

**On disk** — one folder per meeting, named from the start time so it is unique and sorts
naturally. Written by `history.py`:

```
output\
  2026-11-11 1430\
    meeting.json      <- {"name", "started", "duration_s", "has_summary"} — rewritten atomically
    transcript.md     <- "(2:34:05 PM) text" lines, appended as each utterance is recognised
    notes.md          <- one "## 2:30 – 2:35 PM" section per block, appended as each lands
    summary.md        <- written once by §7.1, if generated
```

- **Append as it happens**, and flush after every write. A crash, a power cut or a closed
  console loses at most the utterance in flight. Nothing is held back until Stop.
- **`meeting.json`** is rewritten via `config.write_atomic` (temp file + replace) on each
  block and at Stop, so a crash never leaves it half-written. The sidebar reads
  `meeting.json` and falls back to the folder name if the file is missing or corrupt.
- **A meeting interrupted by a crash** is found at startup with no clean Stop. Fix up its
  duration from the last transcript line and list it normally. The partial block at the
  crash is never summarised.
- **Folder name collision** (two Starts within the same minute): append ` (2)`.

**Transport: Server-Sent Events** on `GET /api/events`. Event types: `status`, `level`,
`utterance`, `block_pending`, `block`, `summary_progress`, `summary_done`, `error`. `GET /api/session` returns the full current
state.

**Reattaching.** On load the page fetches `/api/session`, renders everything, then attaches
to the event stream. Closing the tab must not stop the session — the server keeps
listening, and reopening the page brings back everything. The browser opens before the
server is ready, so the page must retry its first connection rather than show an error.

**Never show a raw exception.** Map failures to plain sentences. The full traceback goes to
`temp\job.log` (which still follows §10).

**No `localStorage` or `sessionStorage`.** UI choices that persist go through the server
into `config.json`.

---

## 12. CANCELLATION, SHUTDOWN, CLEANUP

There are four child processes (`ffmpeg`, both `whisper-server`s, `llama-server`), all
spawned with `CREATE_NEW_PROCESS_GROUP` and assigned to the kill-on-close Job Object
(`jobs.install_kill_on_close`). Force-quitting the console must leave **none** of them
running.

- **Stop:** terminate ffmpeg (`taskkill /F /T /PID`), let the VAD flush its open utterance,
  let whisper finish the queue, refine the partial block (§6.1), then summarise it (§7).
  The servers stay up
  for `idle_timeout_s`.
- **Shutdown** (console closed, FastAPI shutdown, `atexit`, `SIGBREAK`): kill all four, no
  flush.
- **Startup:** sweep `temp\` and kill any stale `whisper-server.exe` / `llama-server.exe` on
  our ports left by a previous crash — a stale one holds VRAM and the new one will fail to
  load beside it.
- **External (Port) server:** never killed, ever.

---

## 13. WHAT TO STRIP FROM THE KEPT CODE

These carry batch-app concepts that no longer exist. Remove them as each file is brought up,
and do not port them:

- **`main.py`:** upload, start-job, speakers/rename, rebuild, mode routes,
  `estimate_minutes`, disk-space checks. Keep `/`, static files, models, about, update,
  health and the SSE plumbing. The batch `history` and `history/open` routes are a starting
  point for §11.1's list and open, but the meeting folder format is new.
- **`jobs.py`:** the staged `Job` with its stage weights, percentages and ETA. Keep process
  tracking, the job log, `clean_temp`, `shutdown_all` and the Job Object code. Session state
  moves to `session.py`.
- **`hardware.py`:** the Vulkan probe, unified-memory handling, `_pick_device`, the coopmat
  workaround, the CPU backend and `LEGACY_MODELS`. Keep `nvidia-smi` detection, the model
  table, `model_gib`, `offload_regex` and `choose_key`, and add the whisper reserve (§8).
- **`config.py`:** `REQUIRED_FILES`, `_DEFAULTS` and the backend-per-engine paths, all
  rewritten for §14's schema. `migrate_flat_output` and the `ui.mode` helpers go.
  `meeting_dir` and `write_atomic` stay for §11.1. `load_config` must keep merging new keys from `_DEFAULTS`, so an old
  `config.json` loses nothing.
- **`llm.py`:** `loading_message` and anything that reads `job.stage`. The rest stays.
- **`version.py`:** already pointed at this repo. `EXPECTED_FILES` must list files that
  exist here.
- **`updater.py`:** the extra-model offer for the 1.0.x Q4_K_M migration.
- **`web/`:** `app.js` is mostly rewritten. Keep the SSE reconnect logic, the About overlay
  and the update flow. `style.css` largely carries.
- **`run.bat`:** preflight for this file set (whisper-server, silero_vad.onnx, at least one
  GGUF unless Port is on). Keep creating `output\`.
- **`DOWNLOAD_MODELS.bat`:** drop the diarization models, the alignment model and the
  Vulkan and CPU binaries. **Keep** `ggml-silero-v5.1.2.bin` for the refine pass (§6.1).
  Add `silero_vad.onnx`, pinned to a release tag of
  `snakers4/silero-vad`, never a branch head. Keep IQ4_XS as optional.
- **`THIRD_PARTY_NOTICES.md`:** remove pyannote, sherpa-onnx, NeMo and wav2vec2. Add Silero
  VAD (MIT).
- **`tools/make_release.py`, `check_release.py`:** update the file lists and exclusions.

---

## 14. CONFIGURATION

All tunables live in `config.json`. **There is no settings screen.** The UI writes only
`capture.device` and `llm.external`.

```json
{
  "capture": {
    "device": "",
    "block_ms": 32,
    "replay_file": ""
  },
  "vad": {
    "model": "models/silero_vad.onnx",
    "threshold": 0.5,
    "min_silence_ms": 600,
    "pad_ms": 200,
    "min_speech_ms": 250,
    "max_utterance_s": 15
  },
  "asr": {
    "model": "models/ggml-large-v3-turbo.bin",
    "language": "en",
    "threads": 4,
    "port": 8081,
    "carry_context": true,
    "max_queue": 4,
    "drop_phrases": ["Thank you.", "Thanks for watching.", "you", "."],
    "refine": true,
    "refine_port": 8082,
    "refine_threads": "auto",
    "refine_vad_model": "models/ggml-silero-v5.1.2.bin"
  },
  "notes": {
    "block_s": 300,
    "min_words": 40,
    "max_tokens": 600
  },
  "summary": {
    "model": "iq4_xs",
    "ctx_size": 32768,
    "target_tokens": 10000,
    "overlap_tokens": 400,
    "max_map_output_tokens": 2500,
    "group_reduce_threshold": 8,
    "group_size": 5,
    "max_group_output_tokens": 2500,
    "final_thinking": true,
    "final_effort": "medium",
    "final_budget_tokens": 4000
  },
  "llm": {
    "model": "auto",
    "ctx_size": 8192,
    "gpu_layers": "auto",
    "cpu_ffn_regex": "auto",
    "cache_type_k": "q8_0",
    "cache_type_v": "q8_0",
    "threads": "auto",
    "port": 8080,
    "startup_timeout_s": 180,
    "idle_timeout_s": 300,
    "spec_type": "auto",
    "external": { "enabled": false, "port": 9931 }
  },
  "server": { "port": 8000 }
}
```

Everything marked `"auto"` is detected per machine. An explicit value always wins.

---

## 15. BUILD ORDER

Confirm each step before moving on. State as of 2026-11-11 in brackets.

1. **Verify binaries** per §1. Obtain `whisper-server.exe`. Start `BUILD_NOTES.md`. [done]
2. **Spike — this decides whether the design holds.** No UI. A script that loads
   whisper-server and IQ3_XXS llama-server (ctx 8192) together, feeds a recorded meeting
   through replay capture (§4) at real time, and fires one block summary mid-stream.
   Record peak VRAM, utterance latency (median and worst) with and without the LLM call
   running, and block-call time. Repeat with MTP on and off, and with IQ4_XS +
   offload. **Report before going further.** [done — `BUILD_NOTES.md` §2]
3. **Strip and boot** — cut `main.py`, `jobs.py`, `config.py` and `hardware.py` down (§13)
   until the app starts and serves an empty two-column page with the About overlay. [done]
4. **Capture + level meter + device list** — the meter moves when you speak. [done; level
   verified on a replay through the room speaker, not yet with a person speaking]
5. **VAD + whisper → transcript column** — live, with the hallucination filter. [done]
5a. **Refine pass** (§6.1) — CPU whisper-server, block audio ring, line swap with scroll
    anchoring. [done]
6. **Block notes → notes column.** [done]
7. **Stop, reattach, idle unload, shutdown, kill-on-close.** [stop, idle unload and clean
   shutdown verified; reattach and force-quit not yet]
7a. **Generate Full Summary** (§7.1): confirm modal, server-side Save dialog, transcript
    file, model swap, restored chunker/reduce, summary file, Cancel. Verify the three-tier
    path by lowering `summary.group_reduce_threshold` to 2 on a short replay. [done via the
    API; the Save dialog and Cancel not yet tried by hand]
8. **Long-session test** — 3 hours of replay. Memory, latency and block timing at hour three
   match hour one. [done — 3h43m, `BUILD_NOTES.md` §6]
9. **Packaging** — `run.bat`, `DOWNLOAD_MODELS.bat`, notices, release scripts, README.
   [done; no release built or published yet]
10. **Error message polish.** [not yet]

---

## 16. ACCEPTANCE TESTS

1. **Latency:** median text-after-pause ≤ 2 s; worst case ≤ 5 s *while a block summary is
   generating*. Measured with replay over at least 30 minutes.
2. **No lost speech:** over a 30-minute replay, every stretch of speech in the source
   produces transcript. Check by sampling ten points at random. A one-word utterance
   survives.
3. **Silence:** 10 minutes of a quiet room produces no transcript lines at all.
4. **Mixed language:** a stretch of Mandarin-mixed speech produces readable English, not
   Chinese characters and not an empty line.
5. **Blocks:** a block's notes appear within 3 minutes of its boundary (refine, then
   notes) and never change afterwards. The partial block appears after Stop.
5a. **Refine:** over a 30-minute replay, every block's lines are replaced by refined ones.
    Refined text against batch-style `whisper-cli` on the same audio is within about 35%
    word error (live lines alone measured 50–68%). Live latency stays within test 1 while
    a refine runs. The view does not jump when a block is swapped.
6. **Notes are grounded:** spot-check ten points against the transcript. No invented owners,
   deadlines or figures.
7. **Three hours:** completes with flat memory (apart from transcript text), stable latency,
   and no queue growth.
8. **VRAM:** peak during a block call is recorded in `BUILD_NOTES.md` and stays under the
   card's total with at least 0.5 GiB to spare.
9. **Reattach:** closing and reopening the tab mid-session shows the full transcript and
   notes and continues live.
10. **Unplug the mic mid-session:** a plain message, the session stops, on-screen content is
    kept, the app remains usable.
11. **Force-quit the console:** no `ffmpeg.exe`, `whisper-server.exe` or `llama-server.exe`
    remains in Task Manager.
12. **Network adapter disabled:** a full session runs.
13. **Text in one place:** after a session, a distinctive word you said appears only under
    that meeting's `output\` folder (and in the summary copy, if made). Grep the whole app
    folder for it, `temp\job.log` included. No audio file exists anywhere.
16. **Full summary:** on a 1-hour replay, Generate Full Summary → confirm → Save writes the
    copy to the Desktop with the default name and `summary.md` into the history folder. No
    transcript is written beside the copy. The summary contains only topic sections, each
    ending in a conclusion or "left open", and no speaker labels. Cancelling at the modal or
    at the Save dialog leaves listening running.
17. **Full summary, cancelled midway:** no summary file is written anywhere, no
    `llama-server.exe` remains, the meeting is in the history with its transcript, and
    Start begins a new session.
18. **Save dialog in front:** the dialog appears above the browser window, not behind it.
19. **History:** a finished meeting appears in the sidebar. Clicking it shows its
    transcript, its notes and (if generated) a Full Summary tab. Rename survives a restart.
    Delete removes the folder.
20. **Crash safety:** kill the console 20 minutes into a replay. On restart the meeting is
    in the history with every utterance up to the kill.
21. **Browse while live:** open a past meeting mid-session. Transcription continues, the
    live bar shows, and going back shows every line spoken meanwhile.
22. **Summary from the past:** a past meeting with no summary gets one, and the ✓ appears.
23. **Mic check leaves nothing:** Start, say two words, Stop. No new history entry.
14. **Port:** with an external llama-server, notes are written by it and it is never killed.
    If it leaves no room for whisper, a plain message appears, not a hang.
15. **Portable:** copy the folder to another drive letter and it runs unchanged.

---

## 17. THINGS NOT TO DO

- Do not write audio to disk, ever.
- Do not write meeting text anywhere except the meeting's `output\` folder and the summary
  copy the user chose (§10). Do not add any other export or save.
- Do not hold transcript or notes back until Stop. Append as they happen (§11.1).
- Do not add diarization or speaker labels, not even in the full summary.
- Do not let Generate Full Summary resume listening afterwards. It ends the meeting.
- Do not switch recognition to language auto-detection or a Chinese-output model (§0
  scope).
- Do not re-summarise a sliding window. Blocks are summarised once (§7).
- Do not let transcription wait on the LLM, ever.
- Do not add Vulkan, CPU-only or unified-memory support back.
- Do not add a settings screen, sliders or temperature controls. Tunables are in
  `config.json`.
- Do not add any dependency that pulls in torch, transformers or the HuggingFace hub client.
  If you find yourself needing one, stop and report it.
- Do not reference any external URL from the frontend.
- Do not bind to anything except `127.0.0.1`.
- Do not write to `%APPDATA%`, `%USERPROFILE%` or the registry.
- Do not use `localStorage` or `sessionStorage`.
- Never put anything from a real meeting into anything published (§10).

---

## 18. REPORT BACK ON COMPLETION

- Every flag or request parameter from this document that differed in the shipped binaries,
  and what was used instead.
- Peak VRAM with both servers resident, idle and during a block call. The measured
  `WHISPER_RESERVE_GIB`.
- Utterance latency: median and worst, with and without a concurrent block call.
- Block-call time and tokens/second, with MTP on and off.
- Whether IQ4_XS fits beside whisper on the 16 GB card at ctx 8192 without offload, and
  if not, the latency cost of using it with offload.
- Memory growth over three hours.
- Final unzipped folder size.
