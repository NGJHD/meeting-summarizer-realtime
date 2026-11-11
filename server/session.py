"""The live meeting: capture -> VAD -> whisper -> transcript -> refine -> notes.

CLAUDE.md sections 4-7 and 11-12. One meeting at a time. Four threads, joined
by queues, none of which ever waits on the language model:

    capture (reader)  frames -> VAD -> utterances, block boundaries   -> asr_q
    asr               GPU whisper per utterance -> live lines         -> refine_q
    refine            CPU whisper per whole block -> refined lines    -> notes_q
    notes             llama-server per block -> notes

Audio lives in memory only, bounded to about two blocks (section 10).
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from typing import Optional

import numpy as np

from . import asr, capture, config, hardware, history, jobs, llm, vad
from .jobs import JobError, Task

SR = 16000

NO_MIC = "No microphone found. Plug one in."
MIC_LOST = "The microphone was disconnected."
FALLING_BEHIND = "Transcription is falling behind."
NOT_ENOUGH_VRAM = ("Not enough video memory for transcription alongside the model "
                   "on port %d.")


# ---------------------------------------------------------------------------
# events to the page
# ---------------------------------------------------------------------------

class EventBus:
    """Server-Sent Events fan-out. emit() is safe from any thread."""

    def __init__(self):
        self._queues: list[asyncio.Queue] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._lock = threading.Lock()

    def subscribe(self) -> asyncio.Queue:
        self._loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue(maxsize=2000)
        with self._lock:
            self._queues.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._lock:
            try:
                self._queues.remove(q)
            except ValueError:
                pass

    @property
    def listeners(self) -> int:
        return len(self._queues)

    def emit(self, event: dict) -> None:
        loop = self._loop
        if loop is None or not loop.is_running():
            return
        with self._lock:
            queues = list(self._queues)
        for q in queues:
            try:
                loop.call_soon_threadsafe(_put, q, event)
            except RuntimeError:
                pass


def _put(q: asyncio.Queue, event: dict) -> None:
    try:
        q.put_nowait(event)
    except asyncio.QueueFull:
        pass                         # a stalled tab must not grow memory


bus = EventBus()


# ---------------------------------------------------------------------------
# the resident models
# ---------------------------------------------------------------------------

class Engines:
    """GPU whisper, CPU whisper and the live llama-server, loaded together.

    Kept loaded for `llm.idle_timeout_s` after Stop, so Start again is instant;
    then everything exits and the VRAM is released (section 8).
    """

    def __init__(self):
        self.task = Task("engines")
        self.gpu: Optional[asr.WhisperServer] = None
        self.cpu: Optional[asr.WhisperServer] = None
        self.llama: Optional[llm.LlamaServer] = None
        self.llama_error = ""
        self.refine_error = ""
        self.choice = ""             # "external" or a model key, as loaded
        self._lock = threading.Lock()
        self._idle_timer: Optional[threading.Timer] = None

    def cancel_idle(self) -> None:
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None

    def schedule_idle_stop(self, seconds: float) -> None:
        self.cancel_idle()
        self._idle_timer = threading.Timer(seconds, self.stop_all)
        self._idle_timer.daemon = True
        self._idle_timer.start()

    def ensure(self, cfg: dict, choice: str, status) -> None:
        """Load whatever is not already loaded. Raises JobError if whisper cannot run."""
        self.cancel_idle()
        with self._lock:
            self.task.cancel_event.clear()
            external = choice == llm.EXTERNAL
            if self.llama is not None and (self.choice != choice or not self.llama.alive()):
                self.llama.stop()
                self.llama = None
            if external:
                need = hardware.WHISPER_RESERVE_GIB * 1024
                if (self.gpu is None or not self.gpu.alive()) and \
                        0 < hardware.free_vram_mb() < need:
                    raise JobError(NOT_ENOUGH_VRAM % config.external_llm(cfg)[1],
                                   "free VRAM below whisper's reserve with Port on")

            errors: dict[str, BaseException] = {}

            def load_gpu():
                if self.gpu is not None and self.gpu.alive():
                    return
                w = asr.WhisperServer(self.task, cfg, gpu=True)
                w.start()
                status("Loading models (up to a minute)…")
                w.warm_up()
                self.gpu = w

            def load_cpu():
                if not cfg["asr"].get("refine", True):
                    return
                if self.cpu is not None and self.cpu.alive():
                    return
                w = asr.WhisperServer(self.task, cfg, gpu=False)
                w.start()
                self.cpu = w

            def load_llama():
                if self.llama is not None and self.llama.alive():
                    return
                if external:
                    server = llm.LlamaServer(self.task, cfg, external=True)
                else:
                    key = choice if choice in hardware.BY_KEY else hardware.choose_live_key()
                    server = llm.LlamaServer(self.task, cfg, hardware.live_plan(key, cfg))
                server.start()
                self.llama = server

            threads = []
            for name, fn in (("gpu", load_gpu), ("cpu", load_cpu), ("llama", load_llama)):
                def run(fn=fn, name=name):
                    try:
                        fn()
                    except BaseException as exc:  # noqa: BLE001 - collected below
                        errors[name] = exc
                t = threading.Thread(target=run, name="load-" + name, daemon=True)
                t.start()
                threads.append(t)
            for t in threads:
                t.join()
            self.choice = choice

            if "gpu" in errors:
                exc = errors["gpu"]
                jobs.log_exception(exc)
                raise exc if isinstance(exc, JobError) else JobError(asr.ASR_FAILED, str(exc))
            self.refine_error = ""
            if "cpu" in errors:
                jobs.log_exception(errors["cpu"])
                self.refine_error = "The second transcription pass is unavailable."
            self.llama_error = ""
            if "llama" in errors:
                exc = errors["llama"]
                jobs.log_exception(exc)
                self.llama = None
                self.llama_error = (exc.message if isinstance(exc, JobError)
                                    else llm.LLM_FAILED)

    def stop_all(self) -> None:
        with self._lock:
            self.cancel_idle()
            for name in ("gpu", "cpu", "llama"):
                server = getattr(self, name)
                setattr(self, name, None)
                if server is not None:
                    try:
                        server.stop()
                    except Exception as exc:  # noqa: BLE001
                        jobs.log_exception(exc)

    def stop_whisper_and_live_llm(self) -> None:
        """Before the full summary: give its model the whole card (section 7.1)."""
        self.stop_all()

    @property
    def loaded(self) -> bool:
        return self.gpu is not None and self.gpu.alive()


engines = Engines()


# ---------------------------------------------------------------------------
# the live meeting
# ---------------------------------------------------------------------------

class AudioRing:
    """Captured PCM since `base_s`, in memory only, trimmed as blocks finish."""

    def __init__(self):
        self.chunks: list[np.ndarray] = []
        self.base_frame = 0
        self._lock = threading.Lock()

    def append(self, pcm: np.ndarray) -> None:
        with self._lock:
            self.chunks.append(pcm)

    def slice(self, start_s: float, end_s: float) -> np.ndarray:
        with self._lock:
            first = max(0, int(start_s * SR / capture.FRAME) - self.base_frame)
            last = max(first, int(end_s * SR / capture.FRAME) + 1 - self.base_frame)
            parts = self.chunks[first:last]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int16)

    def trim_before(self, s: float) -> None:
        with self._lock:
            drop = int(s * SR / capture.FRAME) - self.base_frame
            if drop > 0:
                del self.chunks[:drop]
                self.base_frame += drop

    @property
    def seconds_held(self) -> float:
        return len(self.chunks) * capture.FRAME / SR


class Session:
    def __init__(self, cfg: dict, device: str, meeting: history.Meeting):
        self.cfg = cfg
        self.device = device
        self.meeting = meeting
        self.task = Task("session")
        self.block_s = float(cfg["notes"].get("block_s", 300))
        self.min_words = int(cfg["notes"].get("min_words", 40))
        self.state = "live"               # live | stopping | stopped
        self.started = time.time()
        self.ended = 0.0
        self.lock = threading.Lock()
        self.lines: list[dict] = []       # {id, t, text, refined, block}
        self.blocks: dict[int, dict] = {}  # index -> {index, start_s, end_s, status, notes}
        self.utt_spans: dict[int, list] = {}
        self.prev_notes = ""
        self.skip_notes = False
        self.behind = False
        self.warning = engines.llama_error or engines.refine_error
        self._next_id = 0

        self.silero = vad.Silero(config.resolve(cfg["vad"].get("model", "models/silero_vad.onnx")))
        self.segmenter = vad.Segmenter.from_config(cfg["vad"])
        self.ring = AudioRing()
        self.cur_block = 0
        self.pending_close: list[int] = []
        self.refine_busy = False

        self.asr_q: queue.Queue = queue.Queue()
        self.refine_q: queue.Queue = queue.Queue()
        self.notes_q: queue.Queue = queue.Queue()
        self._threads = [
            threading.Thread(target=self._asr_worker, name="asr", daemon=True),
            threading.Thread(target=self._refine_worker, name="refine", daemon=True),
            threading.Thread(target=self._notes_worker, name="notes", daemon=True),
        ]
        replay = str(cfg["capture"].get("replay_file") or "")
        self.capture = capture.Capture(self.task, device, replay_file=replay,
                                       on_frame=self._on_frame, on_level=_emit_level,
                                       on_end=self._on_capture_end)
        self.replay = bool(replay)

    # -- lifecycle -------------------------------------------------------

    def start(self) -> None:
        for t in self._threads:
            t.start()
        self.capture.start()
        # Session time 0 is the first captured sample, so clock times shown
        # everywhere are this plus the sample count (section 5).
        self.started = time.time()
        self.meeting.update_meta(started=self.started)
        self.task.log("session: listening (%s)" % ("replay" if self.replay else "microphone"))

    def _on_capture_end(self) -> None:
        """ffmpeg ended without Stop: replay finished, or the mic was unplugged."""
        if self.state != "live":
            return
        if not self.replay:
            self.task.log("session: capture ended -- microphone disconnected")
            bus.emit({"type": "error", "message": MIC_LOST})
        threading.Thread(target=app.stop, name="stop", daemon=True).start()

    def stop(self, for_summary: bool = False) -> None:
        """Flush, refine the partial block, write its notes (unless summarising)."""
        if self.state != "live":
            return
        self.state = "stopping"
        self.skip_notes = for_summary
        self.capture.stop()
        end_s = self.capture.seconds
        utt = self.segmenter.flush()
        if utt is not None:
            self.asr_q.put(("utt", utt))
        for k in self.pending_close:
            self.asr_q.put(("close", k, (k + 1) * self.block_s))
        self.pending_close = []
        self.asr_q.put(("close", self.cur_block, end_s))
        self.asr_q.put(None)
        for t in self._threads:
            t.join()
        self.ended = time.time()
        self.state = "stopped"
        self.meeting.update_meta(duration_s=end_s, ended=True)
        self.task.log("session: stopped after %.0fs of audio, %d lines" % (end_s, len(self.lines)))

    # -- capture thread --------------------------------------------------

    def _on_frame(self, pcm: np.ndarray, idx: int) -> None:
        self.ring.append(pcm)
        p = self.silero.prob(pcm)
        utt = self.segmenter.push(pcm, p, idx, time.time())
        if utt is not None:
            self.asr_q.put(("utt", utt))
        pos_s = (idx + 1) * capture.FRAME / SR
        if pos_s >= (self.cur_block + 1) * self.block_s:
            self.pending_close.append(self.cur_block)
            self.cur_block += 1
        # A block closes once nothing that started inside it is still open, so
        # its last utterance is never split from it (section 6.1).
        if self.pending_close:
            open_start = self.segmenter.open_start_s
            while self.pending_close:
                k = self.pending_close[0]
                boundary = (k + 1) * self.block_s
                if open_start is not None and open_start < boundary:
                    break
                self.pending_close.pop(0)
                self.asr_q.put(("close", k, boundary))

    # -- asr thread ------------------------------------------------------

    def _asr_worker(self) -> None:
        max_queue = int(self.cfg["asr"].get("max_queue", 4))
        carry = bool(self.cfg["asr"].get("carry_context", True))
        while True:
            item = self.asr_q.get()
            if item is None:
                self.refine_q.put(None)
                return
            if item[0] == "close":
                self._close_block(item[1], item[2])
                continue
            utt: vad.Utterance = item[1]
            block = int(utt.start_s // self.block_s)
            self.utt_spans.setdefault(block, []).append((utt.start_s, utt.end_s))
            depth = self.asr_q.qsize()
            if depth > max_queue and not self.behind:
                self.behind = True
                self.task.log("asr: queue depth %d" % depth)
                bus.emit({"type": "status", **app.status()})
            elif depth <= 1 and self.behind:
                self.behind = False
                bus.emit({"type": "status", **app.status()})
            prompt = ""
            if carry:
                with self.lock:
                    prompt = " ".join(l["text"] for l in self.lines[-6:])[-200:]
            gpu = engines.gpu
            if gpu is None:
                continue
            t0 = time.time()
            try:
                segs = gpu.infer(utt.audio, prompt)
            except Exception as exc:  # noqa: BLE001 - one bad utterance, carry on
                self.task.log("asr: utterance at %s failed (%s)"
                              % (history.hms(utt.start_s), type(exc).__name__))
                continue
            segs, dropped = asr.clean_segments(segs, utt.duration_s, self.cfg)
            if dropped:
                self.task.log("asr: dropped %d segment(s) at %s" % (dropped, history.hms(utt.start_s)))
            text = " ".join(s["text"] for s in segs).strip()
            if not text:
                continue
            with self.lock:
                line = {"id": self._next_id, "t": round(utt.start_s, 2), "text": text,
                        "refined": False, "block": block}
                self._next_id += 1
                self.lines.append(line)
                self.meeting.append_line(utt.start_s, text)
            self.meeting.meta["duration_s"] = utt.end_s
            bus.emit({"type": "utterance", "line": line,
                      "latency": round(time.time() - utt.speech_end_wall, 2)})
            if (time.time() - t0) > 3:
                self.task.log("asr: slow utterance %.1fs" % (time.time() - t0))

    def _close_block(self, k: int, end_s: float) -> None:
        start_s = k * self.block_s
        with self.lock:
            words = sum(len(l["text"].split()) for l in self.lines if l["block"] == k)
        spans = self.utt_spans.pop(k, [])
        if not spans and words == 0 and end_s - start_s < self.block_s:
            return                         # a silent partial block at Stop: nothing to show
        block = {"index": k, "start_s": start_s, "end_s": max(end_s, start_s),
                 "status": "pending", "notes": ""}
        self.blocks[k] = block
        self.meeting.update_meta(duration_s=max(end_s, self.meeting.meta.get("duration_s", 0)))
        if not spans or words == 0:
            self.ring.trim_before(end_s)
            self.notes_q.put(k)
            return
        want_refine = engines.cpu is not None and self.cfg["asr"].get("refine", True)
        if want_refine and self.refine_busy:
            # Never queue a refine behind a slow one: degrade to rough text,
            # not to growing delay (section 6.1).
            self.task.log("refine: still busy at block %d; keeping its live lines" % k)
            want_refine = False
        if not want_refine:
            self.ring.trim_before(end_s)
            self._finalise_lines(k)
            self.notes_q.put(k)
            return
        block["status"] = "refining"
        bus.emit({"type": "block_pending", "block": dict(block)})
        a0 = max(0.0, spans[0][0])
        a1 = spans[-1][1]
        audio = self.ring.slice(a0, a1)
        self.refine_busy = True
        self.refine_q.put((k, a0, audio, end_s))

    def _finalise_lines(self, k: int) -> None:
        """Lines that will never be refined stop being shown as provisional."""
        with self.lock:
            changed = [l for l in self.lines if l["block"] == k and not l["refined"]]
            for l in changed:
                l["refined"] = True
        if changed:
            bus.emit({"type": "block_lines", "block": k,
                      "lines": [dict(l) for l in self.lines if l["block"] == k]})

    # -- refine thread ---------------------------------------------------

    def _refine_worker(self) -> None:
        while True:
            item = self.refine_q.get()
            if item is None:
                self.notes_q.put(None)
                return
            k, a0, audio, end_s = item
            cpu = engines.cpu
            ok = False
            t0 = time.time()
            if cpu is not None and len(audio):
                try:
                    segs = cpu.infer(audio, "", timeout=900)
                    segs, dropped = asr.clean_segments(segs, len(audio) / SR, self.cfg)
                    self._replace_block(k, a0, segs)
                    ok = True
                    self.task.log("refine: block %d, %.0fs of audio in %.0fs, %d segments"
                                  % (k, len(audio) / SR, time.time() - t0, len(segs)))
                except Exception as exc:  # noqa: BLE001 - keep the live lines
                    self.task.log("refine: block %d failed (%s); keeping live lines"
                                  % (k, type(exc).__name__))
            del audio
            self.ring.trim_before(end_s)
            if not ok:
                self._finalise_lines(k)
            self.refine_busy = False
            self.notes_q.put(k)

    def _replace_block(self, k: int, a0: float, segs: list[dict]) -> None:
        with self.lock:
            old = [l for l in self.lines if l["block"] == k]
            if not segs:
                # Nothing recognised on the second pass: the live lines stand.
                for l in old:
                    l["refined"] = True
                new = old
            else:
                new = []
                for s in segs:
                    new.append({"id": self._next_id, "t": round(a0 + s["start"], 2),
                                "text": s["text"], "refined": True, "block": k})
                    self._next_id += 1
                kept = [l for l in self.lines if l["block"] != k]
                # Lines of later blocks that arrived meanwhile stay after these.
                before = [l for l in kept if l["block"] < k]
                after = [l for l in kept if l["block"] > k]
                self.lines = before + new + after
                self.meeting.rewrite_transcript([(l["t"], l["text"]) for l in self.lines])
            replaced = [l["id"] for l in old]
        bus.emit({"type": "block_lines", "block": k, "replaced": replaced,
                  "lines": [dict(l) for l in new]})

    # -- notes thread ----------------------------------------------------

    def _notes_worker(self) -> None:
        template = llm.load_prompt("block")
        while True:
            k = self.notes_q.get()
            if k is None:
                return
            block = self.blocks.get(k)
            if block is None:
                continue
            with self.lock:
                lines = [l for l in self.lines if l["block"] == k]
            words = sum(len(l["text"].split()) for l in lines)
            if self.skip_notes and block["status"] in ("pending", "refining"):
                # Generate Full Summary abandons the pending block (section 7.1).
                block.update(status="skipped", notes="")
                bus.emit({"type": "block", "block": dict(block)})
                continue
            if words < self.min_words:
                block.update(status="empty", notes="_Little was said._")
            elif engines.llama is None:
                block.update(status="failed",
                             notes="_Notes for this block couldn't be written._")
            else:
                block["status"] = "writing"
                bus.emit({"type": "block_pending", "block": dict(block)})
                prompt = llm.fill(
                    template,
                    transcript="\n".join("(%s) %s" % (self.meeting.stamp(l["t"]), l["text"])
                                         for l in lines),
                    previous_notes=self.prev_notes or "(none -- this is the first block)",
                    time_start=self.meeting.stamp(block["start_s"], False),
                    time_end=self.meeting.stamp(block["end_s"], False))
                t0 = time.time()
                try:
                    text = engines.llama.chat(prompt, int(self.cfg["notes"].get("max_tokens", 600)),
                                              thinking=False, attempts=2)
                    block.update(status="done", notes=text)
                    self.prev_notes = text
                    self.task.log("notes: block %d in %.1fs" % (k, time.time() - t0))
                except Exception as exc:  # noqa: BLE001 - never stops transcription
                    jobs.log_exception(exc)
                    block.update(status="failed",
                                 notes="_Notes for this block couldn't be written._")
            if block["status"] in ("done", "empty"):
                self.meeting.append_block(block["start_s"], block["end_s"], block["notes"])
            bus.emit({"type": "block", "block": dict(block)})

    # -- views -----------------------------------------------------------

    def snapshot(self) -> dict:
        with self.lock:
            lines = [dict(l) for l in self.lines]
        return {
            "id": self.meeting.id,
            "name": self.meeting.meta.get("name", self.meeting.id),
            "state": self.state,
            "started": self.started,
            "elapsed": round((self.ended or time.time()) - self.started, 1),
            "block_s": self.block_s,
            "next_notes_at": (self.cur_block + 1) * self.block_s,
            "lines": lines,
            "blocks": [dict(self.blocks[k]) for k in sorted(self.blocks)],
            "has_summary": bool(self.meeting.meta.get("has_summary")),
            "summary": self.meeting.read_summary() if self.meeting.meta.get("has_summary") else "",
        }

    @property
    def words(self) -> int:
        with self.lock:
            return sum(len(l["text"].split()) for l in self.lines)


_last_level = [0.0]


def _emit_level(db: float) -> None:
    _last_level[0] = db
    bus.emit({"type": "level", "db": round(db, 1)})


# ---------------------------------------------------------------------------
# the level meter before Start (section 11)
# ---------------------------------------------------------------------------

class Monitor:
    """Reads the selected mic for the level meter only, while nothing is live.

    Stopped as soon as a meeting starts, and when no page is listening, so the
    mic is not held open for nobody.
    """

    def __init__(self):
        self.capture: Optional[capture.Capture] = None
        self.device = ""
        self.task = Task("monitor")
        self._lock = threading.Lock()

    def start(self, device: str) -> None:
        with self._lock:
            if self.capture is not None and self.device == device:
                return
            self._stop_locked()
            if not device:
                return
            self.device = device
            self.capture = capture.Capture(self.task, device, on_level=_emit_level,
                                           on_end=self._ended)
            try:
                self.capture.start()
            except OSError as exc:
                jobs.log_exception(exc)
                self.capture = None

    def _ended(self) -> None:
        with self._lock:
            self.capture = None

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()

    def _stop_locked(self) -> None:
        if self.capture is not None:
            cap, self.capture = self.capture, None
            cap.stop()
        self.device = ""


monitor = Monitor()


# ---------------------------------------------------------------------------
# the app: one live meeting, the resident models, the summary job
# ---------------------------------------------------------------------------

class App:
    def __init__(self):
        self.state = "idle"              # idle | loading | live | stopping | summarizing
        self.message = ""
        self.session: Optional[Session] = None
        self.device = ""
        self.device_how = ""
        self.choice = ""                 # Model dropdown: model key or "external"
        self._lock = threading.Lock()
        self.summary = None              # summary.SummaryJob while one runs

    def status(self) -> dict:
        s = self.session
        warning = ""
        if s is not None and s.state == "live":
            warning = FALLING_BEHIND if s.behind else s.warning
        return {"state": self.state, "message": self.message, "warning": warning,
                "device": self.device, "device_how": self.device_how,
                "live_id": s.meeting.id if s is not None and s.state == "live" else "",
                "engines_loaded": engines.loaded}

    def _set(self, state: str, message: str = "") -> None:
        self.state = state
        self.message = message
        bus.emit({"type": "status", **self.status()})

    def resolve_device(self, cfg: dict) -> tuple[str, str, list[str]]:
        devices = capture.list_devices()
        name, how = capture.match_device(str(cfg["capture"].get("device") or ""), devices)
        if how == "renamed":
            try:
                config.save_device(name)       # same mic on another USB port
            except RuntimeError:
                pass
        self.device, self.device_how = name, how
        return name, how, devices

    def default_choice(self, cfg: dict) -> str:
        external, _port = config.external_llm(cfg)
        if external:
            return llm.EXTERNAL
        if self.choice and self.choice != llm.EXTERNAL:
            return self.choice
        return hardware.choose_live_key()

    def start(self) -> None:
        """Load the models, open a meeting folder and start listening."""
        with self._lock:
            if self.state not in ("idle",):
                return
            self._set("loading", "Loading models (up to a minute)…")
        try:
            cfg = config.load_config()
            replay = bool(cfg["capture"].get("replay_file"))
            device, how, _devices = self.resolve_device(cfg)
            if not device and not replay:
                self._set("idle", NO_MIC)
                return
            monitor.stop()
            choice = self.choice or self.default_choice(cfg)
            engines.ensure(cfg, choice, lambda m: self._set("loading", m))
            meeting = history.Meeting.create(time.time())
            session = Session(cfg, device, meeting)
            self.session = session
            session.start()
            self._set("live", "")
            bus.emit({"type": "session", "session": session.snapshot()})
            bus.emit({"type": "history"})
        except JobError as exc:
            jobs.log_exception(exc)
            if exc.detail:
                jobs.log(exc.detail)
            self._set("idle", exc.message)
        except Exception as exc:  # noqa: BLE001
            jobs.log_exception(exc)
            self._set("idle", "Couldn't start listening.")

    def stop(self, for_summary: bool = False) -> Optional[Session]:
        with self._lock:
            s = self.session
            if s is None or s.state != "live" or self.state != "live":
                return None
            self._set("stopping", "Finishing the last few minutes…")
        try:
            s.stop(for_summary=for_summary)
        except Exception as exc:  # noqa: BLE001
            jobs.log_exception(exc)
        # A quick mic check leaves nothing behind (section 11.1).
        if not for_summary and s.meeting.word_count() < s.min_words:
            history.delete(s.meeting.id)
            jobs.log("session: under %d words, not kept" % s.min_words)
        cfg = config.load_config()
        if not for_summary:
            engines.schedule_idle_stop(float(cfg["llm"].get("idle_timeout_s", 300)))
            self._set("idle", "")
        bus.emit({"type": "session", "session": s.snapshot()})
        bus.emit({"type": "history"})
        return s

    def shutdown(self) -> None:
        s = self.session
        if s is not None and s.state == "live":
            try:
                s.capture.stop()
            except Exception:  # noqa: BLE001
                pass
        monitor.stop()
        engines.stop_all()


app = App()
