"""whisper-server lifecycle, requests and the hallucination filter (sections 6, 6.1).

Two instances, never one doing both jobs:

- **GPU**, for live utterances. Turbo on a few seconds of audio takes ~0.45 s.
- **CPU** (`-ng`, every card hidden), for the refine pass: a whole 5-minute
  block in one request, with whisper's own VAD -- batch quality, 106-122 s.
  On the GPU instance that would stall live lines, since a server handles one
  request at a time; a second GPU instance needs VRAM that is not there.

Audio is posted as an in-memory WAV. `--convert` is never passed: it shells
out to ffmpeg and writes temp files, and audio must never touch disk.
"""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import threading
import time
import uuid
import wave
from http.client import HTTPConnection
from typing import Optional

import numpy as np

from . import config
from .jobs import JobError, Task, kill_tree

CREATE_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

SR = 16000
ASR_FAILED = "Speech recognition could not be started."

# Context for the refine pass. Whole-block transcription with whisper's own VAD
# comes back all lower case with no punctuation -- measured on a 5-minute
# block: 0 sentence marks, 6% capitalised words. A punctuated opening line as
# the prompt is enough to restore it: 104 sentence marks, 25% capitalised, the
# same words (BUILD_NOTES section 4). It conditions style only; it is never output.
REFINE_PRIMER = ("Good morning, everyone. Thanks for joining. "
                 "Let's get started with the first item on the agenda.")


def wav_bytes(pcm: np.ndarray) -> bytes:
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.astype(np.int16).tobytes())
    return b.getvalue()


def refine_threads(cfg: dict) -> int:
    """`asr.refine_threads: "auto"` = half the logical cores, at least 4."""
    value = cfg["asr"].get("refine_threads", "auto")
    if str(value) == "auto":
        return max(4, (os.cpu_count() or 8) // 2)
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 4


class WhisperServer:
    def __init__(self, task: Task, cfg: dict, gpu: bool):
        self.task = task
        self.cfg = cfg
        self.gpu = gpu
        asr = cfg["asr"]
        self.port = int(asr.get("port", 8081) if gpu else asr.get("refine_port", 8082))
        self.threads = int(asr.get("threads", 4)) if gpu else refine_threads(cfg)
        self.proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._conn: Optional[HTTPConnection] = None
        self._log_handle = None
        self.name = "whisper-gpu" if gpu else "whisper-cpu"

    def command(self) -> list[str]:
        asr = self.cfg["asr"]
        cmd = [str(config.WHISPER_SERVER),
               "-m", str(config.resolve(asr.get("model", "models/ggml-large-v3-turbo.bin"))),
               "-l", str(asr.get("language", "en")),
               "-t", str(self.threads),
               "--host", "127.0.0.1", "--port", str(self.port)]
        if not self.gpu:
            cmd += ["-ng"]
            vad_model = config.resolve(asr.get("refine_vad_model",
                                               "models/ggml-silero-v5.1.2.bin"))
            if vad_model.exists():
                # Batch quality was measured with whisper's own VAD (BUILD_NOTES 4).
                cmd += ["--vad", "--vad-model", str(vad_model)]
        return cmd

    def start(self) -> None:
        # Holds model loading and per-request lines only, never text (BUILD_NOTES 3).
        self._log_handle = open(config.TEMP / ("%s.log" % self.name), "w",
                                encoding="utf-8", errors="replace")
        try:
            self.proc = subprocess.Popen(
                self.command(), stdout=self._log_handle, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, env=config.child_env(gpu=self.gpu),
                cwd=str(config.WHISPER_SERVER.parent),
                creationflags=CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW)
        except OSError as exc:
            raise JobError(ASR_FAILED, "could not spawn %s: %s" % (self.name, exc)) from exc
        self.task.register_proc(self.proc)
        started = time.time()
        while time.time() - started < 120:
            self.task.check_cancelled()
            if self.proc.poll() is not None:
                raise JobError(ASR_FAILED, "%s exited during load" % self.name)
            if self._healthy():
                self.task.log("%s: ready after %.1fs (%d threads)"
                              % (self.name, time.time() - started, self.threads))
                return
            time.sleep(0.3)
        raise JobError(ASR_FAILED, "%s did not become ready" % self.name)

    def warm_up(self) -> float:
        """One inference on noise before going live (section 8).

        The first inference on a card the build has no native kernels for --
        compute 12.0 against a build whose architectures stop at 860 -- JIT-
        compiles for about a minute, and the driver's cache of the result does
        not reliably survive (BUILD_NOTES sections 2-3).
        """
        noise = (np.random.default_rng(0).standard_normal(SR * 2) * 30).astype(np.int16)
        t0 = time.time()
        self.infer(noise, timeout=300)
        took = time.time() - t0
        self.task.log("%s: warm-up %.1fs" % (self.name, took))
        return took

    def _healthy(self) -> bool:
        try:
            conn = HTTPConnection("127.0.0.1", self.port, timeout=2)
            conn.request("GET", "/")
            resp = conn.getresponse()
            resp.read()
            conn.close()
            return resp.status < 500
        except OSError:
            return False

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def infer(self, pcm: np.ndarray, prompt: str = "", timeout: float = 120) -> list[dict]:
        """Segments [{start, end, text, no_speech_prob, avg_logprob}], times in seconds."""
        boundary = uuid.uuid4().hex
        parts = []

        def field(name: str, value: str) -> None:
            parts.append(('--%s\r\nContent-Disposition: form-data; name="%s"\r\n\r\n%s\r\n'
                          % (boundary, name, value)).encode("utf-8"))

        field("response_format", "verbose_json")
        field("temperature", "0")
        field("language", str(self.cfg["asr"].get("language", "en")))
        if prompt:
            field("prompt", prompt)
        parts.append(('--%s\r\nContent-Disposition: form-data; name="file"; filename="a.wav"'
                      '\r\nContent-Type: audio/wav\r\n\r\n' % boundary).encode("ascii"))
        parts.append(wav_bytes(pcm))
        parts.append(("\r\n--%s--\r\n" % boundary).encode("ascii"))
        body = b"".join(parts)

        with self._lock:                  # one request in flight per server
            conn = HTTPConnection("127.0.0.1", self.port, timeout=timeout)
            self._conn = conn
            try:
                conn.request("POST", "/inference", body, {
                    "Content-Type": "multipart/form-data; boundary=%s" % boundary})
                resp = conn.getresponse()
                raw = resp.read()
                if resp.status != 200:
                    raise RuntimeError("%s HTTP %d" % (self.name, resp.status))
                data = json.loads(raw)
            finally:
                self._conn = None
                try:
                    conn.close()
                except Exception:  # noqa: BLE001
                    pass
        out = []
        for s in data.get("segments") or []:
            out.append({
                "start": float(s.get("start", 0.0)),
                "end": float(s.get("end", 0.0)),
                "text": str(s.get("text", "")).strip(),
                "no_speech_prob": float(s.get("no_speech_prob", 0.0) or 0.0),
                "avg_logprob": float(s.get("avg_logprob", 0.0) or 0.0),
            })
        if not out and str(data.get("text", "")).strip():
            out.append({"start": 0.0, "end": len(pcm) / SR, "text": str(data["text"]).strip(),
                        "no_speech_prob": 0.0, "avg_logprob": 0.0})
        return out

    def abort(self) -> None:
        conn = self._conn
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass

    def stop(self) -> None:
        self.abort()
        proc, self.proc = self.proc, None
        if proc is None:
            return
        self.task.unregister_proc(proc)
        kill_tree(proc)
        try:
            proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            pass
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            except Exception:  # noqa: BLE001
                pass
        self.task.log("%s: stopped" % self.name)


# ---------------------------------------------------------------------------
# hallucination filter (section 6)
# ---------------------------------------------------------------------------

# Three or more consecutive repeats of one phrase collapse to one.
_REPEAT_RE = re.compile(r"(\b.{3,}?[.!?,]?)(?:\s+\1){2,}", re.IGNORECASE)


def clean_segments(segments: list[dict], duration_s: float, cfg: dict) -> tuple[list[dict], int]:
    """(kept segments, how many were dropped). Counts go to the log, text never does."""
    drops = {p.strip().lower() for p in cfg["asr"].get("drop_phrases") or []}
    kept, dropped = [], 0
    for s in segments:
        text = _REPEAT_RE.sub(r"\1", s["text"]).strip()
        if not text:
            continue
        if s["no_speech_prob"] > 0.6 and s["avg_logprob"] < -1.0:
            dropped += 1
            continue
        kept.append({**s, "text": text})
    # A short utterance whose whole text is a known filler is a hallucination.
    whole = " ".join(s["text"] for s in kept).strip().lower()
    if kept and duration_s < 2.0 and whole in drops:
        dropped += len(kept)
        kept = []
    return kept, dropped
