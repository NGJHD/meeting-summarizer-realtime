"""Silero VAD v5 under onnxruntime, and the utterance segmenter (CLAUDE.md section 5).

Runs in-process: about 0.46 ms of CPU per 32 ms frame on one thread
(BUILD_NOTES section 1). Plain onnxruntime rather than sherpa-onnx's wrapper of
the same model, because sherpa holds the GIL through its native calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

SR = 16000
FRAME = 512                     # 32 ms
CONTEXT = 64                    # v5 prepends the previous 64 samples to each frame


class Silero:
    def __init__(self, path: Path):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self.reset()

    def reset(self) -> None:
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.ctx = np.zeros((1, CONTEXT), dtype=np.float32)
        self.sr = np.array(SR, dtype=np.int64)

    def prob(self, pcm: np.ndarray) -> float:
        frame = (pcm.astype(np.float32) / 32768.0).reshape(1, -1)
        x = np.concatenate([self.ctx, frame], axis=1)
        out, self.state = self.sess.run(None, {"input": x, "state": self.state, "sr": self.sr})
        self.ctx = x[:, -CONTEXT:]
        return float(out[0][0])


@dataclass
class Utterance:
    start_s: float              # session time, from the sample count (never the clock)
    end_s: float
    audio: np.ndarray           # int16 PCM, held in memory only
    speech_end_wall: float      # when the last speech frame was heard, for latency
    forced: bool = False

    @property
    def duration_s(self) -> float:
        return len(self.audio) / SR


class Segmenter:
    """32 ms frames in, utterances out.

    Counted in whole frames so the audio and the speech probabilities stay
    aligned. An utterance ends after `min_silence_ms` of silence, keeps
    `pad_ms` either side, and is cut at the quietest frame of the last 3 s once
    it reaches `max_utterance_s` -- never at exactly 15.000 s, which would send
    half a word to each side.
    """

    def __init__(self, threshold=0.5, min_silence_ms=600, pad_ms=200,
                 min_speech_ms=250, max_utterance_s=15.0):
        self.th = float(threshold)
        self.silence_frames = max(1, round(min_silence_ms / 32))
        self.pad_frames = max(0, round(pad_ms / 32))
        self.min_speech_frames = max(1, round(min_speech_ms / 32))
        self.max_frames = int(float(max_utterance_s) * SR / FRAME)
        self.history: list[np.ndarray] = []
        self.buf: list[np.ndarray] = []
        self.probs: list[float] = []
        self.in_speech = False
        self.quiet = 0
        self.start_frame = 0
        self.last_speech_wall = 0.0

    @classmethod
    def from_config(cls, vad: dict) -> "Segmenter":
        return cls(vad.get("threshold", 0.5), vad.get("min_silence_ms", 600),
                   vad.get("pad_ms", 200), vad.get("min_speech_ms", 250),
                   vad.get("max_utterance_s", 15))

    @property
    def open_start_s(self) -> float | None:
        """Start of the utterance in progress, or None between utterances."""
        return self.start_frame * FRAME / SR if self.in_speech else None

    def push(self, pcm: np.ndarray, p: float, frame_idx: int, wall: float):
        if not self.in_speech:
            if p < self.th:
                if self.pad_frames:
                    self.history = (self.history + [pcm])[-self.pad_frames:]
                return None
            self.in_speech = True
            self.buf = self.history + [pcm]
            self.probs = [0.0] * len(self.history) + [p]
            self.start_frame = frame_idx - len(self.history)
            self.history = []
            self.quiet = 0
            self.last_speech_wall = wall
            return None
        self.buf.append(pcm)
        self.probs.append(p)
        if p >= self.th:
            self.quiet = 0
            self.last_speech_wall = wall
        else:
            self.quiet += 1
        if self.quiet >= self.silence_frames:
            return self._close(min(len(self.buf), len(self.buf) - self.quiet + self.pad_frames))
        if len(self.buf) >= self.max_frames:
            tail = self.probs[-int(3 * SR / FRAME):]
            cut = len(self.probs) - len(tail) + int(np.argmin(tail)) + 1
            return self._close(cut, forced=True)
        return None

    def flush(self):
        """End whatever is open -- Stop pressed mid-sentence."""
        if not self.in_speech or not self.buf:
            return None
        return self._close(len(self.buf))

    def _close(self, keep: int, forced: bool = False):
        audio = np.concatenate(self.buf[:keep])
        speech = sum(1 for q in self.probs[:keep] if q >= self.th)
        start = self.start_frame * FRAME / SR
        utt = None
        if speech >= self.min_speech_frames:
            utt = Utterance(start_s=start, end_s=start + len(audio) / SR, audio=audio,
                            speech_end_wall=self.last_speech_wall, forced=forced)
        rest, rest_p = self.buf[keep:], self.probs[keep:]
        if forced and rest:
            self.start_frame += keep          # the remainder carries straight on
            self.buf, self.probs = rest, rest_p
            self.quiet = 0
        else:
            self.in_speech = False
            self.buf, self.probs = [], []
        return utt
