"""Build step 2 (CLAUDE.md §15): can whisper and the LLM share the machine live?

Replays a recording at real time through the same path the app will use --
ffmpeg pipe -> Silero VAD -> whisper-server -> transcript, with a block summary
fired at llama-server every --block-s seconds of audio -- and measures:

  * utterance latency: last speech frame heard -> text back, split by whether
    an LLM call was in flight while it was being recognised;
  * whisper inference time per utterance;
  * block-call wall time and tokens/second;
  * peak VRAM per GPU, sampled from nvidia-smi once a second.

Transcript and notes go to --out (a scratch folder) and are never printed:
they are meeting content (CLAUDE.md §10). Only numbers reach stdout.

    runtime\\python.exe -I tools\\spike.py --audio "<file>" --start 600 --minutes 12 ^
        --whisper-dev 1 --llm-dev CUDA1 --model Qwen3.8-27B-UD-IQ3_XXS.gguf --out <dir>
"""

from __future__ import annotations

import argparse
import io
import os
import json
import queue
import statistics
import subprocess
import sys
import threading
import time
import uuid
import wave
from http.client import HTTPConnection
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "bin"
MODELS = ROOT / "models"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

SR = 16000
FRAME = 512                     # 32 ms, Silero v5's frame at 16 kHz
CONTEXT = 64                    # v5 prepends the previous 64 samples


# ---------------------------------------------------------------- VAD

class Silero:
    def __init__(self, path: Path):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.ctx = np.zeros((1, CONTEXT), dtype=np.float32)
        self.sr = np.array(SR, dtype=np.int64)

    def prob(self, frame: np.ndarray) -> float:
        x = np.concatenate([self.ctx, frame.reshape(1, -1)], axis=1)
        out, self.state = self.sess.run(None, {"input": x, "state": self.state, "sr": self.sr})
        self.ctx = x[:, -CONTEXT:]
        return float(out[0][0])


class Segmenter:
    """Frames in, utterances out. Mirrors CLAUDE.md §5's defaults."""

    def __init__(self, threshold=0.5, min_silence_ms=600, pad_ms=200,
                 min_speech_ms=250, max_utterance_s=15.0):
        # Everything is counted in whole 32 ms frames so buf and probs align.
        self.th = threshold
        self.silence_frames = max(1, round(min_silence_ms / 32))
        self.pad_frames = max(0, round(pad_ms / 32))
        self.min_speech_frames = max(1, round(min_speech_ms / 32))
        self.max_frames = int(max_utterance_s * SR / FRAME)
        self.history: list[np.ndarray] = []     # rolling pre-roll, in frames
        self.buf: list[np.ndarray] = []
        self.probs: list[float] = []
        self.in_speech = False
        self.quiet = 0
        self.start_frame = 0
        self.last_speech_wall = 0.0

    def push(self, pcm: np.ndarray, p: float, frame_idx: int, wall: float):
        """frame_idx is this frame's index from the start. Returns an utterance or None."""
        if not self.in_speech:
            if p < self.th:
                self.history = (self.history + [pcm])[-self.pad_frames:] if self.pad_frames else []
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
            keep = min(len(self.buf), len(self.buf) - self.quiet + self.pad_frames)
            return self._close(keep)
        if len(self.buf) >= self.max_frames:
            # cut at the quietest frame in the last 3 s, not at exactly max_len
            n_tail = int(3 * SR / FRAME)
            tail = self.probs[-n_tail:]
            cut = len(self.probs) - len(tail) + int(np.argmin(tail)) + 1
            return self._close(cut, forced=True)
        return None

    def _close(self, keep: int, forced: bool = False):
        audio = np.concatenate(self.buf[:keep])
        speech = sum(1 for q in self.probs[:keep] if q >= self.th)
        utt = None
        if speech >= self.min_speech_frames:
            utt = {"start_s": self.start_frame * FRAME / SR, "dur_s": len(audio) / SR,
                   "audio": audio, "speech_end_wall": self.last_speech_wall,
                   "forced": forced}
        rest, rest_p = self.buf[keep:], self.probs[keep:]
        if forced and rest:
            # carry the remainder straight into the next utterance
            self.start_frame += keep
            self.buf, self.probs = rest, rest_p
            self.quiet = 0
        else:
            self.in_speech = False
            self.buf, self.probs = [], []
        return utt


# ---------------------------------------------------------------- servers

def wait_http(port: int, path: str, timeout: float) -> float:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            c = HTTPConnection("127.0.0.1", port, timeout=2)
            c.request("GET", path)
            r = c.getresponse()
            r.read()
            if r.status < 500 and r.status != 503:
                return time.time() - t0
        except OSError:
            pass
        time.sleep(0.5)
    raise RuntimeError("server on port %d did not come up" % port)


def wav_bytes(pcm: np.ndarray) -> bytes:
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return b.getvalue()


def whisper_infer(port: int, pcm: np.ndarray, prompt: str) -> dict:
    boundary = uuid.uuid4().hex
    parts = []

    def field(name, value):
        parts.append(("--%s\r\nContent-Disposition: form-data; name=\"%s\"\r\n\r\n%s\r\n"
                      % (boundary, name, value)).encode())

    field("response_format", "verbose_json")
    field("temperature", "0")
    field("language", "en")
    if prompt:
        field("prompt", prompt)
    parts.append(("--%s\r\nContent-Disposition: form-data; name=\"file\"; filename=\"u.wav\"\r\n"
                  "Content-Type: audio/wav\r\n\r\n" % boundary).encode())
    parts.append(wav_bytes(pcm))
    parts.append(("\r\n--%s--\r\n" % boundary).encode())
    body = b"".join(parts)
    c = HTTPConnection("127.0.0.1", port, timeout=120)
    c.request("POST", "/inference", body,
              {"Content-Type": "multipart/form-data; boundary=%s" % boundary})
    r = c.getresponse()
    data = r.read()
    if r.status != 200:
        raise RuntimeError("whisper %d: %s" % (r.status, data[:200]))
    return json.loads(data)


def llm_chat(port: int, prompt: str, max_tokens: int) -> dict:
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
        "presence_penalty": 1.5, "repeat_penalty": 1.0,
        "max_tokens": max_tokens, "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    c = HTTPConnection("127.0.0.1", port, timeout=600)
    c.request("POST", "/v1/chat/completions", json.dumps(payload),
              {"Content-Type": "application/json"})
    r = c.getresponse()
    data = json.loads(r.read())
    if r.status != 200:
        raise RuntimeError("llm %d: %s" % (r.status, str(data)[:200]))
    return data


# ---------------------------------------------------------------- VRAM

class VramSampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.peak: dict[str, int] = {}
        self.base: dict[str, int] = {}
        self.stop = threading.Event()

    def sample(self) -> dict[str, int]:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, creationflags=NO_WINDOW).stdout
        res = {}
        for line in out.strip().splitlines():
            name, used = line.rsplit(",", 1)
            res[name.strip()] = int(used)
        return res

    def run(self):
        while not self.stop.is_set():
            for k, v in self.sample().items():
                self.peak[k] = max(self.peak.get(k, 0), v)
            time.sleep(1)


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--minutes", type=float, default=12)
    ap.add_argument("--block-s", type=float, default=120)
    ap.add_argument("--whisper-dev", default="1", help="whisper-server -dev (CUDA index)")
    ap.add_argument("--llm-dev", default="CUDA1", help="llama-server --device")
    ap.add_argument("--model", default="Qwen3.8-27B-UD-IQ3_XXS.gguf")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--mtp", action="store_true")
    ap.add_argument("--ot", default="", help="--override-tensor regex, '' for none")
    ap.add_argument("--carry", action="store_true", help="pass recent text as prompt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="run")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    vram = VramSampler()
    vram.base = vram.sample()

    wport, lport = 8081, 8080
    # cudart/cublas live in bin\, which run.bat puts on PATH.
    env = dict(os.environ, PATH=str(BIN) + os.pathsep + os.environ.get("PATH", ""))
    wlog = open(out / ("%s-whisper.log" % a.label), "w")
    llog = open(out / ("%s-llama.log" % a.label), "w")
    whisper = subprocess.Popen(
        [str(BIN / "whisper-cuda" / "whisper-server.exe"),
         "-m", str(MODELS / "ggml-large-v3-turbo.bin"), "-l", "en", "-t", "4",
         "-dev", a.whisper_dev, "--host", "127.0.0.1", "--port", str(wport)],
        cwd=str(BIN / "whisper-cuda"), stdout=wlog, stderr=subprocess.STDOUT,
        env=env, creationflags=GROUP | NO_WINDOW)
    lcmd = [str(BIN / "llama-cuda" / "llama-server.exe"),
            "-m", str(MODELS / a.model), "--ctx-size", str(a.ctx), "--flash-attn", "on",
            "--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "--jinja",
            "--batch-size", "512", "--ubatch-size", "512", "--parallel", "1",
            "--host", "127.0.0.1", "--port", str(lport), "--no-webui",
            "--n-gpu-layers", "99", "--device", a.llm_dev]
    if a.ot:
        lcmd += ["--override-tensor", a.ot]
    if a.mtp:
        lcmd += ["--spec-type", "draft-mtp"]
    llama = subprocess.Popen(lcmd, cwd=str(BIN / "llama-cuda"), stdout=llog,
                             stderr=subprocess.STDOUT, env=env, creationflags=GROUP | NO_WINDOW)
    ff = None
    try:
        t_load = time.time()
        wl = wait_http(wport, "/", 120)
        ll = wait_http(lport, "/health", 300)
        print("[%s] loaded: whisper %.1fs, llama %.1fs" % (a.label, wl, ll), flush=True)
        # First inference on a card the build has no native kernels for
        # (compute 12.0 vs a CUDA 12.4 build) JIT-compiles PTX. Time it: the
        # app's "Loading models" has to cover it.
        noise = (np.random.default_rng(0).standard_normal(SR * 2) * 30).astype(np.int16)
        warm = []
        for _ in range(3):
            t0 = time.time()
            whisper_infer(wport, noise, "")
            warm.append(round(time.time() - t0, 2))
        print("[%s] whisper warm-up calls: %s s" % (a.label, warm), flush=True)
        vram.start()
        time.sleep(2)
        loaded = vram.sample()

        template = (ROOT / "prompts" / "block.txt").read_text(encoding="utf-8")
        vad = Silero(MODELS / "silero_vad.onnx")
        seg = Segmenter()
        utt_q: "queue.Queue" = queue.Queue()
        block_q: "queue.Queue" = queue.Queue()
        llm_busy = threading.Event()
        results, blocks, transcript = [], [], []
        lock = threading.Lock()

        def asr_worker():
            while True:
                u = utt_q.get()
                if u is None:
                    return
                qdepth = utt_q.qsize()
                busy_before = llm_busy.is_set()
                with lock:
                    prompt = " ".join(t for _, t in transcript)[-200:] if a.carry else ""
                t0 = time.time()
                try:
                    res = whisper_infer(wport, u["audio"], prompt)
                except Exception as e:      # noqa: BLE001 -- a spike logs and moves on
                    print("[%s] whisper error: %s" % (a.label, e), flush=True)
                    continue
                t1 = time.time()
                text = res.get("text", "").strip()
                segs = res.get("segments", [])
                nsp = max([s.get("no_speech_prob", 0) for s in segs] or [0])
                with lock:
                    transcript.append((u["start_s"], text))
                results.append({
                    "start_s": u["start_s"], "dur_s": u["dur_s"], "forced": u["forced"],
                    "infer_s": t1 - t0, "latency_s": t1 - u["speech_end_wall"],
                    "t0": t0, "t1": t1,
                    "llm_busy": busy_before or llm_busy.is_set(), "qdepth": qdepth,
                    "words": len(text.split()), "no_speech_prob": nsp,
                })

        def notes_worker():
            prev = "(none)"
            while True:
                b = block_q.get()
                if b is None:
                    return
                lo, hi = b
                with lock:
                    lines = ["(%s) %s" % (hms(s), t) for s, t in transcript if lo <= s < hi and t]
                text = "\n".join(lines)
                prompt = (template.replace("{{PREVIOUS_NOTES}}", prev)
                          .replace("{{TIME_START}}", hms(lo)).replace("{{TIME_END}}", hms(hi))
                          .replace("{{TRANSCRIPT}}", text))
                llm_busy.set()
                t0 = time.time()
                try:
                    r = llm_chat(lport, prompt, 600)
                except Exception as e:      # noqa: BLE001
                    llm_busy.clear()
                    print("[%s] llm error: %s" % (a.label, e), flush=True)
                    continue
                dt = time.time() - t0
                llm_busy.clear()
                tim = r.get("timings", {})
                note = r["choices"][0]["message"]["content"]
                prev = note
                blocks.append({"block": [lo, hi], "wall_s": dt, "t0": t0, "t1": t0 + dt,
                               "prompt_tokens": r["usage"]["prompt_tokens"],
                               "completion_tokens": r["usage"]["completion_tokens"],
                               "prompt_tps": tim.get("prompt_per_second"),
                               "gen_tps": tim.get("predicted_per_second"),
                               "draft_accepted": tim.get("draft_n_accepted"),
                               "draft_n": tim.get("draft_n")})
                with open(out / ("%s-notes.md" % a.label), "a", encoding="utf-8") as f:
                    f.write("## %s-%s\n%s\n\n" % (hms(lo), hms(hi), note))
                print("[%s] block %s-%s: %.1fs, %d in / %d out, gen %.1f tok/s"
                      % (a.label, hms(lo), hms(hi), dt, r["usage"]["prompt_tokens"],
                         r["usage"]["completion_tokens"], tim.get("predicted_per_second", 0)),
                      flush=True)

        ta = threading.Thread(target=asr_worker, daemon=True)
        tn = threading.Thread(target=notes_worker, daemon=True)
        ta.start()
        tn.start()

        ff = subprocess.Popen(
            [str(BIN / "ffmpeg.exe"), "-hide_banner", "-loglevel", "error",
             "-ss", str(a.start), "-t", str(a.minutes * 60), "-re", "-i", a.audio,
             "-ac", "1", "-ar", str(SR), "-f", "s16le", "pipe:1"],
            stdout=subprocess.PIPE, creationflags=GROUP | NO_WINDOW)
        pos = 0
        next_block = a.block_s
        vad_time = 0.0
        t_run = time.time()
        while True:
            raw = ff.stdout.read(FRAME * 2)
            if len(raw) < FRAME * 2:
                break
            pcm = np.frombuffer(raw, dtype=np.int16)
            wall = time.time()
            v0 = time.perf_counter()
            p = vad.prob(pcm.astype(np.float32) / 32768.0)
            vad_time += time.perf_counter() - v0
            u = seg.push(pcm, p, pos // FRAME, wall)
            pos += FRAME
            if u:
                utt_q.put(u)
            if pos / SR >= next_block:
                block_q.put((next_block - a.block_s, next_block))
                next_block += a.block_s
        utt_q.put(None)
        ta.join()
        block_q.put(None)
        tn.join()
        run_s = time.time() - t_run
        vram.stop.set()

        def stats(xs):
            if not xs:
                return None
            xs = sorted(xs)
            return {"n": len(xs), "median": round(statistics.median(xs), 2),
                    "p90": round(xs[int(0.9 * (len(xs) - 1))], 2), "max": round(xs[-1], 2)}

        spans = [(b["t0"], b["t1"]) for b in blocks]
        for r in results:
            r["llm_busy"] = any(r["t0"] < e and r["t1"] > s0 for s0, e in spans)
        busy = [r for r in results if r["llm_busy"]]
        idle = [r for r in results if not r["llm_busy"]]
        summary = {
            "label": a.label, "model": a.model, "ctx": a.ctx, "mtp": a.mtp, "ot": a.ot,
            "whisper_dev": a.whisper_dev, "llm_dev": a.llm_dev,
            "audio_minutes": round(pos / SR / 60, 1), "run_s": round(run_s, 1),
            "whisper_warmup_s": warm, "vram_base_mib": vram.base, "vram_loaded_mib": loaded, "vram_peak_mib": vram.peak,
            "utterances": len(results), "forced_cuts": sum(r["forced"] for r in results),
            "empty_results": sum(1 for r in results if r["words"] == 0),
            "latency_idle": stats([r["latency_s"] for r in idle]),
            "latency_llm_busy": stats([r["latency_s"] for r in busy]),
            "infer_idle": stats([r["infer_s"] for r in idle]),
            "infer_llm_busy": stats([r["infer_s"] for r in busy]),
            "rtf_idle": stats([r["infer_s"] / r["dur_s"] for r in idle]),
            "max_queue": max([r["qdepth"] for r in results] or [0]),
            "vad_cpu_ms_per_frame": round(1000 * vad_time / max(1, pos // FRAME), 3),
            "blocks": blocks,
        }
        (out / ("%s-summary.json" % a.label)).write_text(json.dumps(summary, indent=2))
        (out / ("%s-utterances.json" % a.label)).write_text(json.dumps(results, indent=1))
        with open(out / ("%s-transcript.txt" % a.label), "w", encoding="utf-8") as f:
            for s, t in transcript:
                f.write("(%s) %s\n" % (hms(s), t))
        print(json.dumps({k: v for k, v in summary.items() if k != "blocks"}, indent=2))
        return 0
    finally:
        for p in (ff, whisper, llama):
            if p and p.poll() is None:
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)],
                               capture_output=True, creationflags=NO_WINDOW)


def hms(s: float) -> str:
    s = int(s)
    return "%02d:%02d:%02d" % (s // 3600, s % 3600 // 60, s % 60)


if __name__ == "__main__":
    sys.exit(main())
