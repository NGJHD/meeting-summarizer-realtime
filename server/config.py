"""Paths, config loading and the child-process environment.

Everything the app touches lives under ROOT so the folder stays portable
(CLAUDE.md section 17: nothing is written outside the app directory, apart
from a summary copy the user explicitly saves somewhere).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BIN = ROOT / "bin"
MODELS = ROOT / "models"
PROMPTS = ROOT / "prompts"
WEB = ROOT / "web"
TEMP = ROOT / "temp"
OUTPUT = ROOT / "output"

FFMPEG = BIN / "ffmpeg.exe"
# NVIDIA only (CLAUDE.md section 0.4), so one build per engine. whisper.cpp and
# llama.cpp ship different ggml DLLs and cannot share a folder; Windows looks in
# the executable's own folder first, and the CUDA runtime they share sits in bin\.
LLAMA_SERVER = BIN / "llama-cuda" / "llama-server.exe"
WHISPER_SERVER = BIN / "whisper-cuda" / "whisper-server.exe"

CONFIG_PATH = ROOT / "config.json"


def meeting_dir(folder: str) -> Path:
    """One meeting's folder under output\\. `folder` is never trusted as a path."""
    return OUTPUT / Path(folder).name


def write_atomic(path: Path, text: str) -> None:
    """Write a file so the previous version survives a failure mid-write.

    Write beside it and rename: os.replace is atomic on Windows within a
    volume, so a reader sees either the old file or the new one and never half
    of either. meeting.json and the transcript are rewritten while a meeting
    runs, and a power cut must not leave them truncated.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


# Files that must exist before the app can do anything useful. Checked at
# startup so the failure is a plain sentence instead of a crash mid-meeting.
REQUIRED_FILES = [
    FFMPEG,
    LLAMA_SERVER,
    WHISPER_SERVER,
    MODELS / "ggml-large-v3-turbo.bin",
    MODELS / "silero_vad.onnx",
]

_DEFAULTS = {
    "capture": {
        # The microphone's friendly name, written by the UI. Never the dshow
        # GUID: that changes when the mic moves to another USB port (section 4).
        "device": "",
        "block_ms": 32,
        # Development only: replay a recording at real time instead of the mic.
        "replay_file": "",
    },
    "vad": {
        "model": "models/silero_vad.onnx",
        "threshold": 0.5,
        "min_silence_ms": 600,
        "pad_ms": 200,
        "min_speech_ms": 250,
        "max_utterance_s": 15,
    },
    "asr": {
        "model": "models/ggml-large-v3-turbo.bin",
        "language": "en",
        "threads": 4,
        "port": 8081,
        "carry_context": True,
        "max_queue": 4,
        "drop_phrases": ["Thank you.", "Thanks for watching.", "you", "."],
        # The second pass: each finished block re-transcribed whole, on the CPU
        # (section 6.1).
        "refine": True,
        "refine_port": 8082,
        "refine_threads": "auto",
        "refine_vad_model": "models/ggml-silero-v5.1.2.bin",
    },
    "notes": {
        "block_s": 300,
        "min_words": 40,
        "max_tokens": 600,
    },
    "summary": {
        "model": "iq4_xs",
        "ctx_size": 32768,
        "target_tokens": 10000,
        "overlap_tokens": 400,
        "max_map_output_tokens": 1500,
        "group_reduce_threshold": 8,
        "group_size": 5,
        "max_group_output_tokens": 2500,
        "final_thinking": True,
        "final_effort": "medium",
        "final_budget_tokens": 4000,
    },
    "llm": {
        "model": "auto",
        "ctx_size": 8192,
        "gpu_layers": "auto",
        "cpu_ffn_regex": "auto",
        "cache_type_k": "q8_0",
        "cache_type_v": "q8_0",
        # "auto" omits --threads and lets llama.cpp use the physical core count.
        "threads": "auto",
        "port": 8080,
        "startup_timeout_s": 180,
        "idle_timeout_s": 300,
        # "auto" turns on the model's own multi-token prediction when it fits
        # beside whisper; "" forces it off, "draft-mtp" forces it on.
        "spec_type": "auto",
        # A llama-server somebody else is already running. Written by the Model
        # dropdown ("Port"); nothing else in this section is touched by the UI.
        "external": {"enabled": False, "port": 9931},
    },
    "server": {"port": 8000},
}


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config() -> dict:
    """Read config.json, falling back to defaults for anything absent.

    New keys are merged in from _DEFAULTS, so a config.json written by an older
    version loses nothing. A malformed file must not be papered over -- the
    operator edits it by hand -- so it raises with a plain message instead.
    """
    cfg = _DEFAULTS
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
                cfg = _merge(_DEFAULTS, json.load(fh))
        except (json.JSONDecodeError, OSError) as exc:
            raise RuntimeError(
                "config.json could not be read (%s). Fix or delete it." % exc
            ) from exc
    return cfg


def _edit_config(mutate) -> None:
    """Apply `mutate` to config.json's raw contents and write it back.

    Raw rather than through load_config(), so an operator's hand-edited config
    keeps exactly the keys they put in it: writing the merged result back would
    bake every current default into the file and freeze it against later ones.
    """
    raw: dict = {}
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except (json.JSONDecodeError, OSError):
            raise RuntimeError("config.json could not be read, so the choice "
                               "was not saved.")
    if not isinstance(raw, dict):
        raise RuntimeError("config.json is not an object, so the choice was "
                           "not saved.")
    mutate(raw)
    write_atomic(CONFIG_PATH, json.dumps(raw, indent=2) + "\n")


def save_llm_choice(external: bool, port: int | None = None) -> None:
    """Remember the Model dropdown's Port choice across restarts.

    High or Low clears the flag rather than pinning a quantisation: detection
    knows the machine better than a choice made once. The port survives either way.
    """
    def mutate(raw: dict) -> None:
        ext = raw.setdefault("llm", {}).setdefault("external", {})
        ext["enabled"] = bool(external)
        if port is not None:
            ext["port"] = int(port)

    _edit_config(mutate)


def save_device(name: str) -> None:
    """Remember the microphone by its friendly name (section 4)."""
    def mutate(raw: dict) -> None:
        raw.setdefault("capture", {})["device"] = str(name)

    _edit_config(mutate)


def external_llm(cfg: dict | None = None) -> tuple[bool, int]:
    """(is an external server configured, which port)."""
    if cfg is None:
        cfg = load_config()
    ext = cfg.get("llm", {}).get("external") or {}
    try:
        port = int(ext.get("port") or 9931)
    except (TypeError, ValueError):
        port = 9931
    return bool(ext.get("enabled")), port


def resolve(rel: str) -> Path:
    """Resolve a config path (always written relative to the app folder)."""
    p = Path(rel)
    return p if p.is_absolute() else (ROOT / p)


def child_env(gpu: bool = True) -> dict:
    """Environment for spawned binaries.

    bin\\ must be on PATH so ggml-cuda.dll can find cudart64_12.dll and the
    cuBLAS DLLs, which both engines share. Without it llama-server reports
    `invalid device` -- the CUDA backend never loaded (BUILD_NOTES section 1).

    CUDA_VISIBLE_DEVICES pins every child to the one card hardware.detect_gpu
    chose (section 8). By UUID, so the child sees exactly that card as device 0
    and the CUDA-versus-nvidia-smi numbering difference never comes into it.
    `gpu=False` hides every card, for the CPU refine server.
    """
    env = dict(os.environ)
    env["PATH"] = str(BIN) + os.pathsep + env.get("PATH", "")
    from . import hardware

    if not gpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    elif hardware.probed():
        uuid = hardware.detect_gpu().get("uuid") or ""
        if uuid:
            env["CUDA_VISIBLE_DEVICES"] = uuid
    return env


def missing_files() -> list[Path]:
    """Everything that must be present before a meeting can be transcribed."""
    missing = [p for p in REQUIRED_FILES if not p.exists()]
    try:
        from . import hardware

        cfg = load_config()
        if not external_llm(cfg)[0] and not hardware.any_model_present():
            missing.append(hardware.model_path(hardware.MODELS[-1]["key"]))
    except Exception:  # noqa: BLE001 - a broken config is reported elsewhere
        pass
    return missing
