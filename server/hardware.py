"""NVIDIA detection, model choice and VRAM placement (CLAUDE.md section 8).

NVIDIA only (section 0.4). The batch app's Vulkan, CPU and unified-memory paths
are gone; their measurements live in that repo's BUILD_NOTES.

Two budgets, because the card is shared differently at the two times the LLM
runs:

- **Live:** whisper (GPU) is resident beside the LLM for the whole meeting, so
  the LLM gets what whisper leaves. Measured: IQ3_XXS + MTP + whisper peaked at
  13.9 GiB on a 16 GB card (BUILD_NOTES section 2).
- **Full summary:** whisper has been unloaded, so the LLM has the whole card at
  32k context -- the batch app's arithmetic, which ran IQ4_XS + MTP whole on the
  same card.
"""

from __future__ import annotations

import re
import subprocess
import threading

from . import config

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# --- live budget, GiB. Measured on the 16 GB card, layout A (BUILD_NOTES 2). ---
# large-v3-turbo resident with its CUDA context and buffers.
WHISPER_RESERVE_GIB = 1.9
# KV cache at ctx 8192 (q8_0) plus llama.cpp's compute buffers. Measured at
# about 0.56; rounded up because being wrong here pages the model.
LIVE_OVERHEAD_GIB = 0.8
# The draft context for --spec-type draft-mtp.
MTP_GIB = 0.7
# What Windows and the desktop already hold on the card (~500 MiB measured).
DESKTOP_GIB = 0.5
# Section 16 test 8: at least this much left over during a block call.
HEADROOM_GIB = 0.5

# --- summary budget: the batch app's, at ctx 32768 with whisper gone. ---
# IQ3_XXS sat at 11.69 GiB resident with 10.18 GiB of weights.
SUMMARY_OVERHEAD_GIB = 1.5

# Qwen3.8-27B has 64 transformer blocks, 0-63.
NUM_LAYERS = 64
# Roughly the share of a block's weights that the FFN tensors account for, and
# therefore what moving one block to the CPU frees on the GPU.
FFN_FRACTION = 0.67

MODELS = [
    {
        "key": "iq4_xs",
        "file": "Qwen3.8-27B-UD-IQ4_XS.gguf",
        "label": "High Quality: Qwen3.8-27B-UD-IQ4_XS",
        "size_gb": 14.3,
        "url": "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/"
               "4ca720788d1e01f1bff70c033e0d0028fd02e502/"
               "Qwen3.8-27B-UD-IQ4_XS.gguf?download=true",
        "min_size": 14_000_000_000,
        "spec": "draft-mtp",
    },
    {
        "key": "iq3_xxs",
        "file": "Qwen3.8-27B-UD-IQ3_XXS.gguf",
        "label": "Low Quality: Qwen3.8-27B-UD-IQ3_XXS",
        "size_gb": 10.9,
        "url": "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/"
               "4ca720788d1e01f1bff70c033e0d0028fd02e502/"
               "Qwen3.8-27B-UD-IQ3_XXS.gguf?download=true",
        "min_size": 10_000_000_000,
        "spec": "draft-mtp",
    },
]
BY_KEY = {m["key"]: m for m in MODELS}

_probe_cache: dict = {}
_probe_lock = threading.Lock()


def _run(cmd: list, timeout: int = 20) -> str:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              creationflags=CREATE_NO_WINDOW)
    except (OSError, subprocess.SubprocessError):
        return ""
    return (proc.stdout or "") + (proc.stderr or "")


def _probe() -> dict:
    """Every NVIDIA card, then the one both servers will use.

    The largest by VRAM, and by name and UUID rather than by index: nvidia-smi
    and CUDA number the cards differently, and both servers default to CUDA
    device 0, which on the dev machine is the 10 GB card (section 8).
    """
    cards = []
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total,uuid",
                "--format=csv,noheader,nounits"], timeout=15)
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3 and parts[1].isdigit():
            cards.append({"name": parts[0], "vram_mb": int(parts[1]), "uuid": parts[2]})
    if not cards:
        return {"vendor": "none", "vram_mb": 0, "device": "no NVIDIA GPU found",
                "uuid": "", "cards": []}
    chosen = max(cards, key=lambda c: c["vram_mb"])
    return {"vendor": "nvidia", "vram_mb": chosen["vram_mb"], "device": chosen["name"],
            "uuid": chosen["uuid"], "cards": cards}


def detect_gpu() -> dict:
    """The card this machine will use. Cached: it shells out to nvidia-smi."""
    if _probe_cache:
        return dict(_probe_cache)
    with _probe_lock:
        if not _probe_cache:
            _probe_cache.update(_probe())
    return dict(_probe_cache)


def probed() -> bool:
    """True once detect_gpu has run. Lets child_env avoid triggering it."""
    return bool(_probe_cache)


def free_vram_mb() -> int:
    """Free memory on the chosen card right now, for the Port check (section 8)."""
    uuid = detect_gpu().get("uuid")
    if not uuid:
        return 0
    out = _run(["nvidia-smi", "--id=%s" % uuid, "--query-gpu=memory.free",
                "--format=csv,noheader,nounits"], timeout=15)
    m = re.search(r"\d+", out)
    return int(m.group()) if m else 0


def model_path(key: str):
    return config.MODELS / BY_KEY[key]["file"]


def any_model_present() -> bool:
    return any(model_path(m["key"]).exists() for m in MODELS)


def model_gib(key: str) -> float:
    """The weights' real size in GiB, from the file where possible.

    GiB on both sides of every comparison: mixing decimal GB with GiB once
    overstated every model by 7% and offloaded blocks that would have fitted.
    """
    path = model_path(key)
    if path.exists():
        return path.stat().st_size / float(2 ** 30)
    return BY_KEY[key]["size_gb"] * 1e9 / float(2 ** 30)


def _regex_for(first_layer: int) -> str:
    """Offload the FFN tensors of `first_layer`..63, as an explicit alternation."""
    first = max(0, min(first_layer, NUM_LAYERS - 1))
    return r"blk\.(%s)\.ffn_.*=CPU" % "|".join(str(n) for n in range(first, NUM_LAYERS))


def _offload_for(key: str, available_gib: float, overhead_gib: float) -> str:
    """FFN blocks to push to system RAM so the rest fits in `available_gib`."""
    size = model_gib(key)
    deficit = (size + overhead_gib) - available_gib
    if deficit <= 0:
        return ""
    per_layer = size * FFN_FRACTION / NUM_LAYERS
    return _regex_for(NUM_LAYERS - min(NUM_LAYERS, int(deficit / per_layer) + 1))


def _live_need(key: str, mtp: bool) -> float:
    return (model_gib(key) + LIVE_OVERHEAD_GIB + (MTP_GIB if mtp else 0.0)
            + WHISPER_RESERVE_GIB + DESKTOP_GIB + HEADROOM_GIB)


def choose_live_key(vram_mb: int | None = None) -> str:
    """The model for the live meeting: the best one that fits beside whisper *whole*.

    High Quality is only chosen if it fits without offload, with its KV cache and
    whisper alongside (section 8). On a 16 GB card that resolves to Low Quality;
    measured, IQ4_XS there leaves 0.33 GiB, under the 0.5 GiB floor.
    """
    if vram_mb is None:
        vram_mb = detect_gpu()["vram_mb"]
    here = [m["key"] for m in MODELS if model_path(m["key"]).exists()]
    if not here:
        return MODELS[-1]["key"]
    for key in here:                      # MODELS is ordered best first
        if _live_need(key, mtp=False) <= vram_mb / 1024.0:
            return key
    return here[-1]


def live_plan(key: str, cfg: dict) -> dict:
    """How to launch llama-server for the live meeting on this card."""
    llm = cfg["llm"]
    vram = detect_gpu()["vram_mb"] / 1024.0
    available = vram - WHISPER_RESERVE_GIB - DESKTOP_GIB - HEADROOM_GIB
    regex = _offload_for(key, available, LIVE_OVERHEAD_GIB)
    spec = str(llm.get("spec_type", "auto"))
    if spec == "auto":
        # MTP costs ~700 MiB: only where it still fits whole beside whisper.
        spec = (BY_KEY[key].get("spec") or "") if (
            not regex and _live_need(key, mtp=True) <= vram) else ""
    layers = str(llm.get("gpu_layers", "auto"))
    explicit = str(llm.get("cpu_ffn_regex") or "")
    return {
        "key": key,
        "path": model_path(key),
        "ctx_size": int(llm.get("ctx_size", 8192)),
        "layers": "99" if layers == "auto" else layers,
        "regex": regex if explicit == "auto" else explicit,
        "spec": spec,
        "budget": 0,                      # thinking is off live (section 7)
    }


def summary_plan(key: str, cfg: dict) -> dict:
    """How to launch llama-server for the full summary: whisper is gone by then."""
    s = cfg["summary"]
    vram = detect_gpu()["vram_mb"] / 1024.0
    regex = _offload_for(key, vram, SUMMARY_OVERHEAD_GIB)
    spec = (BY_KEY[key].get("spec") or "") if not regex else ""
    budget = int(s.get("final_budget_tokens") or 0) if s.get("final_thinking", True) else 0
    return {
        "key": key,
        "path": model_path(key),
        "ctx_size": int(s.get("ctx_size", 32768)),
        "layers": "99",
        "regex": regex,
        "spec": spec,
        "budget": max(0, budget),
    }


def summary_key(cfg: dict) -> str:
    """High Quality for the summary if it is on disk, else whatever is (section 7.1)."""
    wanted = str(cfg["summary"].get("model") or "iq4_xs")
    if wanted in BY_KEY and model_path(wanted).exists():
        return wanted
    for m in MODELS:
        if model_path(m["key"]).exists():
            return m["key"]
    return wanted


def describe(cfg: dict) -> dict:
    """Everything the Model dropdown needs."""
    gpu = detect_gpu()
    recommended = choose_live_key(gpu["vram_mb"])
    external, port = config.external_llm(cfg)
    return {
        "vram_mb": gpu["vram_mb"],
        "device": gpu["device"],
        "recommended": recommended,
        "selected": "external" if external else recommended,
        "external": {"enabled": external, "port": port},
        "models": [
            {
                "key": m["key"],
                "label": m["label"],
                "available": model_path(m["key"]).exists(),
                "fits": _live_need(m["key"], mtp=False) <= gpu["vram_mb"] / 1024.0,
            }
            for m in MODELS
        ],
    }
