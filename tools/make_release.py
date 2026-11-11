r"""Build the two release assets. Development only; not shipped.

    runtime/python.exe tools/make_release.py

A release carries two zips and they are not variants of the same thing:

  Meeting-Transcriber-Live-vX.Y.Z.zip        the source tree, well under 1 MB
  Meeting-Transcriber-Live-vX.Y.Z-full.zip   the same plus runtime\, bin\ and the small models

The small one is the **update payload** -- what the in-app updater downloads and
robocopies over an install. It must stay small and must never contain
`runtime\python.exe`: the updater would be overwriting the interpreter the
running app is executing from, and a half-copied interpreter cannot start, so
it cannot self-repair. `version.pick_asset` skips anything with `-full` in the
name for exactly that reason.

The full one is for a **first install**: unzip it and only the speech model and
the language models are left to download (DOWNLOAD_MODELS.bat skips the rest).
"""

from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from server import version  # noqa: E402

PREFIX = "Meeting-Transcriber-Live"
BUNDLED_DIRS = ("runtime", "bin")

# The models small enough to travel, named individually -- `models\` as a whole
# is about 27 GB. The two Silero VAD files are 3 MB together. Whisper
# large-v3-turbo is deliberately not here: it compresses to 1.42 GB, which would
# take the bundle past GitHub's 2 GB per-asset limit.
BUNDLED_FILES = (
    "models/silero_vad.onnx",
    "models/ggml-silero-v5.1.2.bin",
)
SKIP_PARTS = {"__pycache__"}
# This script is tracked so that a release is reproducible from the repository,
# but it is development tooling and has no business in an install.
EXCLUDE_PREFIXES = ("tools/",)


def unused(rel: str) -> bool:
    """Bundled files this app never touches, left out of the full zip (~96 MB).

    The runtime is the batch app's and the engine folders are whole upstream zips,
    so both carry what that app or the upstream demos needed. Removing them was
    tested end to end -- live capture, both whisper servers, notes, and a full
    summary -- and whisper's first-use GPU compile was run from cold without the
    nvrtc DLLs: the driver does that compile, not nvrtc (BUILD_NOTES section 8).
    """
    name = rel.rsplit("/", 1)[-1].lower()
    site = "runtime/Lib/site-packages/"
    if rel.startswith(site):
        pkg = rel[len(site):].split("/", 1)[0].lower()
        if pkg.startswith(("pip", "sherpa_onnx")):           # installer; batch diarizer
            return True
        if rel.startswith(site + "onnxruntime/") and rel[len(site + "onnxruntime/"):].split(
                "/", 1)[0] in ("transformers", "quantization", "tools"):
            return True                                      # model-conversion helpers
        return False
    if rel.startswith("bin/whisper-cuda/"):
        if name.endswith(".exe"):
            return name != "whisper-server.exe"               # demos, tests, the CLI
        return name in ("sdl2.dll", "llama.dll", "parakeet.dll") or name.startswith("nvrtc")
    if rel.startswith("bin/llama-cuda/"):
        if name.endswith(".exe"):
            return name != "llama-server.exe"
        return name.endswith("-impl.dll") and name != "llama-server-impl.dll"
    return False


def source_entries() -> list:
    """Every tracked file, so the zip matches the commit rather than the disk."""
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                         text=True, check=True)
    return [line for line in out.stdout.splitlines()
            if line.strip() and not line.startswith(EXCLUDE_PREFIXES)]


def write(target: Path, extras: bool) -> None:
    files = source_entries()
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for rel in files:
            src = ROOT / rel
            if src.is_file():
                z.write(src, "%s/%s" % (PREFIX, rel))
        if not extras:
            return
        for folder in BUNDLED_DIRS:
            base = ROOT / folder
            if not base.is_dir():
                raise SystemExit("missing %s -- run DOWNLOAD_MODELS.bat first" % folder)
            for src in base.rglob("*"):
                if not src.is_file() or SKIP_PARTS & set(src.parts):
                    continue
                rel = src.relative_to(ROOT).as_posix()
                if unused(rel):
                    continue
                z.write(src, "%s/%s" % (PREFIX, rel))
        for rel in BUNDLED_FILES:
            src = ROOT / rel
            if not src.is_file():
                raise SystemExit("missing %s -- run DOWNLOAD_MODELS.bat first" % rel)
            z.write(src, "%s/%s" % (PREFIX, rel))


def check_shippable_config() -> None:
    r"""Refuse to build if config.json carries this machine's local state.

    config.json is tracked -- it is the default a new install gets -- and the
    app also writes to it: the Model dropdown sets `llm.external`, the
    Microphone dropdown sets `capture.device`. Anyone who runs the app while
    developing dirties the file, and `git add -A` ships it.

    The batch app shipped `external.enabled: true` in three releases that way,
    so every fresh install looked for a server on a port nobody was running.

    Test settings are refused outright -- a replay file, short blocks, tiny
    chunks -- because each would ship a working-but-wrong app. The microphone
    name is reset instead: shipped, it is harmless (the app falls back to the
    Windows default), but it is somebody's hardware, not a default.
    """
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    problems = []
    if ((cfg.get("llm") or {}).get("external") or {}).get("enabled"):
        problems.append("llm.external.enabled is true (this machine's Port choice)")
    if (cfg.get("capture") or {}).get("replay_file"):
        problems.append("capture.replay_file is set (a development replay)")
    if int((cfg.get("notes") or {}).get("block_s", 300)) != 300:
        problems.append("notes.block_s is not 300 (a test value)")
    s = cfg.get("summary") or {}
    if int(s.get("target_tokens", 10000)) != 10000 or int(s.get("group_reduce_threshold", 8)) != 8:
        problems.append("summary chunking is not the default (a test value)")
    if problems:
        raise SystemExit("config.json is not shippable:\n  - " + "\n  - ".join(problems))

    device = (cfg.get("capture") or {}).get("device", "")
    if device:
        cfg["capture"]["device"] = ""
        (ROOT / "config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        print("config.json: capture.device was %r, reset to \"\" for the release" % device)


def main() -> None:
    check_shippable_config()
    v = version.APP_VERSION
    out = ROOT.parent
    plain = out / ("%s-v%s.zip" % (PREFIX, v))
    full = out / ("%s-v%s-full.zip" % (PREFIX, v))

    for target, extras in ((plain, False), (full, True)):
        print("building %s ..." % target.name)
        if target.exists():
            target.unlink()
        write(target, extras)
        size = target.stat().st_size
        print("  %d bytes (%.2f GB)" % (size, size / 1e9))
        if size > 2_000_000_000:
            raise SystemExit("%s exceeds GitHub's 2 GB per-asset limit" % target.name)

    print("\nassets for v%s:" % v)
    print("  %s   <- update payload, what pick_asset chooses" % plain.name)
    print("  %s   <- first install" % full.name)


if __name__ == "__main__":
    main()
