"""Microphone capture through ffmpeg's dshow input (CLAUDE.md section 4).

    ffmpeg -f dshow -audio_buffer_size 50 -i audio="<name>" -ac 1 -ar 16000 -f s16le pipe:1

The PCM never touches disk: it is read off the pipe in 32 ms frames on a
dedicated thread, which must never stall -- if it does, ffmpeg's buffer fills,
dshow drops audio, and the gap turns up later as words missing for no reason.
"""

from __future__ import annotations

import re
import subprocess
import threading
import time
from typing import Callable, Optional

import numpy as np

from . import config
from .jobs import Task, kill_tree

CREATE_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

SR = 16000
FRAME = 512                      # 32 ms at 16 kHz -- Silero v5's frame
LEVEL_INTERVAL_S = 0.2           # ~5 Hz to the meter

_DEVICE_RE = re.compile(r'"([^"]+)"\s+\(audio\)')
# Windows' instance numbering for a second copy of the same device, as in
# "Microphone (2- BOYA CM30)". A mic with no serial number gets a new instance,
# and often this prefix, on every USB port (BUILD_NOTES section 3).
_INSTANCE_RE = re.compile(r"\(\d+-\s*")


def list_devices() -> list[str]:
    """Audio capture devices as dshow names them. Re-read every time (section 4)."""
    try:
        proc = subprocess.run(
            [str(config.FFMPEG), "-hide_banner", "-list_devices", "true", "-f", "dshow",
             "-i", "dummy"], capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=20, creationflags=CREATE_NO_WINDOW)
    except (OSError, subprocess.SubprocessError):
        return []
    seen, out = set(), []
    for m in _DEVICE_RE.finditer(proc.stderr or ""):
        name = m.group(1)
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def normalise(name: str) -> str:
    """The device's name with Windows' per-port instance prefix removed."""
    return _INSTANCE_RE.sub("(", name).strip().lower()


_DEFAULT_INPUT_PS = r'''
Add-Type -TypeDefinition @"
using System; using System.Runtime.InteropServices;
[Guid("A95664D2-9614-4F35-A746-DE8DB63617E6"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IMMDeviceEnumerator { int NotImpl1(); [PreserveSig] int GetDefaultAudioEndpoint(int dataFlow, int role, out IMMDevice ppDevice); }
[Guid("D666063F-1587-4E43-81F1-B948E807363F"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IMMDevice { int NotImpl1(); [PreserveSig] int OpenPropertyStore(int stgmAccess, out IPropertyStore ppProperties); }
[Guid("886d8eeb-8cf2-4446-8d02-cdba1dbdcf99"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IPropertyStore { int GetCount(out int c); int GetAt(int i, out PROPERTYKEY k); [PreserveSig] int GetValue(ref PROPERTYKEY key, out PROPVARIANT pv); }
[StructLayout(LayoutKind.Sequential)] struct PROPERTYKEY { public Guid fmtid; public int pid; }
[StructLayout(LayoutKind.Sequential)] struct PROPVARIANT { public short vt; short r1, r2, r3; public IntPtr p; IntPtr p2; }
[ComImport, Guid("BCDE0395-E52F-467C-8E3D-C4579291692E")] class MMDeviceEnumerator { }
public static class DefaultInput {
  public static string Get() {
    var e = (IMMDeviceEnumerator)(new MMDeviceEnumerator()); IMMDevice d;
    if (e.GetDefaultAudioEndpoint(1, 0, out d) != 0) return "";
    IPropertyStore s; d.OpenPropertyStore(0, out s);
    var k = new PROPERTYKEY { fmtid = new Guid("a45c254e-df1c-4efd-8020-67d146a850e0"), pid = 14 };
    PROPVARIANT v; s.GetValue(ref k, out v); return Marshal.PtrToStringUni(v.p);
  }
}
"@
[DefaultInput]::Get()
'''


def windows_default_input() -> str:
    """The Windows default recording device's friendly name, or "".

    Asked of Core Audio through PowerShell, which is slow (~1 s) -- so only on
    the fallback path, when the remembered mic is not there.
    """
    try:
        proc = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                               _DEFAULT_INPUT_PS], capture_output=True, text=True,
                              timeout=20, creationflags=CREATE_NO_WINDOW)
    except (OSError, subprocess.SubprocessError):
        return ""
    lines = [l.strip() for l in (proc.stdout or "").splitlines() if l.strip()]
    return lines[-1] if lines else ""


def match_device(remembered: str, devices: list[str]) -> tuple[str, str]:
    """(device to use, how it was found) -- CLAUDE.md section 4's order.

    how: "exact" | "renamed" (same mic, another USB port) | "default" | "first" | "none".
    """
    if not devices:
        return "", "none"
    if remembered and remembered in devices:
        return remembered, "exact"
    if remembered:
        target = normalise(remembered)
        same = [d for d in devices if normalise(d) == target]
        if len(same) == 1:
            return same[0], "renamed"
    default = windows_default_input()
    if default:
        for d in devices:
            if d == default or normalise(d) == normalise(default):
                return d, "default"
    return devices[0], "first"


class Capture:
    """One ffmpeg reading one device (or, in replay mode, one file) into frames.

    `on_frame(pcm_int16, frame_index)` is called for every 32 ms frame on the
    reader thread. `on_level(rms_dbfs)` about five times a second. `on_end()`
    once, if ffmpeg ends without stop() having been called -- the mic was
    unplugged, or the replay file ran out.
    """

    def __init__(self, task: Task, device: str, replay_file: str = "",
                 on_frame: Optional[Callable] = None, on_level: Optional[Callable] = None,
                 on_end: Optional[Callable] = None):
        self.task = task
        self.device = device
        self.replay_file = replay_file
        self.on_frame = on_frame
        self.on_level = on_level
        self.on_end = on_end
        self.proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._stopping = threading.Event()
        self.frames = 0

    def command(self) -> list[str]:
        if self.replay_file:
            src = ["-re", "-i", str(config.resolve(self.replay_file))]
        else:
            src = ["-f", "dshow", "-audio_buffer_size", "50", "-i", "audio=" + self.device]
        return ([str(config.FFMPEG), "-hide_banner", "-loglevel", "error", "-nostdin"]
                + src + ["-ac", "1", "-ar", str(SR), "-f", "s16le", "pipe:1"])

    def start(self) -> None:
        self.proc = subprocess.Popen(
            self.command(), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL, bufsize=0,
            creationflags=CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW)
        self.task.register_proc(self.proc)
        self._thread = threading.Thread(target=self._read, name="capture", daemon=True)
        self._thread.start()

    def _read(self) -> None:
        need = FRAME * 2
        buf = b""
        acc, acc_n = 0.0, 0
        last_level = time.time()
        stream = self.proc.stdout
        try:
            while True:
                chunk = stream.read(need - len(buf))
                if not chunk:
                    break
                buf += chunk
                if len(buf) < need:
                    continue
                pcm = np.frombuffer(buf, dtype=np.int16)
                buf = b""
                if self.on_frame is not None:
                    self.on_frame(pcm, self.frames)
                self.frames += 1
                f = pcm.astype(np.float32)
                acc += float(np.dot(f, f))
                acc_n += len(f)
                now = time.time()
                if now - last_level >= LEVEL_INTERVAL_S and self.on_level is not None:
                    rms = (acc / max(acc_n, 1)) ** 0.5
                    self.on_level(20.0 * np.log10(max(rms, 1e-3) / 32768.0))
                    acc, acc_n, last_level = 0.0, 0, now
        except Exception as exc:  # noqa: BLE001 - logged; on_end still fires
            self.task.log("capture: reader stopped (%s)" % type(exc).__name__)
        finally:
            ended_by_us = self._stopping.is_set()
            if self.proc is not None:
                self.task.unregister_proc(self.proc)
            if not ended_by_us and self.on_end is not None:
                self.on_end()

    def stop(self) -> None:
        self._stopping.set()
        proc = self.proc
        if proc is not None:
            kill_tree(proc)
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=5)

    @property
    def seconds(self) -> float:
        return self.frames * FRAME / SR
