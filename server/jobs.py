"""Process tracking, cancellation, the job log and temp cleanup.

The batch app's staged Job, with its stage weights and ETA, is gone. What is
left is what every long-running piece of work here needs: a log line, a cancel
flag, and a record of the child processes it started so they can be killed --
by Stop, by shutdown, and by Windows itself if the console is force-quit.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
import traceback
from pathlib import Path

from . import config

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class Cancelled(Exception):
    """Raised inside a task when the user cancels."""


class JobError(Exception):
    """A failure with a message already fit for a non-technical user."""

    def __init__(self, message: str, detail: str = ""):
        super().__init__(message)
        self.message = message
        self.detail = detail


class Task:
    """Logging, cancellation and child processes for one piece of work.

    Never logs meeting content: transcript text, notes and prompts stay out of
    temp\\job.log (CLAUDE.md section 10). Callers log timings and counts only.
    """

    def __init__(self, name: str):
        self.name = name
        self.cancel_event = threading.Event()
        self.procs: list[subprocess.Popen] = []
        self.model_key = ""
        self.current_call = None
        self.on_log = None            # optional callback(line) for the UI

    @property
    def cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def check_cancelled(self) -> None:
        if self.cancel_event.is_set():
            raise Cancelled()

    def log(self, text: str) -> None:
        line = "[%s] %s: %s" % (time.strftime("%H:%M:%S"), self.name, text)
        _append_job_log(line)
        if self.on_log is not None:
            try:
                self.on_log(line)
            except Exception:  # noqa: BLE001 - the UI is never worth a crash
                pass

    def register_proc(self, proc: subprocess.Popen) -> None:
        self.procs.append(proc)
        _register_global(proc)
        assign_to_job_object(proc.pid)

    def unregister_proc(self, proc: subprocess.Popen) -> None:
        try:
            self.procs.remove(proc)
        except ValueError:
            pass
        _unregister_global(proc)

    def cancel(self) -> None:
        self.cancel_event.set()


def kill_tree(proc: subprocess.Popen) -> None:
    """Kill a child *process tree*. /T matters: a surviving child keeps VRAM."""
    if proc is None or proc.poll() is not None:
        return
    try:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       capture_output=True, creationflags=CREATE_NO_WINDOW, timeout=15)
    except Exception:  # noqa: BLE001
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass


# Every child the app has started, so shutdown can reach them whichever task
# owns them.
_procs: list[subprocess.Popen] = []
_procs_lock = threading.Lock()


def _register_global(proc: subprocess.Popen) -> None:
    with _procs_lock:
        _procs.append(proc)


def _unregister_global(proc: subprocess.Popen) -> None:
    with _procs_lock:
        try:
            _procs.remove(proc)
        except ValueError:
            pass


def shutdown_all() -> None:
    """Kill every child process and clear temp. Safe to call more than once."""
    with _procs_lock:
        procs = list(_procs)
        _procs.clear()
    for proc in procs:
        kill_tree(proc)
    clean_temp()


def kill_stale(ports: list[int]) -> int:
    """Kill whisper-server / llama-server left on our ports by a crash.

    A stale one holds VRAM, and the new one would fail to load beside it
    (CLAUDE.md section 12). Only our own executables, only on our own ports:
    an external llama-server on the Port option's port is never touched.
    """
    try:
        out = subprocess.run(["netstat", "-ano", "-p", "tcp"], capture_output=True,
                             text=True, timeout=15, creationflags=CREATE_NO_WINDOW).stdout
    except Exception:  # noqa: BLE001
        return 0
    pids = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[3] == "LISTENING":
            port = parts[1].rsplit(":", 1)[-1]
            if port.isdigit() and int(port) in ports:
                pids.add(parts[4])
    killed = 0
    for pid in pids:
        try:
            info = subprocess.run(["tasklist", "/FI", "PID eq %s" % pid, "/FO", "CSV", "/NH"],
                                  capture_output=True, text=True, timeout=10,
                                  creationflags=CREATE_NO_WINDOW).stdout.lower()
        except Exception:  # noqa: BLE001
            continue
        if "whisper-server.exe" in info or "llama-server.exe" in info:
            subprocess.run(["taskkill", "/F", "/T", "/PID", pid], capture_output=True,
                           creationflags=CREATE_NO_WINDOW, timeout=15)
            killed += 1
    return killed


# ---------------------------------------------------------------------------
# job log -- timings, counts and errors only, never meeting content
# ---------------------------------------------------------------------------

_log_lock = threading.Lock()


def job_log_path() -> Path:
    return config.TEMP / "job.log"


def _append_job_log(text: str) -> None:
    with _log_lock:
        try:
            config.TEMP.mkdir(parents=True, exist_ok=True)
            with open(job_log_path(), "a", encoding="utf-8", errors="replace") as fh:
                fh.write(text.rstrip() + "\n")
        except OSError:
            pass


def log(text: str) -> None:
    _append_job_log("[%s] %s" % (time.strftime("%H:%M:%S"), text))


def log_exception(exc: BaseException) -> str:
    """The traceback, to the log. Exception messages here never carry content."""
    detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    _append_job_log(detail)
    return detail


def read_job_log() -> str:
    """The bundle behind "Copy diagnostic info": our log plus the servers' tails."""
    parts = []
    try:
        parts.append(job_log_path().read_text(encoding="utf-8", errors="replace"))
    except OSError:
        pass
    for name in ("llama-server.log", "whisper-gpu.log", "whisper-cpu.log"):
        try:
            tail = (config.TEMP / name).read_text(
                encoding="utf-8", errors="replace").splitlines()[-60:]
        except OSError:
            continue
        if tail:
            parts.append("\n--- %s (last %d lines) ---\n%s" % (name, len(tail), "\n".join(tail)))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# temp cleanup
# ---------------------------------------------------------------------------

_TEMP_GLOBS = ["*.log", "*.tmp", "*.partial", "savedialog-*.ps1"]


def clean_temp(keep_log: bool = True) -> None:
    if not config.TEMP.exists():
        return
    for pattern in _TEMP_GLOBS:
        for path in config.TEMP.glob(pattern):
            if keep_log and path.name == "job.log":
                continue
            try:
                path.unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# kill-on-close job object
# ---------------------------------------------------------------------------

# atexit and signal handlers do not run when a console window is force-quit,
# which would strand whisper-server.exe or llama-server.exe holding VRAM until
# reboot (CLAUDE.md section 16, test 11). A Windows job object with
# KILL_ON_JOB_CLOSE is the only mechanism the OS honours unconditionally: when
# this process dies for any reason, its handle closes and every process
# assigned to the job is terminated with it.

_job_handle = None

_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9


def install_kill_on_close() -> None:
    global _job_handle
    if _job_handle is not None or os.name != "nt":
        return
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.POINTER(ctypes.c_ulong)),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            return
        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
            handle,
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(info),
            ctypes.sizeof(info),
        ):
            kernel32.CloseHandle(handle)
            return
        _job_handle = handle
    except Exception:  # noqa: BLE001 - best effort; taskkill remains the fallback
        _job_handle = None


def assign_to_job_object(pid: int) -> None:
    """Put a spawned process into the kill-on-close job object."""
    if _job_handle is None or os.name != "nt":
        return
    try:
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        PROCESS_SET_QUOTA, PROCESS_TERMINATE = 0x0100, 0x0001
        h = kernel32.OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, pid)
        if not h:
            return
        try:
            kernel32.AssignProcessToJobObject(_job_handle, h)
        finally:
            kernel32.CloseHandle(h)
    except Exception:  # noqa: BLE001
        pass
