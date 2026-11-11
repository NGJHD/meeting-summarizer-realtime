r"""Meeting history: one folder per meeting under output\ (CLAUDE.md section 11.1).

    output\2026-11-11 1430\
      meeting.json     {"name", "started", "duration_s", "has_summary", "ended"}
      transcript.md    "(2:34:05 PM) text" lines, appended as each is recognised
      notes.md         one section per 5-minute block, appended as each lands
      summary.md       written once by Generate Full Summary

Text only, never audio (section 10). Everything is written as it happens and
flushed, so a crash loses at most the utterance in flight.
"""

from __future__ import annotations

import json
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Optional

from . import config

TRANSCRIPT = "transcript.md"
NOTES = "notes.md"
SUMMARY = "summary.md"
META = "meeting.json"

# Real clock time in 12-hour form: the operator asked for the time things were
# said, not the time since Start. The older session-time form is still read.
_CLOCK_RE = re.compile(r"^\((\d{1,2}):(\d{2}):(\d{2})\s*([AP]M)\)\s?(.*)$")
_LINE_RE = re.compile(r"^\((\d{2}):(\d{2}):(\d{2})\)\s?(.*)$")
_BLOCK_RE = re.compile(r"^<!-- block (\d+(?:\.\d+)?) (\d+(?:\.\d+)?) -->$")


def hms(seconds: float) -> str:
    s = max(0, int(seconds))
    return "%02d:%02d:%02d" % (s // 3600, s % 3600 // 60, s % 60)


def clock(epoch: float, seconds: bool = True) -> str:
    """Wall-clock time, 12-hour: "2:34:05 PM", or "2:34 PM" without seconds."""
    t = time.localtime(epoch)
    hour = t.tm_hour % 12 or 12
    half = "AM" if t.tm_hour < 12 else "PM"
    if seconds:
        return "%d:%02d:%02d %s" % (hour, t.tm_min, t.tm_sec, half)
    return "%d:%02d %s" % (hour, t.tm_min, half)


def clock_range(epoch_a: float, epoch_b: float) -> str:
    """ "2:30 – 2:35 PM", or "11:58 AM – 12:03 PM" across noon."""
    a, b = clock(epoch_a, False), clock(epoch_b, False)
    if a[-2:] == b[-2:]:
        return "%s – %s" % (a[:-3], b)
    return "%s – %s" % (a, b)


def default_name(started: float) -> str:
    """ "11 Nov 2026, 14:30" -- the label until the user renames it."""
    t = time.localtime(started)
    return "%d %s, %s" % (t.tm_mday, time.strftime("%b %Y", t), time.strftime("%H:%M", t))


class Meeting:
    """One meeting's folder. Thread-safe: the ASR, refine and notes threads all write."""

    def __init__(self, folder: Path):
        self.folder = folder
        self.id = folder.name
        self._lock = threading.Lock()
        self.meta = {}

    @classmethod
    def create(cls, started: float) -> "Meeting":
        base = time.strftime("%Y-%m-%d %H%M", time.localtime(started))
        folder = config.OUTPUT / base
        n = 2
        while folder.exists():                 # two Starts within one minute
            folder = config.OUTPUT / ("%s (%d)" % (base, n))
            n += 1
        folder.mkdir(parents=True)
        m = cls(folder)
        m.meta = {"name": default_name(started), "started": started, "duration_s": 0.0,
                  "has_summary": False, "ended": False}
        m.save_meta()
        (folder / TRANSCRIPT).write_text("", encoding="utf-8")
        (folder / NOTES).write_text("", encoding="utf-8")
        return m

    @classmethod
    def open(cls, folder_id: str) -> Optional["Meeting"]:
        folder = config.meeting_dir(folder_id)
        if not folder.is_dir():
            return None
        m = cls(folder)
        m.meta = m._read_meta()
        return m

    # -- metadata ----------------------------------------------------------

    def _read_meta(self) -> dict:
        """meeting.json, or a fallback built from the folder if it is missing or bad."""
        try:
            data = json.loads((self.folder / META).read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data.setdefault("name", self.id)
                data["has_summary"] = (self.folder / SUMMARY).exists()
                return data
        except (OSError, ValueError):
            pass
        try:
            started = time.mktime(time.strptime(self.id[:15], "%Y-%m-%d %H%M"))
        except ValueError:
            started = self.folder.stat().st_mtime
        self.meta = {"started": started}        # read_lines needs it to parse clock times
        return {"name": default_name(started), "started": started,
                "duration_s": self._last_line_s(), "ended": True,
                "has_summary": (self.folder / SUMMARY).exists()}

    @property
    def started(self) -> float:
        return float(self.meta.get("started") or 0.0)

    def stamp(self, t: float, seconds: bool = True) -> str:
        """Session seconds -> the clock time they were said at."""
        return clock(self.started + t, seconds)

    def save_meta(self) -> None:
        with self._lock:
            config.write_atomic(self.folder / META, json.dumps(self.meta, indent=2))

    def update_meta(self, **fields) -> None:
        self.meta.update(fields)
        self.save_meta()

    # -- transcript -----------------------------------------------------------

    def append_line(self, t: float, text: str) -> None:
        """One line, appended and flushed now -- not held back until Stop."""
        with self._lock:
            with open(self.folder / TRANSCRIPT, "a", encoding="utf-8") as fh:
                fh.write("(%s) %s\n" % (self.stamp(t), text))
                fh.flush()

    def rewrite_transcript(self, lines: list[tuple[float, str]]) -> None:
        """After a refine: the whole file again, atomically (section 6.1)."""
        body = "".join("(%s) %s\n" % (self.stamp(t), text) for t, text in lines)
        with self._lock:
            config.write_atomic(self.folder / TRANSCRIPT, body)

    def read_lines(self) -> list[tuple[float, str]]:
        out = []
        try:
            raw = (self.folder / TRANSCRIPT).read_text(encoding="utf-8")
        except OSError:
            return out
        start = time.localtime(self.started)
        start_sod = start.tm_hour * 3600 + start.tm_min * 60 + start.tm_sec
        for line in raw.splitlines():
            m = _CLOCK_RE.match(line)
            if m:
                hour = int(m.group(1)) % 12 + (12 if m.group(4) == "PM" else 0)
                sod = hour * 3600 + int(m.group(2)) * 60 + int(m.group(3))
                # Back to session seconds; a meeting may run past midnight.
                out.append(((sod - start_sod) % 86400, m.group(5)))
                continue
            m = _LINE_RE.match(line)
            if m:
                out.append((int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)),
                            m.group(4)))
        return out

    def _last_line_s(self) -> float:
        lines = self.read_lines()
        return float(lines[-1][0]) if lines else 0.0

    def word_count(self) -> int:
        return sum(len(text.split()) for _t, text in self.read_lines())

    # -- notes -----------------------------------------------------------

    def append_block(self, start_s: float, end_s: float, notes: str) -> None:
        """One block's notes. The comment line is what read_blocks parses."""
        entry = "<!-- block %.1f %.1f -->\n## %s\n\n%s\n\n" % (
            start_s, end_s, clock_range(self.started + start_s, self.started + end_s),
            notes.strip())
        with self._lock:
            with open(self.folder / NOTES, "a", encoding="utf-8") as fh:
                fh.write(entry)
                fh.flush()

    def read_blocks(self) -> list[dict]:
        try:
            raw = (self.folder / NOTES).read_text(encoding="utf-8")
        except OSError:
            return []
        blocks, cur, body = [], None, []
        for line in raw.splitlines():
            m = _BLOCK_RE.match(line)
            if m:
                if cur is not None:
                    cur["notes"] = "\n".join(body).strip()
                    blocks.append(cur)
                cur, body = {"start_s": float(m.group(1)), "end_s": float(m.group(2))}, []
                continue
            if cur is not None and not (not body and line.startswith("## ")):
                body.append(line)
        if cur is not None:
            cur["notes"] = "\n".join(body).strip()
            blocks.append(cur)
        return blocks

    # -- summary ---------------------------------------------------------

    def save_summary(self, text: str) -> None:
        with self._lock:
            config.write_atomic(self.folder / SUMMARY, text.rstrip() + "\n")
        self.update_meta(has_summary=True)

    def read_summary(self) -> str:
        try:
            return (self.folder / SUMMARY).read_text(encoding="utf-8")
        except OSError:
            return ""

    # -- views -----------------------------------------------------------

    def summary_row(self) -> dict:
        return {"id": self.id, "name": self.meta.get("name") or self.id,
                "started": self.meta.get("started", 0),
                "duration_s": self.meta.get("duration_s", 0),
                "has_summary": bool(self.meta.get("has_summary"))}

    def full(self) -> dict:
        return {**self.summary_row(),
                "lines": [{"t": t, "text": text, "refined": True} for t, text in self.read_lines()],
                "blocks": [{**b, "status": "done"} for b in self.read_blocks()],
                "summary": self.read_summary()}


def list_meetings(exclude: str = "") -> list[dict]:
    if not config.OUTPUT.exists():
        return []
    rows = []
    for folder in config.OUTPUT.iterdir():
        if not folder.is_dir() or folder.name == exclude:
            continue
        if not ((folder / META).exists() or (folder / TRANSCRIPT).exists()):
            continue
        m = Meeting.open(folder.name)
        if m is not None:
            rows.append(m.summary_row())
    rows.sort(key=lambda r: r["started"], reverse=True)
    return rows


def rename(folder_id: str, name: str) -> bool:
    """The display name only. The folder never changes, so nothing can break."""
    m = Meeting.open(folder_id)
    name = " ".join(str(name).split())[:120]
    if m is None or not name:
        return False
    m.update_meta(name=name)
    return True


def delete(folder_id: str) -> bool:
    folder = config.meeting_dir(folder_id)
    if not folder.is_dir() or folder.parent != config.OUTPUT:
        return False
    shutil.rmtree(folder, ignore_errors=True)
    return not folder.exists()


def recover_interrupted() -> int:
    """At startup: a meeting with no clean Stop gets its duration from its last line.

    The partial block at the crash is never summarised (section 11.1).
    """
    if not config.OUTPUT.exists():
        return 0
    fixed = 0
    for folder in config.OUTPUT.iterdir():
        if not folder.is_dir():
            continue
        m = Meeting.open(folder.name)
        if m is None or m.meta.get("ended", True):
            continue
        m.update_meta(ended=True, duration_s=max(float(m.meta.get("duration_s") or 0),
                                                  m._last_line_s()))
        fixed += 1
    return fixed
