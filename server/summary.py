"""Generate Full Summary (CLAUDE.md section 7.1).

The batch app's tested path, restored rather than reinvented: /tokenize-sized
chunks on line boundaries with an overlap, map with thinking off, group reduce
when there are more chunks than the threshold (recursively), then one final
reduce with thinking on. Speakers, minutes mode and calibration are gone.

Runs on High Quality over the full transcript, not the 5-minute notes: by then
whisper is unloaded and the model has the whole card.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import config, hardware, history, jobs, llm
from .jobs import Cancelled, JobError, Task

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

PLACEHOLDER = "_(This part of the meeting could not be summarised.)_"
PROMPT_ALLOWANCE = 1000          # the instructions that wrap every chunk
MIN_TARGET_TOKENS = 1500
# Thinking and the document share this. 12000 holds a 4000-token think and a
# whole document (batch BUILD_NOTES 9x); an external server's think is not
# capped by us, so there, spend spare context instead.
REDUCE_MAX_TOKENS = 12000
REDUCE_MAX_TOKENS_EXTERNAL = 24000

# Share of the progress bar per phase.
_PHASES = {"load": (0.0, 0.05), "map": (0.05, 0.80), "group": (0.80, 0.85),
           "final": (0.85, 1.0)}


# ---------------------------------------------------------------------------
# the Save As dialog -- opened by the server, so the path survives a closed tab
# ---------------------------------------------------------------------------

_DIALOG_PS = r'''
Add-Type -AssemblyName System.Windows.Forms
[System.Windows.Forms.Application]::EnableVisualStyles()
$owner = New-Object System.Windows.Forms.Form
$owner.TopMost = $true
$owner.ShowInTaskbar = $false
$owner.StartPosition = "CenterScreen"
$owner.Opacity = 0
$owner.Size = New-Object System.Drawing.Size(1, 1)
$owner.Show()
$owner.Activate()
$d = New-Object System.Windows.Forms.SaveFileDialog
$d.Title = "Save a copy of the summary"
$d.Filter = "Markdown (*.md)|*.md"
$d.DefaultExt = "md"
$d.AddExtension = $true
$d.OverwritePrompt = $true
$d.InitialDirectory = [Environment]::GetFolderPath("Desktop")
$d.FileName = $env:MTL_DEFAULT_NAME
$r = $d.ShowDialog($owner)
$owner.Close()
if ($r -eq [System.Windows.Forms.DialogResult]::OK) { [Console]::Out.Write($d.FileName) }
'''


def save_dialog(default_name: str) -> str:
    """The chosen path, or "" if the user cancelled. Blocks until they choose."""
    env = dict(os.environ)
    env["MTL_DEFAULT_NAME"] = _safe_filename(default_name)
    try:
        proc = subprocess.run(["powershell", "-STA", "-NoProfile", "-NonInteractive",
                               "-Command", _DIALOG_PS], capture_output=True, text=True,
                              encoding="utf-8", errors="replace", env=env, timeout=3600,
                              creationflags=CREATE_NO_WINDOW)
    except (OSError, subprocess.SubprocessError) as exc:
        jobs.log_exception(exc)
        return ""
    return (proc.stdout or "").strip()


def _safe_filename(name: str) -> str:
    bad = '<>:"/\\|?*'
    out = "".join("-" if c in bad else c for c in name).strip(" .")
    return out or "Meeting summary.md"


def default_copy_name(meeting_name: str) -> str:
    return _safe_filename("%s - summary.md" % meeting_name)


# ---------------------------------------------------------------------------
# chunking (batch chunker.py, lines instead of speaker turns)
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    index: int = 0
    total: int = 0
    lines: list = field(default_factory=list)
    tokens: int = 0
    start_s: float = 0.0
    end_s: float = 0.0

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def fits(prompt_tokens: int, max_output_tokens: int, ctx_size: int) -> bool:
    return prompt_tokens + max_output_tokens + 2000 <= ctx_size


def build_chunks(task: Task, server: llm.LlamaServer, lines: list[tuple[float, str]],
                 s: dict, meeting: history.Meeting) -> list[Chunk]:
    target = int(s.get("target_tokens", 10000))
    overlap = int(s.get("overlap_tokens", 400))
    room = server.ctx_size - int(s.get("max_map_output_tokens", 4500)) - 2000 - PROMPT_ALLOWANCE
    if room < target:
        target = max(room, MIN_TARGET_TOKENS)
        overlap = min(overlap, max(target // 8, 1))
        task.log("summary: %d-token context, chunking at %d" % (server.ctx_size, target))

    # Real clock times, so the summary cites them (operator's choice, 2026-11-11).
    rendered = ["(%s) %s" % (meeting.stamp(t), text) for t, text in lines]
    sep = max(server.token_count("\n\n") - server.token_count("\n"), 0)
    counts = []
    for i, line in enumerate(rendered):
        if i % 200 == 0:
            task.check_cancelled()
        counts.append(server.token_count(line) + sep)
    starts = [t for t, _ in lines]

    chunks: list[Chunk] = []
    i, n = 0, len(rendered)
    while i < n:
        task.check_cancelled()
        c = Chunk(start_s=starts[i])
        j = i
        while j < n and (c.tokens + counts[j] <= target or not c.lines):
            c.lines.append(rendered[j])
            c.tokens += counts[j]
            c.end_s = starts[j]
            j += 1
        chunks.append(c)
        if j >= n:
            break
        back, k = 0, j - 1
        while k > i and back + counts[k] <= overlap:
            back += counts[k]
            k -= 1
        i = max(k + 1, i + 1)
    for idx, c in enumerate(chunks, 1):
        c.index, c.total = idx, len(chunks)
    task.log("summary: %d lines -> %d chunks (target %d, overlap %d)"
             % (n, len(chunks), target, overlap))
    return chunks


# ---------------------------------------------------------------------------
# the job
# ---------------------------------------------------------------------------

class SummaryJob:
    def __init__(self, meeting_id: str, copy_path: str, emit):
        self.meeting_id = meeting_id
        self.copy_path = copy_path
        self.emit = emit
        self.task = Task("summary")
        self.server: Optional[llm.LlamaServer] = None
        self.phase = "starting"
        self.message = "Starting…"
        self.fraction = 0.0
        self.state = "running"        # running | done | failed | cancelled
        self.copy_error = ""

    def progress(self, phase: str, within: float, message: str = "") -> None:
        lo, hi = _PHASES.get(phase, (0.0, 1.0))
        self.phase = phase
        self.fraction = lo + (hi - lo) * max(0.0, min(1.0, within))
        if message:
            self.message = message
        self.emit({"type": "summary_progress", **self.view()})

    def view(self) -> dict:
        return {"id": self.meeting_id, "phase": self.phase, "message": self.message,
                "fraction": round(self.fraction, 4), "state": self.state,
                "copy_path": self.copy_path, "copy_error": self.copy_error}

    def cancel(self) -> None:
        self.task.cancel()
        if self.server is not None:
            self.server.abort()

    def run(self, cfg: dict, external: bool) -> str:
        """Returns the summary text. Raises Cancelled or JobError."""
        meeting = history.Meeting.open(self.meeting_id)
        if meeting is None:
            raise JobError("That meeting is no longer in the history.")
        lines = meeting.read_lines()
        if not lines:
            raise JobError("There is no transcript to summarise.")
        s = cfg["summary"]

        if external:
            self.progress("load", 0.2, "Connecting to the language model…")
            self.server = llm.LlamaServer(self.task, cfg, external=True)
        else:
            key = hardware.summary_key(cfg)
            if key != "iq4_xs":
                self.task.log("summary: High Quality model absent, using %s" % key)
            self.progress("load", 0.2, "Loading the High Quality model (up to 2 minutes)…")
            self.server = llm.LlamaServer(self.task, cfg, hardware.summary_plan(key, cfg))
        try:
            self.server.start()
            self.progress("load", 1.0)
            chunks = build_chunks(self.task, self.server, lines, s, meeting)
            notes = self._map(chunks, s, meeting)
            spans = [(meeting.stamp(c.start_s, False), meeting.stamp(c.end_s, False))
                     for c in chunks]
            notes, spans = self._group(notes, spans, s)
            notes, spans = self._fit_final(notes, spans, s, meeting)
            return self._final(notes, s, meeting)
        finally:
            self.server.stop()
            self.server = None

    def _guard(self, prompt: str, max_out: int, n: Optional[int] = None) -> int:
        if n is None:
            n = self.server.token_count(prompt)
        if not fits(n, max_out, self.server.ctx_size):
            raise JobError("That meeting produced more material than the model can work "
                           "with in one pass. Its transcript is safe in the history.",
                           "prompt %d + output %d + 2000 > ctx %d"
                           % (n, max_out, self.server.ctx_size))
        return n

    def _map(self, chunks: list[Chunk], s: dict, meeting: history.Meeting) -> list[str]:
        template = llm.load_prompt("summary_map")
        max_out = int(s.get("max_map_output_tokens", 4500))
        notes = []
        for c in chunks:
            self.task.check_cancelled()
            done, total = c.index - 1, max(len(chunks), 1)
            self.progress("map", done / total, "Reading the meeting: part %d of %d"
                          % (c.index, c.total))
            prompt = llm.fill(template, transcript=c.text, chunk_index=c.index,
                              chunk_total=c.total, time_start=meeting.stamp(c.start_s, False),
                              time_end=meeting.stamp(c.end_s, False))
            self._guard(prompt, max_out)
            try:
                text = self.server.chat(
                    prompt, max_out, thinking=False, expect_seconds=60,
                    progress=lambda f, d=done, t=total: self.progress("map", (d + f) / t))
            except Cancelled:
                raise
            except JobError as exc:
                # One bad chunk must not destroy the whole summary.
                jobs.log_exception(exc)
                self.task.log("summary: map chunk %d failed, placeholder inserted" % c.index)
                text = "## Themes\n%s\n" % PLACEHOLDER
            # The span, not "part 3 of 4": the reader never saw any parts.
            notes.append("### %s - %s\n\n%s" % (meeting.stamp(c.start_s, False),
                                                meeting.stamp(c.end_s, False), text))
        self.progress("map", 1.0)
        return notes

    def _group(self, notes: list[str], spans: list, s: dict, force_below: int = 0):
        threshold = force_below or int(s.get("group_reduce_threshold", 8))
        size = max(2, int(s.get("group_size", 5)))
        max_out = int(s.get("max_group_output_tokens", 2500))
        template = llm.load_prompt("summary_group")
        tier = 0
        while len(notes) > threshold:
            tier += 1
            groups = [notes[i:i + size] for i in range(0, len(notes), size)]
            gspans = [spans[i:i + size] for i in range(0, len(spans), size)]
            self.task.log("summary: group tier %d, %d notes -> %d groups"
                          % (tier, len(notes), len(groups)))
            out_n, out_s = [], []
            for i, (group, gs) in enumerate(zip(groups, gspans), 1):
                self.task.check_cancelled()
                self.progress("group", (i - 1) / len(groups),
                              "Consolidating: part %d of %d" % (i, len(groups)))
                start, end = gs[0][0], gs[-1][1]
                prompt = llm.fill(template, notes="\n\n".join(group),
                                  time_start=start, time_end=end)
                self._guard(prompt, max_out)
                try:
                    text = self.server.chat(prompt, max_out, thinking=False, expect_seconds=90)
                except Cancelled:
                    raise
                except JobError as exc:
                    jobs.log_exception(exc)
                    text = "\n\n".join(group)
                out_n.append("### %s - %s\n\n%s" % (start, end, text))
                out_s.append((start, end))
            notes, spans = out_n, out_s
        self.progress("group", 1.0)
        return notes, spans

    def _final_prompt(self, notes: list[str], meeting: history.Meeting) -> str:
        return llm.fill(llm.load_prompt("summary_final"), notes="\n\n".join(notes),
                        meeting_name=meeting.meta.get("name", meeting.id),
                        duration=_duration(float(meeting.meta.get("duration_s") or 0)))

    def _max_out(self, prompt_tokens: int) -> int:
        if not self.server.external:
            return REDUCE_MAX_TOKENS
        room = self.server.ctx_size - prompt_tokens - 2000
        return max(REDUCE_MAX_TOKENS, min(REDUCE_MAX_TOKENS_EXTERNAL, room))

    def _fit_final(self, notes, spans, s, meeting):
        """Extra group tiers until the real assembled final prompt fits."""
        for _ in range(4):
            n = self.server.token_count(self._final_prompt(notes, meeting))
            if fits(n, REDUCE_MAX_TOKENS, self.server.ctx_size) or len(notes) < 2:
                break
            self.task.log("summary: final prompt %d tokens; one more tier" % n)
            notes, spans = self._group(notes, spans, s, force_below=max(1, len(notes) - 1))
        return notes, spans

    def _final(self, notes: list[str], s: dict, meeting: history.Meeting) -> str:
        self.progress("final", 0.0, "Writing the summary…")
        prompt = self._final_prompt(notes, meeting)
        n = self.server.token_count(prompt)
        max_out = self._max_out(n)
        self._guard(prompt, max_out, n=n)
        think = bool(s.get("final_thinking", True))
        self.task.log("summary: final %d prompt tokens, max %d, thinking %s" % (n, max_out, think))
        return self.server.chat(prompt, max_out, thinking=think,
                                effort=str(s.get("final_effort", "medium")), expect_seconds=240,
                                progress=lambda f: self.progress("final", f))


def _duration(seconds: float) -> str:
    m = int(seconds // 60)
    return "%dh %02dm" % (m // 60, m % 60) if m >= 60 else "%dm" % max(m, 1)


def write_copy(path: str, text: str) -> str:
    """Write the user's copy. Returns "" or a plain error sentence."""
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".partial")
        tmp.write_text(text.rstrip() + "\n", encoding="utf-8")
        os.replace(tmp, target)
        return ""
    except OSError as exc:
        jobs.log("summary: copy not written (%s)" % type(exc).__name__)
        return "Saved in the history, but not to %s." % path
