# Testing

The acceptance tests are defined in `CLAUDE.md` §16. This file tracks which have run, how,
and what is left. Measurements behind the results are in `BUILD_NOTES.md`.

**"Replay"** means a recording fed through the real app at real time instead of the
microphone (`capture.replay_file` in `config.json`, development only). It exercises
everything except the microphone itself.

## Status at v1.0.0

| # | Test | Status | How |
|---|---|---|---|
| 1 | Latency | **passed** | Replay, 3h43m: median 1.03-1.05 s every hour, worst 2.5 s (`BUILD_NOTES` §6) |
| 2 | No lost speech | not yet | |
| 3 | Silence | partly | Quiet 5-minute blocks get no note (verified); 10 minutes of a quiet room not yet run |
| 4 | Mixed language | **passed** | No Chinese characters, on replay and through the live mic (§3) |
| 5 | Blocks | **passed** | Replay: notes per block, never changed, partial block after Stop |
| 5a | Refine | **passed**, one part open | Lines replaced; refined text 21-31% from batch style (§4). Scroll anchoring during a swap not checked by hand |
| 6 | Notes are grounded | not yet | Needs a person to spot-check against the transcript |
| 7 | Three hours | **passed** | Replay, 3h43m; flat memory after `--cache-ram 0` (§6) |
| 8 | VRAM headroom | **passed** | 14,774 of 16,311 MiB peak (§5) |
| 9 | Reattach | partly | Reloading the page mid-meeting restored it and carried on. Closing and reopening the tab not tried |
| 10 | Unplug the mic | partly | Simulated by freezing ffmpeg: detected in 3.6 s, resumed 2 s later (§7). Real unplug and replug to confirm |
| 11 | Force-quit the console | not yet | Stopping the server cleanly left no child processes (§5); a force-quit not tried |
| 12 | Network adapter disabled | not yet | |
| 13 | Text in one place | **passed** | A word from the meeting found only in its `output\` folder (§5) |
| 14 | Port | **passed** | Against a long-context external server; never stopped (§6). Whisper's headroom there was thin |
| 15 | Portable | not yet | |
| 16 | Full summary | partly | Through the API on 12-minute and 3h43m meetings. The button and Save dialog used by the operator, not formally checked |
| 17 | Summary cancelled midway | partly | Cancelled by the operator; the file and process checks not recorded |
| 18 | Save dialog in front | not yet confirmed | |
| 19 | History | partly | List, open and delete (API) verified; rename after a restart and right-click by hand not confirmed |
| 20 | Crash safety | not yet | |
| 21 | Browse while live | not yet | |
| 22 | Summary from a past meeting | **passed** | Both full-summary runs were of past meetings |
| 23 | Mic check leaves nothing | **passed** | No history entry; the notes column says so (checked in the browser) |

## Hands-on checklist

The tests a replay cannot do. Each takes a few minutes unless noted.

1. **A real meeting** (6, 4): use it normally, then read the notes against the transcript.
   Anything in a note that was not said is a failure.
2. **Quiet room** (3): Start, say nothing for 10 minutes, Stop. No lines, no notes, no
   history entry.
3. **Unplug and replug** (10): mid-meeting, pull the mic. The red message should appear
   within about 4 s. Plug it into a *different* USB port. It should carry on by itself,
   and the times on the next lines should match the clock.
4. **Close the tab** (9): mid-meeting, close the browser tab, wait a minute, reopen
   `http://127.0.0.1:8000`. Everything should be there and still live.
5. **Force-quit** (11, 20): mid-meeting, close the black console window. In Task Manager,
   no `ffmpeg.exe`, `whisper-server.exe` or `llama-server.exe` should remain. Run `run.bat`
   again: the meeting should be in the history up to the moment it was closed.
6. **Generate Full Summary** (16, 18): on a meeting of at least 20 minutes. The Save As
   dialog must appear *in front of* the browser. Check the copy on the Desktop and the
   Full Summary tab.
7. **Cancel a summary** (17): start one and press Cancel. No summary file anywhere, no
   `llama-server.exe` left, the meeting still in the history.
8. **History** (19, 21): rename a meeting, restart the app, check the name stuck.
   Right-click → Delete another. During a live meeting, open a past one and use the
   *Live — back to meeting* bar to return.
9. **Offline** (12): disable the network adapter and run a short meeting end to end.
10. **Portable** (15): copy the whole folder to another drive and run it there. About 30 GB.

## For developers: replaying a recording

Set `capture.replay_file` in `config.json` to an audio file; Start then plays it through
the whole app at real time. `notes.block_s: 120` gives more blocks in a short replay.
`summary.target_tokens: 800` with `summary.group_reduce_threshold: 2` forces the
multi-tier summary path on a short meeting. **Put all three back before committing** —
`tools/make_release.py` refuses to build with any of them changed.

`tools/spike.py` runs the pipeline without the app (whisper + LLM, no UI); see its
docstring.
