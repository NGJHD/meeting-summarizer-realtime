# Changelog

## 1.0.0 — 2026-11-11

First release. A fully offline Windows app for in-room meetings, started from the batch
[Meeting-Summarizer](https://github.com/NGJHD/Meeting-Summarizer).

**Live transcript**
- Listens to a USB or built-in microphone and shows an English transcript about a second
  after each pause. Mixed English and Mandarin comes out as English.
- Every 5 minutes, the block just finished is transcribed again whole, on the processor,
  and its lines are replaced with the cleaner, punctuated text without moving what you are
  reading.
- Real 12-hour clock times on every line.

**Rolling notes**
- A short plain-English note about each 5-minute block, written by a local language model,
  appended in the right-hand column. Quiet blocks get no note.

**Full summary**
- One button writes a summary of the whole meeting, topic by topic, in paragraphs, on the
  High Quality model. It is kept with the meeting and a copy is saved wherever you choose.
  It can also be made later for any past meeting.

**History**
- Every meeting's transcript, notes and summary are kept as text in `output\`, written as
  they happen. Rename and delete from the sidebar (right-click). Sessions with almost
  nothing said are not kept. Audio is never saved.

**Robustness**
- The microphone is remembered by name and found again on another USB port.
- If the microphone drops out mid-meeting, the meeting stays open and carries on by itself
  when it is plugged back in.
- Runs on one NVIDIA card with 16 GB of video memory, on the card with the most memory if
  there are several. Measured stable over a 3h43m session.
- Optional Port mode sends the notes and summary to a llama-server you already run.
