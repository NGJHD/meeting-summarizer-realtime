# Meeting Transcriber Live

A fully offline Windows app that listens to a microphone in a meeting room, shows a live
English transcript a second or two behind the speaker, and keeps a column of short notes
summarising each 5-minute block of the meeting.

- Runs entirely on the local machine: whisper.cpp for speech, llama.cpp for the notes.
- Mixed English and Mandarin speech comes out as English.
- Every meeting's transcript and notes are kept, as text, in a history sidebar. Audio is
  never saved.
- One button writes a full summary of the meeting, topic by topic.
- NVIDIA GPU with 16 GB of VRAM, Windows 11.

**Status: in development.** This repository was started from
[Meeting-Summarizer](https://github.com/NGJHD/Meeting-Summarizer), the batch version that
turns a recording into minutes. The kept code does not run yet. `CLAUDE.md` is the
specification and the build order.

## License

MIT — see `LICENSE`.
