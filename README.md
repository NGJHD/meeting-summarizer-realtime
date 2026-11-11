# Meeting Transcriber Live

A fully offline Windows app for in-room meetings. It listens to a microphone, shows a live
English transcript a second or two behind the speaker, and writes a short plain-English note
about every 5 minutes of the meeting. When the meeting ends, one button writes a full summary
of it, topic by topic.

- **Offline.** Speech recognition (whisper.cpp) and the notes (llama.cpp) run on this
  machine. Nothing is sent anywhere; the only network use is the *Check for updates*
  button.
- **Mixed English and Mandarin** comes out as English.
- **History.** Every meeting's transcript and notes are kept, as text, in the app's
  `output\` folder and listed in the sidebar. Audio is never saved.
- **Real clock times** on every line and note.

## Requirements

- Windows 11
- An NVIDIA graphics card with 16 GB of video memory (one card is used, even if there are
  several)
- 32 GB of system RAM
- A microphone in the room
- About 30 GB of disk space

## Installing

1. Download **`Meeting-Transcriber-Live-vX.Y.Z-full.zip`** from the latest release and
   unzip it.
2. Double-click **`DOWNLOAD_MODELS.bat`** once, on a machine with internet access. It
   fetches the speech model and the language models (about 27 GB). Re-running it skips
   anything already downloaded.

After that the folder is self-contained: copy it anywhere, to any drive, and it works
without a network connection.

## Using it

1. Double-click **`run.bat`**. Keep its black window open; closing it stops the app. The
   browser opens by itself.
2. Pick the **Microphone**. The thin bar under it moves when the mic hears sound — check it
   before the meeting starts. If the mic is plugged into a different USB port next time,
   the app finds it again by name.
3. Press **Start**. The first start loads the models (up to a minute).
   - Lines appear in grey within a second or two of someone pausing.
   - Every 5 minutes, that block is transcribed again more carefully and the grey lines
     are replaced with the cleaner text. A short note about the block then appears on the
     right. A block where almost nothing was said gets no note.
   - If the microphone comes loose, a red message says so and the meeting waits. Plug it
     back in — any USB port — and it carries on by itself.
4. Press **Stop** when the meeting ends. The meeting is saved in the **History** sidebar.
   If almost nothing was said, it is not saved, and the notes column says so.
5. **Generate Full Summary** (below the transcript) stops listening if the meeting is still
   running, and writes a summary of the whole meeting: each topic discussed and how it
   ended. It is kept with the meeting, and a copy is saved wherever you choose (the Desktop
   by default). It takes a few minutes; it can also be run later on any past meeting.

Right-click a meeting in the sidebar to rename or delete it.

## Settings

There is no settings screen. Everything tunable is in **`config.json`**, which can be edited
in Notepad — see `CLAUDE.md` §14 for what each value does.

## For developers

| File | What is in it |
|---|---|
| `CLAUDE.md` | The specification, and the reasoning behind every design decision |
| `BUILD_NOTES.md` | The measurements behind those decisions |
| `TESTING.md` | Acceptance tests, what has run, and how to replay a recording through the app |
| `CHANGELOG.md` | What changed in each release |
| `RELEASING.md` | How to build and publish a release |

This app started from [Meeting-Summarizer](https://github.com/NGJHD/Meeting-Summarizer), the
batch version that turns a recording into minutes.

## License

MIT — see `LICENSE`. Third-party components are listed in `THIRD_PARTY_NOTICES.md`.
