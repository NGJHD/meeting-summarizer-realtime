# Releasing

How to build and publish a release. You need this repository, the app's full working
folder (`runtime\`, `bin\` and the small models present — run `DOWNLOAD_MODELS.bat` once),
and the GitHub CLI signed in (`gh auth status`).

## 1. Prepare

1. **`config.json` must be the shipped defaults.** The app writes to it while you use it
   (Microphone, Model dropdowns), and testing changes it (`capture.replay_file`,
   `notes.block_s`, summary chunking). `tools/make_release.py` refuses to build with any of
   those left changed, and clears the remembered microphone.
2. **Set the version** in `server/version.py` (`APP_VERSION`). The in-app updater compares
   it with the release tag, so they must match: version `1.2.3` is tag `v1.2.3`.
3. **Add a `CHANGELOG.md` entry.**
4. **Re-read the diff and the release notes for anything from a real meeting** — names,
   organisations, systems, figures, quoted lines, anything a model wrote from one
   (`CLAUDE.md` §10). A release is public and permanent once the repository is.
5. Commit and push.

## 2. Build

```
runtime\python.exe tools\make_release.py
```

This writes two zips **beside the repository folder** (in its parent):

| Asset | Contents | Used by |
|---|---|---|
| `Meeting-Transcriber-Live-vX.Y.Z.zip` | the tracked source files, under 1 MB | the in-app **Check for updates** button |
| `Meeting-Transcriber-Live-vX.Y.Z-full.zip` | the same plus `runtime\`, `bin\` and the two voice detection models, minus files this app never uses (`make_release.unused`) | a first install |

The full zip must stay under GitHub's 2 GB per-asset limit; the script stops if it does
not. The speech and language models are never bundled — `DOWNLOAD_MODELS.bat` fetches them.

The updater must never take the full zip (it would overwrite the running interpreter), so
`version.pick_asset` skips anything with `-full` in its name. Keep that suffix.

## 3. Publish

```
gh release create vX.Y.Z ^
  "..\Meeting-Transcriber-Live-vX.Y.Z.zip" ^
  "..\Meeting-Transcriber-Live-vX.Y.Z-full.zip" ^
  --title "vX.Y.Z" --notes-file <notes.md>
```

The full zip is about 1 GB, so the upload takes a while. If it fails part-way, GitHub keeps
the release as a **draft**: upload the missing file with `gh release upload vX.Y.Z <zip>`
and publish with `gh release edit vX.Y.Z --draft=false`.

## 4. Check

```
runtime\python.exe tools\check_release.py
```

It asks GitHub what an installed copy would see: the tag matches the version, both assets
are attached, the updater picks the small one, and the Python runtime that
`DOWNLOAD_MODELS.bat` fetches from the batch app's releases still exists.

Both this check and the in-app update button ask GitHub's API without signing in, so they
only see releases while the repository is public (it has been since v1.0.0).
