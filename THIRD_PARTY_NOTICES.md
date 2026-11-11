# Third-party notices

The code in this repository is MIT (see `LICENSE`). Everything below is somebody else's
work, kept under its own licence; an assembled folder is a collection of separately
licensed works rather than a single one.

**Some of it is redistributed directly.** The `-full` release zip contains the Python
runtime, ffmpeg, the NVIDIA inference binaries and the two small voice detection models.
That is a distribution in the licensing sense, so the obligations attached to those
components — chiefly LGPL v3 for ffmpeg — are ours to meet, and this file is how they are
met. It must travel with the zip.

The rest is fetched by `DOWNLOAD_MODELS.bat` from its publisher and never passes through
us: the speech recognition model and the two language models, which are too large to be
release assets. Versions are pinned in `DOWNLOAD_MODELS.bat`, so this list stays true until
somebody deliberately changes a pin.

---

## The one that constrains redistribution: ffmpeg

| | |
|---|---|
| **Component** | FFmpeg `n9.0.1-27-g9b0578816c`, win64 **LGPL** build by [BtbN](https://github.com/BtbN/FFmpeg-Builds) |
| **Licence** | **LGPL v3** (`--enable-version3`, *not* `--enable-gpl`) |
| **Source** | https://github.com/FFmpeg/FFmpeg — build recipe at https://github.com/BtbN/FFmpeg-Builds |

The LGPL build is chosen **deliberately over the GPL one**: it captures from the microphone
(DirectShow) as well as the GPL build does, and a GPLv3 ffmpeg in the folder would place
GPLv3 obligations on anyone passing that folder on.

Under LGPL v3 you may redistribute the folder provided you keep this notice, do not
restrict the recipient's rights over ffmpeg itself, and can point them at the source
above. The app invokes `ffmpeg.exe` as a separate process and links nothing, so the MIT
licence on this project's own code is unaffected either way.

If you rebuild or replace `bin/ffmpeg.exe`, check `ffmpeg -version` for `--enable-gpl`
before shipping it.

---

## Inference engines

| Component | Version | Licence |
|---|---|---|
| [whisper.cpp](https://github.com/ggml-org/whisper.cpp) — `bin/whisper-cuda/` (`whisper-server`) | `b4938` | MIT |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) — `bin/llama-cuda/` (`llama-server`) | `b10852` | MIT |
| ggml (bundled with both) | — | MIT |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) — runs the voice detection model | 1.29.0 | MIT |

`bin/cudart64_12.dll`, `cublas64_12.dll` and `cublasLt64_12.dll` are the NVIDIA CUDA
runtime, redistributable under the [CUDA Toolkit EULA](https://docs.nvidia.com/cuda/eula/).

---

## Models

Marked **(shipped)** where the file is inside the `-full` zip rather than downloaded.

| Model | Used for | Licence |
|---|---|---|
| [Whisper large-v3-turbo](https://huggingface.co/ggerganov/whisper.cpp) (ggml) | Speech recognition | MIT |
| [Silero VAD v5.1.2](https://github.com/snakers4/silero-vad) (ONNX) **(shipped)** | Detecting speech, live | MIT |
| [Silero VAD v5.1.2](https://huggingface.co/ggml-org/whisper-vad) (ggml) **(shipped)** | Detecting speech, second pass | MIT |
| [Qwen3.8-27B](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) — UD-IQ3_XXS and UD-IQ4_XS | Notes and the full summary | Apache-2.0 |

---

## Python runtime

`runtime/` is CPython **3.12.14** from
[python-build-standalone](https://github.com/astral-sh/python-build-standalone), under the
**PSF Licence**, assembled by the batch Meeting Summariser project and reused as is. Its own
`runtime/LICENSE.txt` travels with it and covers CPython and the libraries CPython itself
bundles.

Vendored packages, as each declares itself:

| Package | Licence |
|---|---|
| fastapi, pydantic, pydantic_core, anyio, h11, annotated_types, annotated_doc, typing_inspection, pip | MIT |
| onnxruntime | MIT |
| starlette, uvicorn, click, idna | BSD-3-Clause |
| numpy | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| packaging | Apache-2.0 OR BSD-2-Clause |
| python_multipart, flatbuffers | Apache-2.0 |
| sherpa_onnx, sherpa_onnx_core (present in the shared runtime, not used by this app) | Apache-2.0 |
| protobuf | BSD-3-Clause |
| typing_extensions | PSF-2.0 |

There is deliberately no PyTorch, no transformers and no HuggingFace client — see
`CLAUDE.md` §0.

---

## If you redistribute the assembled folder

1. Keep this file and `LICENSE` in it.
2. Keep `runtime/LICENSE.txt`.
3. Be able to point recipients at the FFmpeg source (LGPL v3) — the link above is enough.

The same three apply to publishing the `-full` zip, which is a redistribution of all of
the above. Nothing here forbids commercial use or requires you to open your own changes,
provided `bin/ffmpeg.exe` remains the LGPL build.
