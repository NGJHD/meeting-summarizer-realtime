@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

rem ---------------------------------------------------------------------------
rem  Meeting Transcriber Live - model and binary downloader
rem
rem  Run this ONCE on a machine that has internet access. The application
rem  itself never touches the network (apart from the Check for updates button);
rem  this script exists only so the models do not have to ship inside the folder.
rem
rem  Needs curl.exe and tar.exe, both included with Windows 10 1803 and later.
rem ---------------------------------------------------------------------------

echo.
echo   Meeting Transcriber Live - downloading models
echo   ---------------------------------------------
echo.
echo   Total download: about 28 GB - the Python runtime, ffmpeg, the speech
echo   recognition and voice detection models, two language models and the
echo   NVIDIA inference binaries. This will take a while.
echo   Already-downloaded files are skipped, so it is safe to re-run
echo   this script if the connection drops.
echo.

if not exist "models" mkdir "models"

where curl.exe >nul 2>&1
if errorlevel 1 (
  echo   ERROR: curl.exe was not found.
  echo   It ships with Windows 10 version 1803 and later.
  echo.
  pause
  exit /b 1
)

set "FAILED="

rem  Every URL below is pinned to an exact revision -- a HuggingFace commit SHA,
rem  a GitHub release tag or a git tag -- never to a branch head, so a new
rem  machine gets the same bytes every measurement in BUILD_NOTES.md was taken
rem  against. To move to a newer model, change the pin deliberately and re-measure.
rem
rem  get <target file> <minimum size in bytes> <url> <description>
call :get "models\ggml-large-v3-turbo.bin" 1500000000 ^
  "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-large-v3-turbo.bin?download=true" ^
  "Speech recognition model - 1.6 GB"

rem  Two copies of Silero VAD v5, in two formats, for two jobs. The ONNX one is
rem  our own live segmenter (CLAUDE.md section 5); the ggml one is whisper's
rem  own VAD on the refine pass (section 6.1).
call :get "models\silero_vad.onnx" 2000000 ^
  "https://github.com/snakers4/silero-vad/raw/v5.1.2/src/silero_vad/data/silero_vad.onnx" ^
  "Voice detection model - 2 MB"

call :get "models\ggml-silero-v5.1.2.bin" 800000 ^
  "https://huggingface.co/ggml-org/whisper-vad/resolve/9ffd54a1e1ee413ddf265af9913beaf518d1639b/ggml-silero-v5.1.2.bin?download=true" ^
  "Voice detection model for the second pass - 1 MB"

rem  Low Quality runs live, beside whisper. High Quality writes the full
rem  summary once whisper has been unloaded; without it the summary falls back
rem  to Low Quality, so it is optional.
call :get "models\Qwen3.8-27B-UD-IQ3_XXS.gguf" 10000000000 ^
  "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-IQ3_XXS.gguf?download=true" ^
  "Language model, Low Quality - 10.9 GB"

call :get_optional "models\Qwen3.8-27B-UD-IQ4_XS.gguf" 14000000000 ^
  "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-IQ4_XS.gguf?download=true" ^
  "Language model, High Quality - 14.3 GB, this is the long one"

rem ---------------------------------------------------------------------------
rem  The Python runtime, ffmpeg and the NVIDIA inference binaries.
rem
rem  The runtime is the batch Meeting Summariser's assembled environment (exact
rem  CPython + exact wheels, no pip at install time), hosted on that project's
rem  releases: it already carries everything this app needs -- fastapi, uvicorn,
rem  numpy, onnxruntime. ffmpeg is the LGPL build, pinned to a dated release.
rem ---------------------------------------------------------------------------

call :getzip "runtime" "runtime\python.exe" 80000000 ^
  "https://github.com/NGJHD/Meeting-Summarizer/releases/download/runtime-cpython-3.12.14/runtime-cpython.zip" ^
  "Python runtime - 89 MB"

call :getffmpeg

set "LLAMA_BUILD=b10852"
set "WHISPER_BUILD=b4938"
set "GH=https://github.com/ggml-org"

call :getbin "llama-cuda" "llama-server.exe" 90000000 ^
  "%GH%/llama.cpp/releases/download/%LLAMA_BUILD%/llama-%LLAMA_BUILD%-bin-win-cuda-12.4-x64.zip" ^
  "llama.cpp for NVIDIA - 254 MB"

rem  The whole zip is kept: whisper-server.exe is in it, and it is what this app
rem  runs. A copy made from the batch app has whisper-cli.exe only, which is why
rem  whisper-server.exe is what is looked for.
call :getbin "whisper-cuda" "whisper-server.exe" 300000000 ^
  "%GH%/whisper.cpp/releases/download/%WHISPER_BUILD%/whisper-cublas-12.4.0-bin-x64.zip" ^
  "whisper.cpp for NVIDIA - 671 MB"

rem  The CUDA runtime is shared by both engines and sits in bin\ itself, where
rem  Windows finds it on PATH for either.
call :getbin "" "cudart64_12.dll" 380000000 ^
  "%GH%/llama.cpp/releases/download/%LLAMA_BUILD%/cudart-llama-bin-win-cuda-12.4-x64.zip" ^
  "NVIDIA CUDA runtime - 391 MB"

echo.
echo   ---------------------------------------------
if defined FAILED (
  echo   SOME DOWNLOADS FAILED. Re-run this script; finished files are skipped.
) else (
  echo   All models and binaries downloaded.
  echo.
  echo   Start the app with run.bat. You can also copy this whole folder
  echo   to another machine with an NVIDIA graphics card.
)
echo.
pause
exit /b 0

rem ---------------------------------------------------------------------------
:getzip
rem  %1 destination folder  %2 sentinel file  %3 minimum bytes  %4 url  %5 description
set "DEST=%~1"
set "SENTINEL=%~2"
set "MINSIZE=%~3"
set "URL=%~4"
set "DESC=%~5"

if exist "%SENTINEL%" (
  echo   [skip] %DESC% - already present
  exit /b 0
)
echo   [....] %DESC%
if not exist "temp" mkdir "temp"
curl.exe -L --fail --retry 5 --retry-delay 5 -C - -# -o "temp\_pkg.zip" "%URL%"
if errorlevel 1 (
  echo   [FAIL] %DESC%
  set "FAILED=1"
  exit /b 1
)
for %%A in ("temp\_pkg.zip") do set "SIZE=%%~zA"
if !SIZE! LSS %MINSIZE% (
  echo   [FAIL] %DESC% - file is smaller than expected
  set "FAILED=1"
  del /Q "temp\_pkg.zip"
  exit /b 1
)
if not exist "%DEST%" mkdir "%DEST%"
"%SystemRoot%\System32\tar.exe" -xf "temp\_pkg.zip" -C "%DEST%"
if errorlevel 1 (
  echo   [FAIL] %DESC% - could not unpack
  set "FAILED=1"
  del /Q "temp\_pkg.zip"
  exit /b 1
)
del /Q "temp\_pkg.zip"
echo   [ ok ] %DESC%
exit /b 0

rem ---------------------------------------------------------------------------
:getffmpeg
rem  Only ffmpeg.exe is kept; the archive also carries ffplay and ffprobe,
rem  which this app never calls.
if exist "bin\ffmpeg.exe" (
  echo   [skip] ffmpeg - already present
  exit /b 0
)
echo   [....] ffmpeg, LGPL build - 147 MB
if not exist "temp" mkdir "temp"
set "FFTAG=autobuild-2026-09-07-15-39"
set "FFNAME=ffmpeg-n9.0.1-27-g9b0578816c-win64-lgpl-9.0"
curl.exe -L --fail --retry 5 --retry-delay 5 -C - -# -o "temp\_ff.zip" ^
  "https://github.com/BtbN/FFmpeg-Builds/releases/download/%FFTAG%/%FFNAME%.zip"
if errorlevel 1 (
  echo   [FAIL] ffmpeg
  set "FAILED=1"
  exit /b 1
)
"%SystemRoot%\System32\tar.exe" -xf "temp\_ff.zip" -C "temp" "%FFNAME%/bin/ffmpeg.exe"
if not exist "temp\%FFNAME%\bin\ffmpeg.exe" (
  echo   [FAIL] ffmpeg - archive did not contain ffmpeg.exe
  set "FAILED=1"
  del /Q "temp\_ff.zip"
  exit /b 1
)
if not exist "bin" mkdir "bin"
move /Y "temp\%FFNAME%\bin\ffmpeg.exe" "bin\ffmpeg.exe" >nul
rmdir /S /Q "temp\%FFNAME%" 2>nul
del /Q "temp\_ff.zip"
echo   [ ok ] ffmpeg, LGPL build - 147 MB
exit /b 0

rem ---------------------------------------------------------------------------
:getbin
rem  %1 folder under bin\ - blank means bin\ itself   %2 the file that proves it
rem  is already installed   %3 minimum bytes   %4 url   %5 description
set "SUB=%~1"
set "SENTINEL=%~2"
set "MINSIZE=%~3"
set "URL=%~4"
set "DESC=%~5"
if "%SUB%"=="" ( set "DEST=bin" ) else ( set "DEST=bin\%SUB%" )

if exist "%DEST%\%SENTINEL%" (
  echo   [skip] %DESC% - already present
  exit /b 0
)

echo   [....] %DESC%
if not exist "temp" mkdir "temp"
curl.exe -L --fail --retry 5 --retry-delay 5 -C - -# -o "temp\_bin.zip" "%URL%"
if errorlevel 1 (
  echo   [FAIL] %DESC%
  set "FAILED=1"
  exit /b 1
)
for %%A in ("temp\_bin.zip") do set "SIZE=%%~zA"
if !SIZE! LSS %MINSIZE% (
  echo   [FAIL] %DESC% - file is smaller than expected
  set "FAILED=1"
  del /Q "temp\_bin.zip"
  exit /b 1
)

if not exist "%DEST%" mkdir "%DEST%"
rem  bsdtar ships with Windows and reads zip. The absolute System32 path is
rem  deliberate: a `tar` on PATH may be GNU tar, which cannot.
"%SystemRoot%\System32\tar.exe" -xf "temp\_bin.zip" -C "%DEST%"
if errorlevel 1 (
  echo   [FAIL] %DESC% - could not unpack
  set "FAILED=1"
  del /Q "temp\_bin.zip"
  exit /b 1
)
del /Q "temp\_bin.zip"

rem  The whisper zip wraps everything in Release\; flatten it.
if exist "%DEST%\Release\" (
  move /Y "%DEST%\Release\*" "%DEST%\" >nul 2>&1
  rmdir /S /Q "%DEST%\Release" 2>nul
)
if not exist "%DEST%\%SENTINEL%" (
  echo   [FAIL] %DESC% - the download did not contain %SENTINEL%
  set "FAILED=1"
  exit /b 1
)
echo   [ ok ] %DESC%
exit /b 0

rem ---------------------------------------------------------------------------
:get_optional
rem  As :get, but a failure is not an install failure. Used for the High Quality
rem  model: without it the full summary uses Low Quality. Marking FAILED here
rem  would tell the user their install is broken when it is not.
rem  Remember whether anything had already failed: clearing FAILED
rem  unconditionally would erase an earlier, real failure.
set "WASFAILED=%FAILED%"
call :get %1 %2 %3 %4
if errorlevel 1 (
  set "FAILED=%WASFAILED%"
  echo   [skip] %~4 - optional, continuing without it
)
exit /b 0

:get
rem  %1 target  %2 minimum bytes  %3 url  %4 description
rem  NOTE: the description must not contain ( or ) -- it is echoed inside
rem  a parenthesised if-block, and cmd would treat them as block delimiters.
set "TARGET=%~1"
set "MINSIZE=%~2"
set "URL=%~3"
set "DESC=%~4"

if exist "%TARGET%" (
  for %%A in ("%TARGET%") do set "SIZE=%%~zA"
  if !SIZE! GEQ %MINSIZE% (
    echo   [skip] %DESC% - already present
    exit /b 0
  )
  echo   [redo] %DESC% - previous file was incomplete
  del /Q "%TARGET%"
)

echo   [....] %DESC%
rem  -C - resumes a partial download rather than starting over.
curl.exe -L --fail --retry 5 --retry-delay 5 -C - -# -o "%TARGET%" "%URL%"
if errorlevel 1 (
  echo   [FAIL] %DESC%
  set "FAILED=1"
  exit /b 1
)

for %%A in ("%TARGET%") do set "SIZE=%%~zA"
if !SIZE! LSS %MINSIZE% (
  echo   [FAIL] %DESC% - file is smaller than expected ^(!SIZE! bytes^)
  set "FAILED=1"
  exit /b 1
)
echo   [ ok ] %DESC%
exit /b 0
