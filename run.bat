@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONUNBUFFERED=1"
set "PATH=%~dp0bin;%PATH%"

if not exist "temp"   mkdir "temp"
if not exist "output" mkdir "output"

rem ---- preflight: name the missing file plainly, do not vanish -------------

set "MISSING="
if not exist "runtime\python.exe"                    set "MISSING=!MISSING! runtime\python.exe"
if not exist "bin\ffmpeg.exe"                        set "MISSING=!MISSING! bin\ffmpeg.exe"
if not exist "bin\llama-cuda\llama-server.exe"       set "MISSING=!MISSING! bin\llama-cuda\llama-server.exe"
if not exist "bin\whisper-cuda\whisper-server.exe"   set "MISSING=!MISSING! bin\whisper-cuda\whisper-server.exe"
if not exist "models\ggml-large-v3-turbo.bin"        set "MISSING=!MISSING! models\ggml-large-v3-turbo.bin"
if not exist "models\silero_vad.onnx"                set "MISSING=!MISSING! models\silero_vad.onnx"
rem The language model weights are not needed when the Model dropdown is set to
rem "Port" -- that sends the notes to a server the user is already running. Ask
rem config.json rather than pattern-matching it: "enabled" appears elsewhere too.
set "EXTERNAL_LLM="
if exist "runtime\python.exe" (
  "%~dp0runtime\python.exe" -c "import json,sys;sys.exit(0 if json.load(open('config.json')).get('llm',{}).get('external',{}).get('enabled') else 1)" 2>nul && set "EXTERNAL_LLM=1"
)
rem At least one language model, not both: High Quality is optional.
if not defined EXTERNAL_LLM (
  set "HAVE_LLM="
  if exist "models\Qwen3.8-27B-UD-IQ3_XXS.gguf" set "HAVE_LLM=1"
  if exist "models\Qwen3.8-27B-UD-IQ4_XS.gguf"  set "HAVE_LLM=1"
  if not defined HAVE_LLM set "MISSING=!MISSING! models\Qwen3.8-27B-UD-IQ3_XXS.gguf"
)

if not "!MISSING!"=="" (
  echo.
  echo   Meeting Transcriber Live cannot start.
  echo.
  echo   These files are missing from the application folder:
  for %%F in (!MISSING!) do echo       %%F
  echo.
  echo   If this is a fresh copy, run DOWNLOAD_MODELS.bat first - it fetches
  echo   the Python runtime, ffmpeg, the models and the inference binaries.
  echo   Otherwise the folder may not have copied completely; copy it again.
  echo.
  pause
  exit /b 1
)

rem ---- find a free port ---------------------------------------------------

set "PORT="
for %%P in (8000 8001 8002 8003 8004 8005) do (
  if not defined PORT (
    netstat -ano -p tcp | findstr /r /c:"LISTENING" | findstr /c:":%%P " >nul 2>&1
    if errorlevel 1 set "PORT=%%P"
  )
)

if not defined PORT (
  echo.
  echo   Meeting Transcriber Live cannot start: ports 8000-8005 are all in use.
  echo   Restart the computer and try again.
  echo.
  pause
  exit /b 1
)

echo.
echo   Meeting Transcriber Live
echo.
echo   Keep this window open while you work.
echo   Closing it stops the application, and ends any meeting being recorded.
echo.
rem A shortcut beside run.bat is the only reliable way to hand a non-technical
rem user a clickable link. Rewritten each launch so it points at the live port.
rem The version is in the URL: a browser that cached an older page would
rem otherwise keep showing it after an update.
set "SHORTCUT=%~dp0Open Meeting Transcriber.url"
> "%SHORTCUT%" echo [InternetShortcut]
rem `call` first: a for /f command that starts with a quoted path has its outer
rem quotes stripped by cmd, which broke this line in the batch app (an empty ?v=).
for /f "delims=" %%V in ('call "%~dp0runtime\python.exe" -c "from server.version import APP_VERSION;print(APP_VERSION)"') do set "APPVER=%%V"
>>"%SHORTCUT%" echo URL=http://127.0.0.1:%PORT%/?v=%APPVER%

echo   [1/3] Program files found.
echo   [2/3] Port %PORT% is free.
echo   [3/3] Starting; the browser opens by itself when it is ready.
echo.
echo   If it does not open, either double-click "Open Meeting Transcriber"
echo   in this folder, or type this address into your browser:
echo.
echo       http://127.0.0.1:%PORT%
echo.

rem Open the browser only once the server actually answers.
start "" /B "%~dp0runtime\python.exe" -m server.open_browser %PORT%

rem --timeout-keep-alive covers the long idle gaps between events on the
rem live stream.
"%~dp0runtime\python.exe" -m uvicorn server.main:app ^
  --host 127.0.0.1 ^
  --port %PORT% ^
  --timeout-keep-alive 300 ^
  --log-level warning

echo.
rem An update stops the server on purpose and opens a new window. Pausing
rem here would leave the old console sitting behind it saying the app had
rem stopped, which reads as a crash. The marker says it was intentional.
if exist "temp\updating.flag" (
  del /q "temp\updating.flag" >nul 2>&1
  echo   Updating - this window will close.
  exit /b 0
)
echo   Meeting Transcriber Live has stopped.
pause
