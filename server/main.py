"""FastAPI app: the page, the live meeting, history, full summary, About.

Bound to 127.0.0.1 only, never 0.0.0.0 (CLAUDE.md section 17).
"""

from __future__ import annotations

import asyncio
import atexit
import json
import os
import signal
import subprocess
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse, PlainTextResponse,
                               StreamingResponse)

from . import capture, config, hardware, history, jobs, llm, summary, updater, version
from .session import app as live, bus, engines, monitor

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _prime_hardware() -> None:
    try:
        hardware.detect_gpu()
    except Exception as exc:  # noqa: BLE001 - detection must never stop startup
        jobs.log_exception(exc)


def _monitor_watchdog() -> None:
    """The pre-Start level meter: release the mic when no page is listening, and
    pick it up again when it comes back after being unplugged (section 4)."""
    while True:
        time.sleep(5)
        try:
            if bus.listeners == 0:
                monitor.stop()
            elif live.state == "idle" and monitor.capture is None and live.device:
                target = capture.normalise(live.device)
                found = [d for d in capture.list_devices() if capture.normalise(d) == target]
                if len(found) == 1:
                    monitor.start(found[0])
        except Exception as exc:  # noqa: BLE001 - the meter is never worth a crash
            jobs.log_exception(exc)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    config.TEMP.mkdir(parents=True, exist_ok=True)
    config.OUTPUT.mkdir(parents=True, exist_ok=True)
    jobs.clean_temp(keep_log=False)
    jobs.install_kill_on_close()
    try:
        cfg = config.load_config()
        killed = jobs.kill_stale([int(cfg["asr"]["port"]), int(cfg["asr"]["refine_port"]),
                                  int(cfg["llm"]["port"])])
        if killed:
            jobs.log("startup: killed %d stale server(s) from a previous run" % killed)
        fixed = history.recover_interrupted()
        if fixed:
            jobs.log("startup: %d interrupted meeting(s) recovered" % fixed)
    except Exception as exc:  # noqa: BLE001
        jobs.log_exception(exc)
    threading.Thread(target=_prime_hardware, name="gpu-probe", daemon=True).start()
    threading.Thread(target=_monitor_watchdog, name="monitor-watchdog", daemon=True).start()
    updater.mark_running()
    try:
        yield
    finally:
        updater.clear_running()
        live.shutdown()
        jobs.shutdown_all()


app = FastAPI(title=version.APP_NAME, docs_url=None, redoc_url=None, lifespan=lifespan)

# Revalidate the frontend on every load: the updater replaces these files under
# a tab that would otherwise keep running the old ones.
NO_CACHE = {"Cache-Control": "no-cache, must-revalidate"}


def _err(message: str, status: int = 400) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


async def _body(request: Request) -> dict:
    try:
        data = await request.json()
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


# ---------------------------------------------------------------------------
# the page
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    html = (config.WEB / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html.replace("__APP_VERSION__", version.APP_VERSION), headers=NO_CACHE)


@app.get("/app.js")
async def app_js() -> FileResponse:
    return FileResponse(config.WEB / "app.js", media_type="application/javascript; charset=utf-8",
                        headers=NO_CACHE)


@app.get("/style.css")
async def style_css() -> FileResponse:
    return FileResponse(config.WEB / "style.css", media_type="text/css; charset=utf-8",
                        headers=NO_CACHE)


@app.get("/api/health")
async def health() -> dict:
    missing = [str(p.relative_to(config.ROOT)) for p in config.missing_files()]
    return {"ok": not missing, "missing": missing}


# ---------------------------------------------------------------------------
# state and events
# ---------------------------------------------------------------------------

def _summary_view():
    job = live.summary
    return job.view() if job is not None else None


@app.get("/api/session")
async def session_state() -> dict:
    s = live.session
    return {"status": live.status(), "session": s.snapshot() if s is not None else None,
            "summary": _summary_view()}


@app.get("/api/events")
async def events():
    q = bus.subscribe()

    async def stream():
        yield _sse({"type": "status", **live.status()})
        try:
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                yield _sse(event)
        finally:
            bus.unsubscribe(q)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _sse(event: dict) -> str:
    return "data: %s\n\n" % json.dumps(event)


# ---------------------------------------------------------------------------
# microphone and model
# ---------------------------------------------------------------------------

@app.get("/api/devices")
async def devices() -> dict:
    cfg = config.load_config()
    name, how, found = await asyncio.to_thread(live.resolve_device, cfg)
    remembered = str(cfg["capture"].get("device") or "")
    message = ""
    if how == "none":
        message = "No microphone found. Plug one in."
    elif how in ("default", "first") and remembered:
        message = ('Couldn\'t find "%s". Using "%s" — change it above if that\'s wrong.'
                   % (remembered, name))
    return {"devices": found, "selected": name, "how": how, "message": message,
            "replay": bool(cfg["capture"].get("replay_file"))}


@app.post("/api/devices/choice")
async def device_choice(request: Request):
    name = str((await _body(request)).get("name") or "")
    if not name:
        return _err("No microphone chosen.")
    try:
        config.save_device(name)
    except RuntimeError as exc:
        return _err(str(exc))
    live.device, live.device_how = name, "exact"
    if live.state == "idle":
        await asyncio.to_thread(monitor.start, name)
    return {"ok": True}


@app.post("/api/monitor")
async def monitor_start() -> dict:
    """The level meter before Start. Only while nothing is live."""
    if live.state == "idle" and live.device:
        await asyncio.to_thread(monitor.start, live.device)
    return {"ok": True}


@app.get("/api/models")
async def models() -> dict:
    cfg = config.load_config()
    d = await asyncio.to_thread(hardware.describe, cfg)
    if live.choice and not d["external"]["enabled"]:
        d["selected"] = live.choice
    return d


@app.post("/api/models/choice")
async def model_choice(request: Request):
    body = await _body(request)
    key = str(body.get("key") or "")
    try:
        if key == llm.EXTERNAL:
            port = int(body.get("port") or 0)
            if not 1 <= port <= 65535:
                return _err("That isn't a valid port number.")
            config.save_llm_choice(True, port)
        elif key in hardware.BY_KEY:
            config.save_llm_choice(False)
        else:
            return _err("Unknown model.")
    except (RuntimeError, ValueError) as exc:
        return _err(str(exc))
    live.choice = key
    return {"ok": True}


# ---------------------------------------------------------------------------
# start / stop
# ---------------------------------------------------------------------------

@app.post("/api/start")
async def start():
    if live.state != "idle":
        return _err("Already running.", 409)
    threading.Thread(target=live.start, name="start", daemon=True).start()
    return {"ok": True}


@app.post("/api/stop")
async def stop():
    if live.state != "live":
        return _err("Nothing is listening.", 409)
    threading.Thread(target=live.stop, name="stop", daemon=True).start()
    return {"ok": True}


# ---------------------------------------------------------------------------
# history
# ---------------------------------------------------------------------------

def _live_id() -> str:
    s = live.session
    return s.meeting.id if s is not None and s.state in ("live", "stopping") else ""


@app.get("/api/history")
async def history_list() -> dict:
    live_id = _live_id()
    rows = await asyncio.to_thread(history.list_meetings, live_id)
    live_row = None
    if live_id:
        s = live.session
        live_row = {**s.meeting.summary_row(), "live": True,
                    "duration_s": time.time() - s.started}
    return {"live": live_row, "meetings": rows}


@app.get("/api/history/{folder_id}")
async def history_open(folder_id: str):
    s = live.session
    if s is not None and s.meeting.id == folder_id:
        return {**s.snapshot(), "live": s.state in ("live", "stopping")}
    m = history.Meeting.open(folder_id)
    if m is None:
        return _err("That meeting is no longer in the history.", 404)
    return {**await asyncio.to_thread(m.full), "live": False}


@app.post("/api/history/{folder_id}/rename")
async def history_rename(folder_id: str, request: Request):
    name = str((await _body(request)).get("name") or "")
    s = live.session
    ok = history.rename(folder_id, name)
    if ok and s is not None and s.meeting.id == folder_id:
        s.meeting.meta["name"] = " ".join(name.split())[:120]
    if not ok:
        return _err("That name couldn't be saved.")
    bus.emit({"type": "history"})
    return {"ok": True}


@app.delete("/api/history/{folder_id}")
async def history_delete(folder_id: str):
    if folder_id == _live_id():
        return _err("The live meeting can't be deleted.", 409)
    job = live.summary
    if job is not None and job.state == "running" and job.meeting_id == folder_id:
        return _err("That meeting is being summarised.", 409)
    if not await asyncio.to_thread(history.delete, folder_id):
        return _err("That meeting couldn't be deleted.")
    bus.emit({"type": "history"})
    return {"ok": True}


# ---------------------------------------------------------------------------
# Generate Full Summary (section 7.1)
# ---------------------------------------------------------------------------

_dialog_lock = threading.Lock()


@app.post("/api/summary/dialog")
async def summary_dialog(request: Request):
    """Step 2: the Save As dialog for the copy. Cancel here changes nothing."""
    folder_id = str((await _body(request)).get("id") or "")
    m = history.Meeting.open(folder_id)
    if m is None:
        return _err("That meeting is no longer in the history.", 404)
    if not _dialog_lock.acquire(blocking=False):
        return _err("A Save dialog is already open.", 409)
    try:
        path = await asyncio.to_thread(
            summary.save_dialog, summary.default_copy_name(m.meta.get("name", m.id)))
    finally:
        _dialog_lock.release()
    return {"path": path}


@app.post("/api/summary/start")
async def summary_start(request: Request):
    """Step 3: from here it is final."""
    body = await _body(request)
    folder_id = str(body.get("id") or "")
    copy_path = str(body.get("path") or "")
    if not copy_path:
        return _err("No location was chosen.")
    if live.summary is not None and live.summary.state == "running":
        return _err("A summary is already being written.", 409)
    is_live = folder_id == _live_id()
    if not is_live and live.state != "idle":
        return _err("Stop the live meeting first.", 409)
    m = history.Meeting.open(folder_id)
    if m is None:
        return _err("That meeting is no longer in the history.", 404)
    job = summary.SummaryJob(folder_id, copy_path, bus.emit)
    live.summary = job
    threading.Thread(target=_run_summary, args=(job, is_live), name="summary",
                     daemon=True).start()
    return {"ok": True}


def _run_summary(job: summary.SummaryJob, is_live: bool) -> None:
    cfg = config.load_config()
    external = live.choice == llm.EXTERNAL or (not live.choice and config.external_llm(cfg)[0])
    try:
        if is_live:
            job.progress("load", 0.0, "Finishing the transcript…")
            live.stop(for_summary=True)
        live._set("summarizing", "Writing the full summary…")
        # Give the summary model the whole card (section 7.1).
        engines.stop_all()
        text = job.run(cfg, external)
        m = history.Meeting.open(job.meeting_id)
        if m is None:
            raise jobs.JobError("That meeting is no longer in the history.")
        m.save_summary(text)
        m.update_meta(summary_copy=job.copy_path)
        job.copy_error = summary.write_copy(job.copy_path, text)
        job.state = "done"
        job.progress("final", 1.0, "Summary saved")
        bus.emit({"type": "summary_done", **job.view(), "summary": text})
    except jobs.Cancelled:
        job.state = "cancelled"
        bus.emit({"type": "summary_done", **job.view()})
    except jobs.JobError as exc:
        jobs.log_exception(exc)
        if exc.detail:
            jobs.log(exc.detail)
        job.state = "failed"
        job.message = ("The summary couldn't be written. The meeting's transcript is safe "
                       "in the history." if exc.message.startswith("The language model")
                       else exc.message)
        bus.emit({"type": "summary_done", **job.view()})
    except Exception as exc:  # noqa: BLE001
        jobs.log_exception(exc)
        job.state = "failed"
        job.message = ("The summary couldn't be written. The meeting's transcript is safe "
                       "in the history.")
        bus.emit({"type": "summary_done", **job.view()})
    finally:
        live._set("idle", "")
        bus.emit({"type": "history"})


@app.post("/api/summary/cancel")
async def summary_cancel() -> dict:
    job = live.summary
    if job is not None and job.state == "running":
        job.cancel()
    return {"ok": True}


@app.post("/api/open")
async def open_path(request: Request):
    """Open the summary copy, or its folder, in Explorer."""
    body = await _body(request)
    m = history.Meeting.open(str(body.get("id") or ""))
    if m is None:
        return _err("That meeting is no longer in the history.", 404)
    what = str(body.get("what") or "file")
    copy = str(m.meta.get("summary_copy") or "")
    target = Path(copy) if copy and Path(copy).exists() else m.folder / history.SUMMARY
    try:
        if what == "folder":
            subprocess.Popen(["explorer", "/select,", str(target)])
        else:
            os.startfile(str(target))      # noqa: S606 - the user's own file
    except OSError:
        return _err("Couldn't open it.", 500)
    return {"ok": True}


# ---------------------------------------------------------------------------
# About, updates, diagnostics
# ---------------------------------------------------------------------------

@app.get("/api/about")
async def about() -> dict:
    return {"name": version.APP_NAME, "author": version.APP_AUTHOR,
            "version": version.APP_VERSION, "repo_url": version.REPO_URL}


@app.post("/api/update/check")
async def update_check() -> dict:
    """The only outbound request the app ever makes, and only on a button press."""
    return await asyncio.to_thread(updater.check)


@app.post("/api/update/install")
async def update_install(request: Request):
    if live.state != "idle":
        return _err("A meeting is still running. Stop it first.", 409)
    if updater.state().get("phase") not in ("idle", "error"):
        return {"ok": True}
    body = await _body(request)
    url = str(body.get("url") or "")
    if not (url.startswith("https://github.com/")
            or url.startswith("https://objects.githubusercontent.com/")):
        return _err("That download location isn't allowed.")
    port = int(request.url.port or config.load_config()["server"]["port"])
    threading.Thread(target=updater.install,
                     args=(url, int(body.get("size_bytes") or 0), str(body.get("tag") or ""),
                           port), name="updater", daemon=True).start()
    return {"ok": True}


@app.get("/api/update/progress")
async def update_progress() -> dict:
    return updater.state()


@app.post("/api/update/cancel")
async def update_cancel() -> dict:
    updater.cancel()
    return {"ok": True}


@app.get("/api/diagnostics")
async def diagnostics() -> PlainTextResponse:
    return PlainTextResponse(jobs.read_job_log())


# ---------------------------------------------------------------------------
# lifecycle
# ---------------------------------------------------------------------------

def _emergency_cleanup(*_args) -> None:
    try:
        live.shutdown()
    except Exception:  # noqa: BLE001
        pass
    jobs.shutdown_all()


atexit.register(_emergency_cleanup)

# SIGINT is deliberately NOT handled: uvicorn uses it for a clean shutdown,
# which reaches the same cleanup through lifespan.
for _sig in ("SIGTERM", "SIGBREAK"):
    if hasattr(signal, _sig):
        try:
            signal.signal(getattr(signal, _sig), _emergency_cleanup)
        except (ValueError, OSError):
            pass
