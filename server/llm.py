"""llama-server lifecycle and chat calls (CLAUDE.md sections 7, 7.1 and 8).

Uses http.client rather than a third-party client: it is in the standard
library, and it exposes the connection object, which is what lets Stop abort a
request already in flight instead of waiting it out.

One class serves both uses. The live meeting launches it from
hardware.live_plan (Low Quality beside whisper, ctx 8192, thinking off); the
full summary from hardware.summary_plan (High Quality, the whole card, ctx
32768). With the Port option it attaches to somebody else's server instead and
launches nothing.
"""

from __future__ import annotations

import json
import math
import re
import socket
import subprocess
import threading
import time
from http.client import HTTPConnection
from pathlib import Path
from typing import Optional

from . import config
from .jobs import Cancelled, JobError, Task, kill_tree

CREATE_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

LLM_FAILED = "The language model could not be started."
LLM_CALL_FAILED = "The language model stopped responding."
EXTERNAL_FAILED = (
    "Nothing answered on the port you chose. Start the language model server "
    "first, or pick High Quality or Low Quality instead."
)

EXTERNAL = "external"

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_OPEN_THINK_RE = re.compile(r"^.*?</think>", re.DOTALL)

# The chat template raises a Jinja exception -- surfacing as HTTP 500 -- for
# any effort outside this set (batch BUILD_NOTES section 5).
VALID_EFFORT = {"xhigh", "medium", "low", "none"}

# How long to wait for a server we did not start: it is either up or it is not.
ATTACH_TIMEOUT_S = 10.0

# Injected in place of the rest of the thinking when --reasoning-budget runs
# out, so the model stops reasoning and starts writing.
BUDGET_MESSAGE = (
    "\n\nThat is enough planning. Write the finished document now, in full, "
    "using the structure asked for.\n"
)


def strip_thinking(text: str) -> str:
    """Remove <think> blocks. Applied even when thinking is disabled."""
    text = _THINK_RE.sub("", text)
    if "</think>" in text:            # unbalanced open tag, truncated stream
        text = _OPEN_THINK_RE.sub("", text)
    return text.strip()


def load_prompt(name: str) -> str:
    """Load a prompt from prompts\\. Filled by str.replace, never str.format."""
    path = config.PROMPTS / ("%s.txt" % name)
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise JobError("A prompt file is missing from the application folder.",
                       "cannot read %s: %s" % (path, exc)) from exc


def fill(template: str, **values: str) -> str:
    for key, value in values.items():
        template = template.replace("{{%s}}" % key.upper(), str(value))
    return template


def _ctx_from_props(props: dict) -> int:
    """The context size a running llama-server reports. 0 if it will not say."""
    gen = props.get("default_generation_settings")
    for candidate in (gen, (gen or {}).get("params"), props):
        if isinstance(candidate, dict):
            for key in ("n_ctx", "ctx_size"):
                try:
                    value = int(candidate.get(key) or 0)
                except (TypeError, ValueError):
                    continue
                if value > 0:
                    return value
    return 0


class LlamaServer:
    """Owns (or attaches to) one llama-server.

    `plan` is hardware.live_plan / summary_plan. `external` attaches to the
    Port option's server, which is never started or killed by us.
    """

    def __init__(self, task: Task, cfg: dict, plan: Optional[dict] = None,
                 external: bool = False):
        self.task = task
        self.cfg = cfg
        self.llm = cfg["llm"]
        self.plan = plan or {}
        self.external = external
        _saved, ext_port = config.external_llm(cfg)
        self.port = ext_port if external else int(self.llm.get("port", 8080))
        self.proc: Optional[subprocess.Popen] = None
        self._remote_ctx = 0
        self._conn: Optional[HTTPConnection] = None
        self._conn_lock = threading.Lock()
        self._log_handle = None
        self.key = EXTERNAL if external else self.plan.get("key", "")

    # -- lifecycle -------------------------------------------------------

    def command(self) -> list[str]:
        p = self.plan
        cmd = [
            str(config.LLAMA_SERVER),
            "-m", str(p["path"]),
            "--ctx-size", str(int(p["ctx_size"])),
            "--flash-attn", "on",
            "--cache-type-k", str(self.llm.get("cache_type_k", "q8_0")),
            "--cache-type-v", str(self.llm.get("cache_type_v", "q8_0")),
            "--jinja",
            "--batch-size", "512", "--ubatch-size", "512",
            "--parallel", "1",
            "--host", "127.0.0.1", "--port", str(self.port),
            "--no-webui",
            # No prompt cache in system RAM. This build defaults to 8 GiB of it,
            # and over a 3h43m meeting llama-server grew from 11.6 to 19.0 GB
            # RSS caching prompts that are never sent twice: every block, chunk
            # and reduce prompt is new (BUILD_NOTES section 5).
            "--cache-ram", "0",
        ]
        threads = str(self.llm.get("threads", "auto"))
        if threads and threads != "auto":
            cmd += ["--threads", threads]
        if p.get("layers"):
            cmd += ["--n-gpu-layers", str(p["layers"])]
        if p.get("regex"):
            # Keep attention on the GPU, push the upper blocks' FFN tensors to
            # system RAM: 2.4x faster than whole-layer fitting at equal VRAM.
            cmd += ["--override-tensor", p["regex"]]
        if p.get("spec"):
            cmd += ["--spec-type", p["spec"]]
        if int(p.get("budget") or 0) > 0:
            cmd += ["--reasoning-budget", str(int(p["budget"]))]
            cmd += ["--reasoning-budget-message", BUDGET_MESSAGE]
        return cmd

    def start(self) -> None:
        if self.external:
            self.attach()
            return
        model = Path(self.plan["path"])
        if not model.exists():
            raise JobError(LLM_FAILED, "model file missing: %s" % model.name)
        p = self.plan
        placement = ("%d FFN blocks to system RAM" % (p["regex"].count("|") + 1)
                     if p.get("regex") else "fully on the GPU")
        self.task.log("llm: %s, ctx %d, %s%s" % (model.name, p["ctx_size"], placement,
                                                 ", MTP" if p.get("spec") else ""))
        # stdout to a file, never a pipe: a full pipe blocks the server. The log
        # holds timings and slot state only, no prompt text (BUILD_NOTES 2).
        self._log_handle = open(config.TEMP / "llama-server.log", "w",
                                encoding="utf-8", errors="replace")
        try:
            self.proc = subprocess.Popen(
                self.command(), stdout=self._log_handle, stderr=subprocess.STDOUT,
                env=config.child_env(), cwd=str(config.LLAMA_SERVER.parent),
                creationflags=CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW,
            )
        except OSError as exc:
            raise JobError(LLM_FAILED, "could not spawn llama-server: %s" % exc) from exc
        self.task.register_proc(self.proc)

        timeout = float(self.llm.get("startup_timeout_s", 180))
        started = time.time()
        while time.time() - started < timeout:
            self.task.check_cancelled()
            if self.proc.poll() is not None:
                raise JobError(LLM_FAILED, "llama-server exited during load\n" + self._log_tail())
            if self._healthy():
                self.task.log("llm: ready after %.0fs" % (time.time() - started))
                return
            time.sleep(0.5)
        raise JobError(LLM_FAILED, "llama-server not ready within %.0fs\n%s"
                       % (timeout, self._log_tail()))

    def attach(self) -> None:
        """Use the llama-server already running on the Port option's port."""
        deadline = time.time() + ATTACH_TIMEOUT_S
        while time.time() < deadline:
            self.task.check_cancelled()
            if self._healthy():
                props = self._props()
                self._remote_ctx = _ctx_from_props(props)
                name = str(props.get("model_path") or props.get("model_alias") or "")
                self.task.log("llm: using the server on port %d (%s, %s context)"
                              % (self.port, Path(name).name or "model unknown",
                                 self._remote_ctx or "unknown"))
                return
            time.sleep(0.5)
        raise JobError(EXTERNAL_FAILED,
                       "nothing healthy on 127.0.0.1:%d after %.0fs" % (self.port, ATTACH_TIMEOUT_S))

    @property
    def ctx_size(self) -> int:
        """The context window actually in force, which on Port is not ours."""
        if self.external:
            return self._remote_ctx or int(self.llm.get("ctx_size", 8192))
        return int(self.plan.get("ctx_size", 8192))

    def alive(self) -> bool:
        if self.external:
            return self._healthy()
        return self.proc is not None and self.proc.poll() is None

    def _props(self) -> dict:
        try:
            conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
            conn.request("GET", "/props")
            data = json.loads(conn.getresponse().read())
            conn.close()
            return data if isinstance(data, dict) else {}
        except Exception:  # noqa: BLE001 - /health without /props is still usable
            return {}

    def _log_tail(self, lines: int = 40) -> str:
        try:
            with open(config.TEMP / "llama-server.log", "r", encoding="utf-8",
                      errors="replace") as fh:
                return "".join(fh.readlines()[-lines:])
        except OSError:
            return ""

    def _healthy(self) -> bool:
        try:
            conn = HTTPConnection("127.0.0.1", self.port, timeout=3)
            conn.request("GET", "/health")
            resp = conn.getresponse()
            resp.read()
            conn.close()
            return resp.status == 200
        except Exception:  # noqa: BLE001 - not up yet is the normal case
            return False

    def stop(self) -> None:
        """Shut our server down and release its VRAM. Never somebody else's."""
        self.abort()
        if self.external:
            return
        proc, self.proc = self.proc, None
        if proc is None:
            return
        self.task.unregister_proc(proc)
        kill_tree(proc)
        try:
            proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            pass
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            except Exception:  # noqa: BLE001
                pass
        self.task.log("llm: server stopped")

    def abort(self) -> None:
        """Drop any in-flight request so Stop or Cancel does not wait it out."""
        with self._conn_lock:
            conn, self._conn = self._conn, None
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass

    # -- requests --------------------------------------------------------

    def _post(self, path: str, payload: dict, timeout: float) -> dict:
        body = json.dumps(payload).encode("utf-8")
        conn = HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        with self._conn_lock:
            self._conn = conn
        try:
            conn.request("POST", path, body=body, headers={
                "Content-Type": "application/json", "Content-Length": str(len(body))})
            resp = conn.getresponse()
            raw = resp.read()
            if resp.status != 200:
                raise RuntimeError("HTTP %d from %s" % (resp.status, path))
            return json.loads(raw)
        finally:
            with self._conn_lock:
                if self._conn is conn:
                    self._conn = None
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass

    def token_count(self, text: str) -> int:
        """Exact token count via /tokenize, never a characters-per-token guess."""
        self.task.check_cancelled()
        data = self._post("/tokenize", {"content": text}, timeout=300)
        return len(data.get("tokens", []))

    def _stream(self, payload: dict, idle_timeout: float, on_token) -> tuple[str, int, str]:
        """POST a streaming completion, returning (content, tokens, finish_reason).

        Streaming buys an *idle* timeout -- the socket read only blocks between
        tokens -- so a slow call runs as long as it needs and a dead one is
        caught quickly.
        """
        body = json.dumps({**payload, "stream": True}).encode("utf-8")
        conn = HTTPConnection("127.0.0.1", self.port, timeout=idle_timeout)
        with self._conn_lock:
            self._conn = conn
        content: list[str] = []
        tokens = 0
        finish = ""
        try:
            conn.request("POST", "/v1/chat/completions", body=body, headers={
                "Content-Type": "application/json", "Content-Length": str(len(body))})
            resp = conn.getresponse()
            if resp.status != 200:
                resp.read()
                raise RuntimeError("HTTP %d from the language model" % resp.status)
            while True:
                line = resp.readline()
                if not line:
                    break
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                try:
                    event = json.loads(data)
                except ValueError:
                    continue
                choices = event.get("choices") or [{}]
                # "length" with nothing in content is the model spending its
                # whole allowance thinking; the retry below changes the question.
                finish = choices[0].get("finish_reason") or finish
                delta = choices[0].get("delta") or {}
                if delta.get("content"):
                    content.append(delta["content"])
                    tokens += 1
                    on_token(False)
                if delta.get("reasoning_content"):
                    on_token(True)
        finally:
            with self._conn_lock:
                if self._conn is conn:
                    self._conn = None
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass
        return "".join(content), tokens, finish

    def chat(self, prompt: str, max_tokens: int, thinking: bool, effort: str = "medium",
             progress=None, expect_seconds: float = 0.0, attempts: int = 3) -> str:
        """One chat completion, retried with backoff.

        `progress(fraction)` is called inside the call: the greater of tokens
        seen against max_tokens and 1 - e^(-elapsed/expected), so the bar can
        neither stall nor overshoot however wrong the expectation is.
        """
        if thinking:
            sampling = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                        "presence_penalty": 0.0, "repeat_penalty": 1.0}
            kwargs = {"enable_thinking": True,
                      "reasoning_effort": effort if effort in VALID_EFFORT else "medium"}
        else:
            sampling = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
                        "presence_penalty": 1.5, "repeat_penalty": 1.0}
            kwargs = {"enable_thinking": False}
        payload = {"messages": [{"role": "user", "content": prompt}],
                   "max_tokens": int(max_tokens), "chat_template_kwargs": kwargs, **sampling}
        idle = float(self.llm.get("idle_timeout_s", 300))

        last = ""
        for attempt in range(attempts):
            self.task.check_cancelled()
            started = time.time()
            seen = [0]

            def on_token(_is_thinking: bool) -> None:
                seen[0] += 1
                if progress is not None and seen[0] % 20 == 0:
                    frac = seen[0] / max(float(max_tokens), 1.0)
                    if expect_seconds > 0:
                        frac = max(frac, 1.0 - math.exp(-(time.time() - started) / expect_seconds))
                    progress(min(0.99, frac))

            try:
                text, tokens, finish = self._stream(payload, idle, on_token)
                elapsed = time.time() - started
                self.task.log("llm: %d tokens in %.1fs (%.1f tok/s), %d incl. thinking"
                              % (tokens, elapsed, tokens / max(elapsed, 1e-6), seen[0]))
                text = strip_thinking(text)
                if text:
                    return text
                last = "empty completion (finish_reason=%s)" % (finish or "unknown")
                if finish == "length" and payload["chat_template_kwargs"].get("enable_thinking"):
                    # Thought for the whole allowance and never wrote. Asking
                    # again identically buys the same nothing; drop thinking.
                    self.task.log("llm: whole allowance spent thinking; retrying without")
                    payload["chat_template_kwargs"] = {"enable_thinking": False}
                    payload.update({"temperature": 0.7, "top_p": 0.8,
                                    "presence_penalty": 1.5, "min_p": 0.0})
            except Cancelled:
                raise
            except Exception as exc:  # noqa: BLE001
                last = str(exc)
                if self.task.cancelled:
                    raise Cancelled() from exc
                if isinstance(exc, (TimeoutError, socket.timeout)):
                    raise JobError(LLM_CALL_FAILED,
                                   "no output for %.0fs; giving up" % idle) from exc
            if attempt < attempts - 1:
                delay = 2.0 * (attempt + 1)
                self.task.log("llm: call failed (%s); retrying in %.0fs" % (last, delay))
                time.sleep(delay)
        raise JobError(LLM_CALL_FAILED, "%d attempts failed; last error: %s" % (attempts, last))
