/* Meeting Transcriber Live front end. No framework, no bundler, no external requests. */
(function () {
  "use strict";

  var $ = function (id) { return document.getElementById(id); };

  var S = {
    status: { state: "idle" },
    view: null,          // the meeting on screen: live, or one from the history
    liveId: "",
    elapsedBase: 0,      // Date.now() at the live meeting's session time 0
    timer: null,
    es: null,
    summary: null,       // the running summary job, if any
    replay: false,
    device: "",
    deviceListAt: 0,
    deviceMessage: "",
    lastModel: ""
  };

  /* ---------------------------------------------------------------- helpers */

  function hms(sec) {
    sec = Math.max(0, Math.floor(sec || 0));
    var h = Math.floor(sec / 3600), m = Math.floor(sec % 3600 / 60), s = sec % 60;
    return (h < 10 ? "0" : "") + h + ":" + (m < 10 ? "0" : "") + m + ":" + (s < 10 ? "0" : "") + s;
  }

  // Real clock time, 12-hour: "2:34:05 PM", or "2:34 PM" without seconds.
  function clock12(epochSec, withSeconds) {
    var d = new Date((epochSec || 0) * 1000);
    var h = d.getHours(), m = d.getMinutes(), s = d.getSeconds();
    var half = h < 12 ? "AM" : "PM";
    h = h % 12 || 12;
    return h + ":" + (m < 10 ? "0" : "") + m +
      (withSeconds ? ":" + (s < 10 ? "0" : "") + s : "") + " " + half;
  }

  function humanDuration(sec) {
    var m = Math.floor((sec || 0) / 60);
    if (m < 1) return "under a minute";
    if (m < 60) return m + "m";
    return Math.floor(m / 60) + "h " + (m % 60 < 10 ? "0" : "") + (m % 60) + "m";
  }

  function humanDate(epoch) {
    var d = new Date(epoch * 1000);
    var months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    return d.getDate() + " " + months[d.getMonth()] + " " + d.getFullYear();
  }

  function api(method, url, body) {
    var opts = { method: method, headers: {} };
    if (body !== undefined) {
      opts.headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(body);
    }
    return fetch(url, opts).then(function (r) {
      return r.json().catch(function () { return {}; }).then(function (b) {
        if (!r.ok) throw new Error(b.error || "Something went wrong.");
        return b;
      });
    });
  }

  function banner(text, kind) {
    var el = $("banner");
    el.textContent = text || "";
    el.className = "banner" + (kind === "error" ? " error" : "");
    el.hidden = !text;
  }

  // A passing message: shown, then cleared unless something else replaced it.
  var noticeTimer = null;
  function notice(text) {
    banner(text);
    if (noticeTimer) clearTimeout(noticeTimer);
    noticeTimer = setTimeout(function () {
      if ($("banner").textContent === text) banner("");
    }, 10000);
  }

  function words(lines) {
    var n = 0;
    (lines || []).forEach(function (l) { n += (l.text || "").split(/\s+/).filter(Boolean).length; });
    return n;
  }

  // "At the bottom" generously: anywhere within about four lines of the end
  // keeps following the meeting.
  function atBottom(el) { return el.scrollHeight - el.scrollTop - el.clientHeight < 140; }
  function toBottom(el) { el.scrollTop = el.scrollHeight; }

  /* ---------------------------------------------------------------- boot */

  var bootStarted = Date.now();

  function waitForServer() {
    fetch("/api/health", { cache: "no-store" })
      .then(function (r) { return r.json(); })
      .then(function (h) {
        if (!h.ok) {
          $("boot-msg").textContent = "Some files are missing from the application folder. " +
            "If this is a fresh copy, run DOWNLOAD_MODELS.bat first.";
          var ul = $("boot-missing");
          ul.innerHTML = "";
          (h.missing || []).forEach(function (m) {
            var li = document.createElement("li"); li.textContent = m; ul.appendChild(li);
          });
          ul.hidden = false;
          document.querySelector("#boot .spinner").hidden = true;
          return;
        }
        $("boot").hidden = true;
        init();
      })
      .catch(function () {
        var s = (Date.now() - bootStarted) / 1000;
        $("boot-msg").textContent = s < 15 ? "Contacting the application…"
          : "Still starting. If this goes on, check the black console window is open.";
        setTimeout(waitForServer, 500);
      });
  }

  function init() {
    setTabs(false);
    loadDevices(true);
    loadModels();
    loadSession();
    loadHistory();
    listen();
  }

  /* ---------------------------------------------------------------- events */

  function listen() {
    if (S.es) S.es.close();
    var es = new EventSource("/api/events");
    S.es = es;
    var opened = false;
    es.onopen = function () {
      if (opened) return;
      opened = true;
      // Reconnected after a server restart or a sleep: resync everything.
      loadSession();
      loadHistory();
      api("POST", "/api/monitor").catch(function () {});
    };
    es.onmessage = function (e) {
      var ev;
      try { ev = JSON.parse(e.data); } catch (x) { return; }
      handle(ev);
    };
    es.onerror = function () {
      es.close();
      S.es = null;
      setTimeout(listen, 1500);
    };
  }

  function handle(ev) {
    switch (ev.type) {
      case "status": applyStatus(ev); break;
      case "level": meter(ev.db); break;
      case "session":
        // A meeting that has just started always takes the screen: pressing
        // Start means "show me this one", whatever was open before.
        if (!S.view || S.view.live || S.view.id === ev.session.id ||
            ev.session.state === "live") showSession(ev.session);
        break;
      case "utterance": onUtterance(ev.line); break;
      case "block_lines": onBlockLines(ev); break;
      case "block_pending":
      case "block": onBlock(ev.block); break;
      case "history": loadHistory(); break;
      case "summary_progress": onSummaryProgress(ev); break;
      case "summary_done": onSummaryDone(ev); break;
      case "notice": notice(ev.message); break;
    }
  }

  function loadSession() {
    api("GET", "/api/session").then(function (d) {
      applyStatus(d.status);
      if (d.summary && d.summary.state === "running") onSummaryProgress(d.summary);
      if (d.session && (!S.view || S.view.live || S.view.id === d.session.id)) {
        showSession(d.session);
      }
    }).catch(function () {});
  }

  /* ---------------------------------------------------------------- status */

  function applyStatus(st) {
    var prev = S.status.state;
    S.status = st;
    S.liveId = st.live_id || "";
    var state = st.state;
    var btn = $("startstop");
    btn.classList.toggle("stop", state === "live");
    btn.textContent = state === "live" ? "Stop" : state === "loading" ? "Loading…"
      : state === "stopping" ? "Stopping…" : "Start";
    btn.disabled = !(state === "live" || (state === "idle" && (S.device || S.replay)));
    $("mic").disabled = state !== "idle" || S.replay;
    $("model").disabled = state !== "idle";
    $("port").disabled = state !== "idle";

    var dot = $("status-dot");
    dot.className = "dot" + (state === "live" && !st.mic_lost ? " live"
      : state === "idle" || st.mic_lost ? "" : " busy");
    var text = state === "live" && st.mic_lost ? "Microphone disconnected"
      : state === "live" ? "Listening" : state === "loading" ? (st.message || "Loading models…")
      : state === "stopping" ? (st.message || "Finishing up…")
      : state === "summarizing" ? "Writing the full summary" : "Stopped";
    $("status-text").textContent = text;

    $("controls").hidden = state === "summarizing";
    $("summary-panel").hidden = state !== "summarizing";

    if (state === "idle" && st.message) banner(st.message, "error");
    else if (state === "live" && st.mic_lost) banner(st.warning, "error");
    else if (state === "live" && st.warning) banner(st.warning);
    else if (state === "live" && prev !== "live") banner(S.deviceMessage || "");
    else if (state === "loading") banner("");

    if (state === "live") startClock(); else stopClock();
    $("livebar").hidden = !(S.liveId && S.view && S.view.id !== S.liveId);
    updateFooter();
    updateNextNotes();
    if (prev !== state && state === "idle") api("POST", "/api/monitor").catch(function () {});
  }

  function startClock() {
    if (S.timer) return;
    S.timer = setInterval(tick, 1000);
    tick();
  }
  function stopClock() {
    if (S.timer) { clearInterval(S.timer); S.timer = null; }
    $("elapsed").textContent = "";
  }
  function tick() {
    if (!S.elapsedBase) return;
    $("elapsed").textContent = hms((Date.now() - S.elapsedBase) / 1000);
  }

  function meter(db) {
    var pct = Math.max(0, Math.min(100, (db + 60) / 50 * 100));
    var el = $("meter-fill");
    el.style.width = pct + "%";
    el.style.opacity = String(0.35 + pct / 160);
  }

  $("startstop").addEventListener("click", function () {
    var state = S.status.state;
    $("startstop").disabled = true;
    if (state === "live") {
      api("POST", "/api/stop").catch(function (e) { banner(e.message, "error"); });
    } else if (state === "idle") {
      banner("");
      api("POST", "/api/start").catch(function (e) { banner(e.message, "error"); });
    }
  });

  /* ---------------------------------------------------------------- mic */

  function loadDevices(force) {
    if (!force && Date.now() - S.deviceListAt < 3000) return;
    S.deviceListAt = Date.now();
    api("GET", "/api/devices").then(function (d) {
      var sel = $("mic");
      S.replay = !!d.replay;
      sel.innerHTML = "";
      if (d.replay) {
        sel.appendChild(new Option("Replay file (development)", ""));
      } else if (!d.devices.length) {
        sel.appendChild(new Option("No microphone found", ""));
      }
      if (!d.replay) {
        d.devices.forEach(function (name) {
          var o = new Option(name, name);
          if (name === d.selected) o.selected = true;
          sel.appendChild(o);
        });
      }
      S.device = d.selected || "";
      S.deviceMessage = d.message || "";
      if (d.message && S.status.state === "idle") banner(d.message, d.how === "none" ? "error" : "");
      applyStatus(S.status);
    }).catch(function () {});
  }

  // Re-enumerate whenever the dropdown is opened: the mic may have been
  // plugged in, or moved to another USB port, since the page loaded.
  $("mic").addEventListener("mousedown", function () { loadDevices(false); });
  $("mic").addEventListener("focus", function () { loadDevices(false); });
  $("mic").addEventListener("change", function () {
    var name = $("mic").value;
    if (!name) return;
    S.device = name;
    S.deviceMessage = "";
    banner("");
    api("POST", "/api/devices/choice", { name: name }).catch(function (e) { banner(e.message, "error"); });
    applyStatus(S.status);
  });

  /* ---------------------------------------------------------------- model */

  function loadModels() {
    api("GET", "/api/models").then(function (d) {
      var sel = $("model");
      sel.innerHTML = "";
      d.models.forEach(function (m) {
        var label = m.label + (m.available ? "" : " (not downloaded)") +
          (m.available && !m.fits ? " (slower here)" : "");
        var o = new Option(label, m.key);
        o.disabled = !m.available;
        sel.appendChild(o);
      });
      sel.appendChild(new Option("Port", "external"));
      sel.value = d.selected;
      $("port").value = d.external.port;
      S.lastModel = d.selected;
      syncPort();
    }).catch(function () {});
  }

  function syncPort() { $("port").hidden = $("model").value !== "external"; }

  function saveModel() {
    var key = $("model").value;
    var body = { key: key };
    if (key === "external") body.port = parseInt($("port").value, 10);
    api("POST", "/api/models/choice", body).catch(function (e) { banner(e.message, "error"); });
  }

  $("model").addEventListener("change", function () { syncPort(); saveModel(); });
  $("port").addEventListener("change", saveModel);

  /* ---------------------------------------------------------------- the view */

  function showSession(snap) {
    var live = snap.state === "live" || snap.state === "stopping";
    if (live) S.elapsedBase = Date.now() - snap.elapsed * 1000;
    showMeeting({
      id: snap.id, name: snap.name, live: live, lines: snap.lines, blocks: snap.blocks,
      summary: snap.summary || "", has_summary: !!snap.has_summary,
      next_notes_at: snap.next_notes_at, block_s: snap.block_s || 300, started: snap.started,
      not_kept: snap.not_kept || ""
    });
  }

  function showMeeting(m) {
    S.view = m;
    $("view-name").textContent = m.live ? "" : m.name;
    renderTranscript();
    renderNotes();
    setTabs(m.has_summary);
    $("summary-view").innerHTML = m.has_summary ? renderMarkdown(m.summary) : "";
    // A finished summary is what someone opening a past meeting came for.
    selectTab(m.has_summary ? "summary" : "transcript");
    $("livebar").hidden = !(S.liveId && m.id !== S.liveId);
    markActiveRow();
    updateFooter();
    updateNextNotes();
  }

  function openMeeting(id) {
    if (S.liveId && id === S.liveId) {
      api("GET", "/api/session").then(function (d) { if (d.session) showSession(d.session); });
      return;
    }
    api("GET", "/api/history/" + encodeURIComponent(id)).then(function (d) {
      if (d.live) { showSession(d); return; }
      showMeeting({ id: d.id, name: d.name, live: false, lines: d.lines, blocks: d.blocks,
                    summary: d.summary, has_summary: d.has_summary, started: d.started });
    }).catch(function (e) { banner(e.message, "error"); loadHistory(); });
  }

  $("livebar").addEventListener("click", function () { openMeeting(S.liveId); });

  /* ---------------------------------------------------------------- transcript */

  function lineEl(l) {
    var p = document.createElement("p");
    p.className = "line" + (l.refined ? "" : " rough");
    p.dataset.id = l.id;
    p.dataset.block = l.block;
    var ts = document.createElement("span");
    ts.className = "ts";
    ts.textContent = clock12(S.view.started + l.t, true);
    p.appendChild(ts);
    p.appendChild(document.createTextNode(l.text));
    return p;
  }

  function renderTranscript() {
    var box = $("transcript");
    var lines = S.view.lines || [];
    box.innerHTML = "";
    if (!lines.length) {
      var ph = document.createElement("p");
      ph.className = "placeholder";
      ph.id = "transcript-empty";
      ph.textContent = S.view.live ? "Listening. What is said will appear here."
        : "Nothing was transcribed in this meeting.";
      box.appendChild(ph);
    }
    var frag = document.createDocumentFragment();
    lines.forEach(function (l) { frag.appendChild(lineEl(l)); });
    box.appendChild(frag);
    if (S.view.live) toBottom(box); else box.scrollTop = 0;
    $("jump-transcript").hidden = true;
  }

  function onUtterance(l) {
    if (!S.view || S.view.id !== S.liveId) {
      if (S.view && S.view.live) S.view.lines.push(l);
      return;
    }
    S.view.lines.push(l);
    var box = $("transcript");
    var stick = atBottom(box);
    var empty = $("transcript-empty");
    if (empty) empty.remove();
    box.appendChild(lineEl(l));
    if (stick) toBottom(box); else $("jump-transcript").hidden = false;
    updateFooter();
  }

  // Refined lines replace a block's live ones. The swap must not move what
  // the reader is looking at: anchor on the first visible line outside the
  // replaced block (section 6.1).
  function onBlockLines(ev) {
    if (!S.view || S.view.id !== S.liveId) return;
    var k = ev.block;
    var kept = S.view.lines.filter(function (l) { return l.block !== k; });
    var before = kept.filter(function (l) { return l.block < k; });
    var after = kept.filter(function (l) { return l.block > k; });
    S.view.lines = before.concat(ev.lines, after);

    var box = $("transcript");
    var stick = atBottom(box);
    var anchor = null, anchorOffset = 0;
    var top = box.scrollTop;
    var els = box.querySelectorAll(".line");
    for (var i = 0; i < els.length; i++) {
      var el = els[i];
      if (el.offsetTop + el.offsetHeight <= top) continue;
      if (parseInt(el.dataset.block, 10) === k) continue;
      anchor = el;
      anchorOffset = el.offsetTop - top;
      break;
    }
    var old = box.querySelectorAll('.line[data-block="' + k + '"]');
    var next = null;
    for (var j = 0; j < els.length; j++) {
      if (parseInt(els[j].dataset.block, 10) > k) { next = els[j]; break; }
    }
    Array.prototype.forEach.call(old, function (el) { el.remove(); });
    var frag = document.createDocumentFragment();
    ev.lines.forEach(function (l) { frag.appendChild(lineEl(l)); });
    if (next) box.insertBefore(frag, next); else box.appendChild(frag);
    if (stick) toBottom(box);
    else if (anchor) box.scrollTop = anchor.offsetTop - anchorOffset;
  }

  $("transcript").addEventListener("scroll", function () {
    if (atBottom($("transcript"))) $("jump-transcript").hidden = true;
  });
  $("jump-transcript").addEventListener("click", function () {
    toBottom($("transcript"));
    $("jump-transcript").hidden = true;
  });

  /* ---------------------------------------------------------------- notes */

  // "2:30 – 2:35 PM": real clock time, not time since Start (operator's choice).
  function clockRange(a, b) {
    var x = clock12(a, false), y = clock12(b, false);
    return x.slice(-2) === y.slice(-2) ? x.slice(0, -3) + " – " + y : x + " – " + y;
  }

  function blockText(b) {
    if (b.status === "pending" || b.status === "refining") return "Tidying up the transcript…";
    if (b.status === "writing") return "Writing notes…";
    return "";
  }

  function blockEl(b) {
    var card = document.createElement("div");
    var waiting = !!blockText(b);
    card.className = "block" + (waiting ? " pending" : "");
    card.dataset.index = b.index;
    var head = document.createElement("div");
    head.className = "block-head";
    var span = document.createElement("span");
    span.textContent = clockRange(S.view.started + b.start_s, S.view.started + b.end_s);
    head.appendChild(span);
    card.appendChild(head);
    var body = document.createElement("div");
    body.className = "block-body";
    if (waiting) body.textContent = blockText(b);
    else body.innerHTML = renderMarkdown(b.notes || "");
    card.appendChild(body);
    return card;
  }

  function renderNotes() {
    var box = $("notes");
    var blocks = (S.view.blocks || []).filter(function (b) { return b.status !== "skipped"; });
    box.innerHTML = "";
    // A session too short to keep says so here, in place of its notes.
    if (S.view.not_kept) {
      blocks = [];
      var gone = document.createElement("p");
      gone.className = "placeholder";
      gone.textContent = S.view.not_kept;
      box.appendChild(gone);
      $("jump-notes").hidden = true;
      return;
    }
    if (!blocks.length) {
      var ph = document.createElement("p");
      ph.className = "placeholder";
      ph.id = "notes-empty";
      ph.textContent = S.view.live ? "A short note appears here after every 5 minutes."
        : "No notes were written for this meeting.";
      box.appendChild(ph);
    }
    blocks.sort(function (a, b) { return a.start_s - b.start_s; });
    blocks.forEach(function (b) { box.appendChild(blockEl(b)); });
    if (S.view.live) toBottom(box); else box.scrollTop = 0;
    $("jump-notes").hidden = true;
  }

  // Appended at the bottom, never removed (section 7).
  function onBlock(b) {
    if (!S.view || S.view.id !== S.liveId) return;
    var list = S.view.blocks || (S.view.blocks = []);
    var found = false;
    for (var i = 0; i < list.length; i++) {
      if (list[i].index === b.index) { list[i] = b; found = true; }
    }
    if (!found) list.push(b);
    var box = $("notes");
    var stick = atBottom(box);
    var empty = $("notes-empty");
    if (empty) empty.remove();
    var existing = box.querySelector('.block[data-index="' + b.index + '"]');
    if (b.status === "skipped") { if (existing) existing.remove(); return; }
    var el = blockEl(b);
    if (existing) existing.replaceWith(el);
    else box.appendChild(el);
    if (stick) toBottom(box); else $("jump-notes").hidden = false;
  }

  $("notes").addEventListener("scroll", function () {
    if (atBottom($("notes"))) $("jump-notes").hidden = true;
  });
  $("jump-notes").addEventListener("click", function () {
    toBottom($("notes"));
    $("jump-notes").hidden = true;
  });

  function updateNextNotes() {
    var v = S.view;
    $("next-notes").textContent = (v && v.live && S.status.state === "live" && v.next_notes_at)
      ? "Next notes at " + clock12(v.started + v.next_notes_at, false) : "";
  }
  setInterval(function () {
    // The boundary moves on as the meeting runs; recompute from the clock.
    var v = S.view;
    if (!v || !v.live || S.status.state !== "live" || !S.elapsedBase) return;
    var block = v.block_s || 300;
    var t = (Date.now() - S.elapsedBase) / 1000;
    v.next_notes_at = (Math.floor(t / block) + 1) * block;
    updateNextNotes();
  }, 5000);

  /* ---------------------------------------------------------------- tabs */

  // One tab on its own is just a title; it only becomes a switch once the
  // meeting has a full summary to switch to.
  function setTabs(hasSummary) {
    $("tab-summary").hidden = !hasSummary;
    document.querySelector(".tabs").classList.toggle("single", !hasSummary);
    $("tab-transcript").disabled = !hasSummary;
  }

  function selectTab(which) {
    var summary = which === "summary";
    $("tab-transcript").classList.toggle("active", !summary);
    $("tab-summary").classList.toggle("active", summary);
    $("transcript").hidden = summary;
    $("summary-view").hidden = !summary;
    $("jump-transcript").hidden = true;
  }
  $("tab-transcript").addEventListener("click", function () { selectTab("transcript"); });
  $("tab-summary").addEventListener("click", function () { selectTab("summary"); });

  /* ---------------------------------------------------------------- full summary */

  var LIVE_TIP = "Stops listening and writes a summary of the whole meeting — the topics " +
    "discussed and what was concluded on each. It's kept with this meeting in the history, " +
    "and you can save a copy anywhere. Takes several minutes for a long meeting. Listening " +
    "can't be resumed afterwards.";
  var PAST_TIP = "Writes a summary of the whole meeting — the topics discussed and what was " +
    "concluded on each. It's kept with this meeting in the history, and you can save a copy " +
    "anywhere. Takes several minutes for a long meeting.";

  function updateFooter() {
    var v = S.view;
    var hasSummary = v && v.has_summary;
    $("summary-actions").hidden = !hasSummary;
    $("summary-make").hidden = !v || hasSummary;
    if (!v || hasSummary) return;
    var isLive = v.id === S.liveId;
    $("make-tip").setAttribute("data-tip", isLive ? LIVE_TIP : PAST_TIP);
    var state = S.status.state;
    var running = S.summary && S.summary.state === "running";
    var ok = words(v.lines) >= 40 && !running &&
      (isLive ? state === "live" : state === "idle");
    $("make-summary").disabled = !ok;
    $("make-summary").title = (!isLive && state !== "idle" && !running)
      ? "Stop the live meeting first." : "";
  }

  $("make-summary").addEventListener("click", function () {
    var v = S.view;
    if (!v) return;
    var isLive = v.id === S.liveId;
    var text = isLive
      ? "This will stop listening and generate a full summary of the meeting so far. It can " +
        "take several minutes, and listening can't be resumed. Continue?"
      : "This will generate a full summary of this meeting. It can take several minutes. Continue?";
    confirmModal("Generate full summary", text, "Continue").then(function (yes) {
      if (!yes) return;
      banner("Choose where to save a copy of the summary…");
      $("make-summary").disabled = true;
      api("POST", "/api/summary/dialog", { id: v.id }).then(function (d) {
        banner("");
        if (!d.path) { updateFooter(); return; }   // cancelled: nothing changes
        return api("POST", "/api/summary/start", { id: v.id, path: d.path });
      }).catch(function (e) { banner(e.message, "error"); updateFooter(); });
    });
  });

  function onSummaryProgress(p) {
    S.summary = p;
    $("summary-panel").hidden = false;
    $("controls").hidden = true;
    $("sp-msg").textContent = p.message || "";
    $("sp-fill").style.width = Math.round((p.fraction || 0) * 100) + "%";
    updateFooter();
  }

  function onSummaryDone(ev) {
    S.summary = ev;
    $("summary-panel").hidden = true;
    $("controls").hidden = false;
    if (ev.state === "done") {
      if (S.view && S.view.id === ev.id) {
        S.view.summary = ev.summary || "";
        S.view.has_summary = true;
        S.view.live = false;
        setTabs(true);
        $("summary-view").innerHTML = renderMarkdown(S.view.summary);
        selectTab("summary");
      }
      banner(ev.copy_error || "");
    } else if (ev.state === "failed") {
      banner(ev.message, "error");
    } else if (ev.state === "cancelled") {
      banner("");
    }
    if (S.view && S.view.id === ev.id && S.view.live) S.view.live = false;
    updateFooter();
    loadHistory();
  }

  $("sp-cancel").addEventListener("click", function () {
    $("sp-msg").textContent = "Cancelling…";
    api("POST", "/api/summary/cancel").catch(function () {});
  });

  $("open-file").addEventListener("click", function () {
    if (S.view) api("POST", "/api/open", { id: S.view.id, what: "file" }).catch(function (e) { banner(e.message, "error"); });
  });
  $("open-folder").addEventListener("click", function () {
    if (S.view) api("POST", "/api/open", { id: S.view.id, what: "folder" }).catch(function (e) { banner(e.message, "error"); });
  });

  /* ---------------------------------------------------------------- history */

  function loadHistory() {
    api("GET", "/api/history").then(function (d) {
      var host = $("history");
      host.innerHTML = "";
      if (d.live) host.appendChild(historyRow(d.live, true));
      d.meetings.forEach(function (m) { host.appendChild(historyRow(m, false)); });
      $("history-empty").hidden = !!(d.live || d.meetings.length);
      markActiveRow();
    }).catch(function () {});
  }

  function historyRow(m, isLive) {
    var row = document.createElement("div");
    row.className = "hrow";
    row.dataset.id = m.id;
    var main = document.createElement("div");
    main.className = "hmain";
    var name = document.createElement("div");
    name.className = "hname";
    name.textContent = m.name + (m.has_summary ? " ✓" : "");
    var meta = document.createElement("div");
    meta.className = "hmeta";
    if (isLive) {
      meta.innerHTML = '<span class="hlive">● Live</span>';
    } else {
      meta.textContent = humanDuration(m.duration_s) +
        (m.name.indexOf(humanDate(m.started).split(" ")[0]) === -1 ? " · " + humanDate(m.started) : "");
    }
    main.appendChild(name);
    main.appendChild(meta);
    row.appendChild(main);

    var ren = document.createElement("button");
    ren.className = "iconbtn";
    ren.type = "button";
    ren.title = "Rename";
    ren.textContent = "✎";
    ren.addEventListener("click", function (e) { e.stopPropagation(); renameRow(row, m); });
    row.appendChild(ren);
    row.addEventListener("click", function () { openMeeting(m.id); });
    // Delete lives behind a right-click, not on the row: too easy to hit.
    row.addEventListener("contextmenu", function (e) {
      e.preventDefault();
      openRowMenu(e.clientX, e.clientY, row, m, isLive);
    });
    return row;
  }

  function openRowMenu(x, y, row, m, isLive) {
    var menu = $("rowmenu");
    menu.innerHTML = "";
    var items = [["Rename", function () { renameRow(row, m); }]];
    if (!isLive) items.push(["Delete", function () { deleteRow(m); }, "danger"]);
    items.forEach(function (it) {
      var b = document.createElement("button");
      b.type = "button";
      b.textContent = it[0];
      if (it[2]) b.className = it[2];
      b.addEventListener("click", function () { closeRowMenu(); it[1](); });
      menu.appendChild(b);
    });
    menu.hidden = false;
    var w = menu.offsetWidth, h = menu.offsetHeight;
    menu.style.left = Math.min(x, window.innerWidth - w - 8) + "px";
    menu.style.top = Math.min(y, window.innerHeight - h - 8) + "px";
    menu.firstChild.focus();
  }

  function closeRowMenu() { $("rowmenu").hidden = true; }

  document.addEventListener("click", function (e) {
    if (!$("rowmenu").hidden && !$("rowmenu").contains(e.target)) closeRowMenu();
  });
  window.addEventListener("blur", closeRowMenu);
  window.addEventListener("resize", closeRowMenu);

  function markActiveRow() {
    var id = S.view ? S.view.id : "";
    Array.prototype.forEach.call(document.querySelectorAll(".hrow"), function (r) {
      r.classList.toggle("active", r.dataset.id === id);
    });
  }

  function renameRow(row, m) {
    var main = row.querySelector(".hmain");
    var input = document.createElement("input");
    input.value = m.name;
    main.innerHTML = "";
    main.appendChild(input);
    input.focus();
    input.select();
    var done = false;
    function finish(save) {
      if (done) return;
      done = true;
      var name = input.value.trim();
      if (save && name && name !== m.name) {
        api("POST", "/api/history/" + encodeURIComponent(m.id) + "/rename", { name: name })
          .then(function () {
            if (S.view && S.view.id === m.id) {
              S.view.name = name;
              if (!S.view.live) $("view-name").textContent = name;
            }
          })
          .catch(function (e) { banner(e.message, "error"); })
          .then(loadHistory);
      } else {
        loadHistory();
      }
    }
    input.addEventListener("click", function (e) { e.stopPropagation(); });
    input.addEventListener("keydown", function (e) {
      if (e.key === "Enter") finish(true);
      if (e.key === "Escape") finish(false);
    });
    input.addEventListener("blur", function () { finish(true); });
  }

  function deleteRow(m) {
    confirmModal("Delete meeting",
      "Delete this meeting's transcript, notes and summary? This can't be undone.",
      "Delete", true).then(function (yes) {
      if (!yes) return;
      api("DELETE", "/api/history/" + encodeURIComponent(m.id)).then(function () {
        if (S.view && S.view.id === m.id) {
          if (S.liveId) openMeeting(S.liveId);
          else showMeeting({ id: "", name: "", live: false, lines: [], blocks: [], summary: "",
                             has_summary: false });
        }
        loadHistory();
      }).catch(function (e) { banner(e.message, "error"); });
    });
  }

  function toggleSidebar() {
    var side = $("sidebar");
    side.classList.toggle("collapsed");
    var collapsed = side.classList.contains("collapsed");
    $("side-toggle").textContent = collapsed ? "›" : "‹";
    $("side-toggle").setAttribute("aria-label", collapsed ? "Expand history" : "Collapse history");
  }
  $("side-toggle").addEventListener("click", toggleSidebar);
  $("side-vertical").addEventListener("click", toggleSidebar);

  /* ---------------------------------------------------------------- modal */

  var modalResolve = null;

  function confirmModal(title, text, okLabel, danger) {
    $("modal-title").textContent = title;
    $("modal-text").textContent = text;
    $("modal-ok").textContent = okLabel || "Continue";
    $("modal-ok").classList.toggle("danger", !!danger);
    $("modal").hidden = false;
    $("modal-ok").focus();
    return new Promise(function (resolve) { modalResolve = resolve; });
  }

  function closeModal(answer) {
    $("modal").hidden = true;
    if (modalResolve) { var r = modalResolve; modalResolve = null; r(answer); }
  }

  $("modal-ok").addEventListener("click", function () { closeModal(true); });
  $("modal-cancel").addEventListener("click", function () { closeModal(false); });
  $("modal").addEventListener("click", function (e) { if (e.target === $("modal")) closeModal(false); });

  /* ---------------------------------------------------------------- diagnostics */

  $("copy-diag").addEventListener("click", function () {
    fetch("/api/diagnostics")
      .then(function (r) { return r.text(); })
      .then(function (text) {
        var btn = $("copy-diag");
        var done = function () {
          btn.textContent = "Copied";
          setTimeout(function () { btn.textContent = "Copy diagnostic info"; }, 1800);
        };
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(text).then(done, function () { legacyCopy(text); done(); });
        } else { legacyCopy(text); done(); }
      });
  });

  function legacyCopy(text) {
    var ta = document.createElement("textarea");
    ta.value = text; document.body.appendChild(ta); ta.select();
    try { document.execCommand("copy"); } catch (e) { /* nothing to do */ }
    document.body.removeChild(ta);
  }

  /* ------------------------------------------------------------- markdown */

  function esc(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function inline(s) {
    return esc(s)
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[^*])\*([^*]+)\*/g, "$1<em>$2</em>")
      .replace(/(^|[^_\w])_([^_]+)_(?=$|[^_\w])/g, "$1<em>$2</em>");
  }

  // Just enough Markdown for what the model writes: headings, bullets,
  // numbered lists, rules, bold and italics.
  function renderMarkdown(md) {
    var out = [], list = null;
    var lines = (md || "").split(/\r?\n/);
    function closeList() { if (list) { out.push("</" + list + ">"); list = null; } }
    for (var i = 0; i < lines.length; i++) {
      var line = lines[i];
      var h = line.match(/^(#{1,4})\s+(.*)$/);
      if (h) { closeList(); out.push("<h" + h[1].length + ">" + inline(h[2]) + "</h" + h[1].length + ">"); continue; }
      if (/^---+\s*$/.test(line)) { closeList(); out.push("<hr>"); continue; }
      var bullet = line.match(/^\s*[-*]\s+(.*)$/);
      if (bullet) {
        if (list !== "ul") { closeList(); out.push("<ul>"); list = "ul"; }
        out.push("<li>" + inline(bullet[1]) + "</li>");
        continue;
      }
      var num = line.match(/^\s*\d+\.\s+(.*)$/);
      if (num) {
        if (list !== "ol") { closeList(); out.push("<ol>"); list = "ol"; }
        out.push("<li>" + inline(num[1]) + "</li>");
        continue;
      }
      if (line.trim() === "") { closeList(); continue; }
      closeList();
      out.push("<p>" + inline(line) + "</p>");
    }
    closeList();
    return out.join("\n");
  }

  /* ----------------------------------------------------------- about */

  // The only outbound request the app ever makes lives behind this button,
  // and it goes through the server: the page never talks to an external host.
  var about = { timer: null, pending: null };

  function humanSize(n) {
    if (!n) return "0 MB";
    return n >= 1e9 ? (n / 1e9).toFixed(1) + " GB" : Math.max(1, Math.round(n / 1e6)) + " MB";
  }

  function openAbout() {
    $("about").hidden = false;
    resetUpdate();
    fetch("/api/about")
      .then(function (r) { return r.json(); })
      .then(function (d) {
        $("about-author").textContent = d.author || "";
        $("about-version").textContent = d.version || "";
        var link = $("about-repo");
        link.textContent = (d.repo_url || "").replace(/^https:\/\//, "");
        link.href = d.repo_url || "#";
      })
      .catch(function () { $("about-version").textContent = "unknown"; });
  }

  function closeAbout() {
    $("about").hidden = true;
    if (about.timer) { clearInterval(about.timer); about.timer = null; }
  }

  function resetUpdate() {
    $("update-install").hidden = true;
    $("update-cancel").hidden = true;
    $("update-bar").hidden = true;
    $("update-msg").hidden = true;
    $("update-fill").style.width = "0";
    $("update-check").disabled = false;
    $("update-check").textContent = "Check for updates";
  }

  function updateMsg(text) {
    var el = $("update-msg");
    el.textContent = text;
    el.hidden = !text;
  }

  $("about-open").addEventListener("click", openAbout);
  $("about-close").addEventListener("click", closeAbout);
  $("about").addEventListener("click", function (e) { if (e.target === $("about")) closeAbout(); });
  document.addEventListener("keydown", function (e) {
    if (e.key !== "Escape") return;
    if (!$("rowmenu").hidden) closeRowMenu();
    else if (!$("modal").hidden) closeModal(false);
    else if (!$("about").hidden) closeAbout();
  });

  $("update-check").addEventListener("click", function () {
    $("update-check").disabled = true;
    $("update-check").textContent = "Checking…";
    $("update-install").hidden = true;
    updateMsg("");
    fetch("/api/update/check", { method: "POST" })
      .then(function (r) { return r.json(); })
      .then(function (d) {
        $("update-check").disabled = false;
        $("update-check").textContent = "Check for updates";
        if (d.status !== "available") { updateMsg(d.message || "Nothing to do."); return; }
        about.pending = d;
        updateMsg("Version " + d.latest + " is available (" + humanSize(d.size_bytes) + ").");
        if (d.writable === false) {
          updateMsg("Version " + d.latest + " is available, but this folder " +
                    "can't be written to, so it can't be installed here.");
          return;
        }
        $("update-install").hidden = false;
      })
      .catch(function () {
        $("update-check").disabled = false;
        $("update-check").textContent = "Check for updates";
        updateMsg("Couldn't reach GitHub. Check the network connection.");
      });
  });

  $("update-install").addEventListener("click", function () {
    var d = about.pending;
    if (!d) return;
    $("update-install").hidden = true;
    $("update-check").disabled = true;
    $("update-cancel").hidden = false;
    $("update-bar").hidden = false;
    api("POST", "/api/update/install", { url: d.url, size_bytes: d.size_bytes, tag: d.latest })
      .then(pollUpdate)
      .catch(function (e) { resetUpdate(); updateMsg(e.message); });
  });

  $("update-cancel").addEventListener("click", function () {
    fetch("/api/update/cancel", { method: "POST" });
  });

  function pollUpdate() {
    if (about.timer) clearInterval(about.timer);
    about.timer = setInterval(function () {
      fetch("/api/update/progress")
        .then(function (r) { return r.json(); })
        .then(function (p) {
          if (p.total) {
            $("update-fill").style.width = (p.downloaded / p.total * 100) + "%";
            updateMsg(p.message + " — " + humanSize(p.downloaded) + " of " + humanSize(p.total));
          } else if (p.message) {
            updateMsg(p.message);
          }
          if (p.phase !== "downloading") $("update-cancel").hidden = true;
          if (p.phase === "restarting") {
            clearInterval(about.timer); about.timer = null;
            $("update-fill").style.width = "100%";
            updateMsg("Update ready. The application will close and reopen by itself — " +
                      "this page will reconnect.");
          } else if (p.phase === "error") {
            clearInterval(about.timer); about.timer = null;
            resetUpdate();
            updateMsg(p.message);
          } else if (p.phase === "idle") {
            clearInterval(about.timer); about.timer = null;
            resetUpdate();
            updateMsg("Update cancelled. Nothing was changed.");
          }
        })
        .catch(function () { /* the server may already be restarting */ });
    }, 500);
  }

  waitForServer();
})();
