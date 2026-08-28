"""Agent Experience debug page for the Microduck sim.

A tiny stdlib HTTP server (no extra dependencies) that shows, live, what the
agents driving the duck are doing: every command arriving on the control
socket (tagged by client), the camera view, and the robot state. All sim/
MuJoCo work still happens on the sim thread — HTTP handlers go through the
same `sim.submit()` queue as every other client.

Serves:  /            the dashboard
         /state       JSON state snapshot
         /log?since=N command-feed events after id N
         /frame.png?view=follow&distance=0.9   fresh camera render
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>microduck AX debug</title>
<style>
  :root { --bg:#101418; --panel:#1a2027; --line:#2a3340; --text:#d7dde4;
          --dim:#8a97a5; --accent:#f5b301; --ok:#4cc38a; --bad:#e5534b; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
         font:14px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace; }
  header { display:flex; align-items:baseline; gap:12px; padding:10px 16px;
           border-bottom:1px solid var(--line); }
  header h1 { font-size:16px; margin:0; color:var(--accent); }
  header .sub { color:var(--dim); font-size:12px; }
  main { display:grid; grid-template-columns: 1fr 660px; gap:12px;
         padding:12px 16px; max-width:1400px; }
  @media (max-width:1100px){ main { grid-template-columns:1fr; } }
  .panel { background:var(--panel); border:1px solid var(--line);
           border-radius:8px; padding:10px 12px; }
  .panel h2 { margin:0 0 8px; font-size:12px; letter-spacing:.08em;
              text-transform:uppercase; color:var(--dim); }
  #feed { height:520px; overflow-y:auto; display:flex; flex-direction:column;
          gap:4px; }
  .ev { display:flex; flex-shrink:0; gap:8px; padding:3px 6px; border-radius:4px;
        background:#141a20; white-space:nowrap; overflow:hidden;
        text-overflow:ellipsis; }
  .ev .t { color:var(--dim); }
  .ev .client { width:38px; color:var(--dim); }
  .ev .client.mcp { color:var(--accent); }
  .ev .cmd { font-weight:bold; }
  .ev.cmd-set_velocity .cmd { color:#6cb6ff; }
  .ev.cmd-trick .cmd { color:#d2a8ff; }
  .ev.cmd-camera .cmd { color:#4cc38a; }
  .ev.cmd-push .cmd, .ev.cmd-reset .cmd { color:#e5534b; }
  .ev .args { color:var(--dim); }
  .ev .err { color:var(--bad); }
  #cam { width:100%; border-radius:6px; background:#000; min-height:300px; }
  .row { display:flex; gap:8px; align-items:center; margin-top:8px;
         flex-wrap:wrap; }
  button { background:#232c36; color:var(--text); border:1px solid var(--line);
           border-radius:6px; padding:4px 10px; font:inherit; cursor:pointer; }
  button.active { border-color:var(--accent); color:var(--accent); }
  #stats { display:grid; grid-template-columns:repeat(3, 1fr); gap:8px;
           margin-top:10px; }
  .stat { background:#141a20; border-radius:6px; padding:8px 10px; }
  .stat .k { font-size:11px; color:var(--dim); text-transform:uppercase; }
  .stat .v { font-size:16px; margin-top:2px; }
  .v.good { color:var(--ok); } .v.bad { color:var(--bad); }
  label { color:var(--dim); font-size:12px; }
</style></head>
<body>
<header><h1>&#129414; microduck AX debug</h1>
  <span class="sub">every intent, as the agent sends it &mdash; and what the agent sees</span>
  <span class="sub" id="conn" style="margin-left:auto"></span>
</header>
<main>
  <section class="panel">
    <h2>Command feed</h2>
    <div id="feed"></div>
    <div class="row">
      <label><input type="checkbox" id="hidePolls" checked> hide state polls</label>
    </div>
  </section>
  <section class="panel">
    <h2>Live camera</h2>
    <img id="cam" alt="camera">
    <div class="row" id="views">
      <button data-v="follow" class="active">follow</button>
      <button data-v="front">front</button>
      <button data-v="side">side</button>
      <button data-v="top">top</button>
      <button data-v="head">head</button>
      <span style="margin-left:auto"><label>fps <select id="fps">
        <option>1</option><option selected>3</option><option>5</option>
      </select></label></span>
    </div>
    <div id="stats"></div>
  </section>
</main>
<script>
let since = 0, view = "follow";
const feed = document.getElementById("feed"), cam = document.getElementById("cam");

document.getElementById("views").addEventListener("click", e => {
  if (!e.target.dataset.v) return;
  view = e.target.dataset.v;
  document.querySelectorAll("#views button").forEach(b =>
    b.classList.toggle("active", b.dataset.v === view));
});

function fmtArgs(a) {
  if (!a || !Object.keys(a).length) return "";
  return Object.entries(a).map(([k, v]) => k + "=" + v).join(" ");
}

async function pollLog() {
  try {
    const r = await fetch("/log?since=" + since);
    const d = await r.json();
    document.getElementById("conn").textContent = "connected";
    for (const ev of d.events) {
      since = Math.max(since, ev.id);
      if (document.getElementById("hidePolls").checked &&
          (ev.cmd === "state" || ev.cmd === "ping")) continue;
      const div = document.createElement("div");
      div.className = "ev cmd-" + ev.cmd;
      div.innerHTML =
        `<span class="t">${new Date(ev.t * 1000).toLocaleTimeString()}</span>` +
        `<span class="client ${ev.client}">${ev.client}</span>` +
        `<span class="cmd">${ev.cmd}</span>` +
        `<span class="args">${fmtArgs(ev.args)}</span>` +
        (ev.ok ? "" : `<span class="err">${ev.note || "error"}</span>`) +
        (ev.ok && ev.note ? `<span class="args">&rarr; ${ev.note}</span>` : "");
      feed.appendChild(div);
      while (feed.children.length > 300) feed.removeChild(feed.firstChild);
      feed.scrollTop = feed.scrollHeight;
    }
  } catch (e) {
    document.getElementById("conn").textContent = "disconnected";
  }
  setTimeout(pollLog, 500);
}

async function pollState() {
  try {
    const s = await (await fetch("/state")).json();
    const vb = s.vel_body_mps || {};
    const stats = [
      ["policy", s.active_policy, ""],
      ["upright", s.upright ? "yes" : "DOWN", s.upright ? "good" : "bad"],
      ["sim time", (s.sim_time_s || 0).toFixed(1) + " s", ""],
      ["position", "(" + s.position_m.slice(0, 2).map(v => v.toFixed(2)) + ")", ""],
      ["fwd vel / cmd", (vb.forward ?? 0).toFixed(2) + " / " + s.vel_cmd[0].toFixed(2), ""],
      ["ball", s.ball_position_m ?
        "(" + s.ball_position_m.slice(0, 2).map(v => v.toFixed(2)) + ")" : "-", ""],
    ];
    document.getElementById("stats").innerHTML = stats.map(([k, v, c]) =>
      `<div class="stat"><div class="k">${k}</div><div class="v ${c}">${v}</div></div>`).join("");
  } catch (e) {}
  setTimeout(pollState, 700);
}

function pollCam() {
  const fps = +document.getElementById("fps").value;
  const img = new Image();
  img.onload = () => { cam.src = img.src; setTimeout(pollCam, 1000 / fps); };
  img.onerror = () => setTimeout(pollCam, 1500);
  img.src = "/frame.png?view=" + view + "&ts=" + Date.now();
}

pollLog(); pollState(); pollCam();
</script>
</body></html>"""


def start_web(sim, port: int) -> ThreadingHTTPServer:
    """Serve the debug page on localhost:port from a daemon thread."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the sim's stdout readable
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path == "/":
                    self._send(200, PAGE.encode(), "text/html; charset=utf-8")
                elif url.path == "/state":
                    self._json(sim.submit({"cmd": "state", "client": "web"}))
                elif url.path == "/log":
                    since = int(q.get("since", 0))
                    events = [e for e in list(sim.events) if e["id"] > since]
                    self._json({"events": events})
                elif url.path == "/frame.png":
                    resp = sim.submit({"cmd": "camera_web", "client": "web",
                                       "view": q.get("view", "follow"),
                                       "distance": float(q.get("distance", 0.9))})
                    if not resp.get("ok"):
                        self._json(resp, 500)
                        return
                    with open(resp["frame"], "rb") as f:
                        self._send(200, f.read(), "image/png")
                else:
                    self._json({"ok": False, "error": "not found"}, 404)
            except Exception as e:
                try:
                    self._json({"ok": False, "error": repr(e)}, 500)
                except Exception:
                    pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"AX debug page: http://127.0.0.1:{port}")
    return server
