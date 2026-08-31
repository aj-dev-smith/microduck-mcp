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

and the desktop pet's stream (notes/desktop-pet.md) — the same queue, no new
protocol, nothing written to disk:

         GET  /pet/frame?size_px=300   RGBA cutout; the pose it was rendered
                                       at rides along in the X-Duck-Pet
                                       response header, because the window has
                                       to move to match the frame it shows and
                                       a second round trip costs a sim tick
         GET  /pet/state               pose, policy, and who has the wheel
         POST /pet/config              the screen the duck is living on
         POST /pet/world               screen rectangles -> platform ledges
         POST /pet/push                a drag gesture -> a real shove
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

# A drag arrives in frame pixels; the sim wants metres per second. Kept here
# rather than in the app so the gesture means the same thing to every client.
PET_DRAG_GAIN = 6.0
PET_PUSH_MAX = 2.0
# Cap on a POST body: these are all small JSON objects, and an HTTP server on
# the sim's front door should not read an arbitrary number of bytes.
MAX_BODY_BYTES = 64 * 1024
# What a push gets when the daemon behind this port is not the pet's. Worded
# like the sim's own refusal (sim_server._pet_not_a_pet_scene) because it is
# the same fact: /pet/frame, /pet/config and /pet/world already say it, and
# only /pet/push could reach a live non-pet session without it.
PET_NOT_A_PET_SCENE = {
    "ok": False,
    "error": "this daemon has no pet geoms — start duck-sim with "
             "--scene desktop (scenes/scene_desktop.xml); refusing to shove "
             "the duck of whatever session owns this port",
}

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
    const vb = s.vel_body_mps || {}, bs = s.ball_seen || {}, mc = s.machine;
    const brg = Math.round(bs.bearing_deg || 0) || 0;  // +ve = turn left
    const stats = [
      ["machine", mc ? mc.node + (mc.armed ? "" : " (disarmed)") : "&mdash;",
       mc && mc.armed ? "good" : ""],
      ["policy", s.active_policy, ""],
      ["upright", s.upright ? "yes" : "DOWN", s.upright ? "good" : "bad"],
      ["sim time", (s.sim_time_s || 0).toFixed(1) + " s", ""],
      ["position", "(" + s.position_m.slice(0, 2).map(v => v.toFixed(2)) + ")", ""],
      ["fwd vel / cmd", (vb.forward ?? 0).toFixed(2) + " / " + s.vel_cmd[0].toFixed(2), ""],
      ["ball (god)", s.ball_position_m ?
        "(" + s.ball_position_m.slice(0, 2).map(v => v.toFixed(2)) + ")" : "-", ""],
      ["ball (seen)", bs.visible ?
        (bs.distance_m ?? 0).toFixed(2) + " m @ " + (brg < 0 ? "" : "+") + brg + "&deg;"
        : "not seen", bs.visible ? "good" : "bad"],
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

    # Is this daemon running a pet scene? Fixed for the daemon's life (it is
    # decided from the compiled model at load), so it is worth exactly one
    # `pet_state` submit — a submit costs a 50 Hz tick, and a shove that spent
    # one to look up a number it was already handed would halve the gesture's
    # latency budget for nothing.
    known = {}

    def _pet_scene(state=None):
        if state is None and "is" not in known:
            state = sim.submit({"cmd": "pet_state", "client": "web"})
        # Only a definite answer is remembered: a probe that failed for some
        # other reason must not latch "not a pet scene" for the daemon's life.
        if isinstance(state, dict) and state.get("ok") and "pet_scene" in state:
            known["is"] = bool(state["pet_scene"])
        return known.get("is", False)

    class Handler(BaseHTTPRequestHandler):
        # HTTP/1.1, i.e. keep-alive. The pet feed holds ONE connection and
        # asks for a frame twenty times a second (pet_feed.PetFeed); under the
        # stdlib default of HTTP/1.0 every one of those was a fresh TCP
        # connect and a fresh ThreadingHTTPServer thread contending with the
        # sim thread for the GIL. Safe because `_send` and `_pet_frame` both
        # emit an exact Content-Length and `_body` reads exactly that many
        # bytes, so no reply can desynchronise the next request.
        protocol_version = "HTTP/1.1"
        # ...and with keep-alive a handler thread otherwise blocks forever on
        # a connection nobody intends to speak on again (or on a body that
        # never arrives). BaseHTTPRequestHandler turns this into a socket
        # timeout and closes the connection, which is the right answer to
        # both.
        timeout = 30.0

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

        def _body(self):
            """The request's JSON object, or None (having already replied)."""
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                n = 0
            if n > MAX_BODY_BYTES:
                self._json({"ok": False, "error": "body too large"}, 413)
                return None
            # A negative Content-Length is not "no body": `rfile.read(-1)`
            # reads to EOF, so a hostile local process could hold a handler
            # thread on the sim's front door open indefinitely. Clamp rather
            # than trust; the `timeout` above catches the short-body case
            # (a header promising more than the client ever sends).
            n = max(0, n)
            raw = self.rfile.read(n) if n else b"{}"
            try:
                body = json.loads(raw or b"{}")
            except ValueError as e:
                self._json({"ok": False, "error": f"bad JSON body: {e}"}, 400)
                return None
            if not isinstance(body, dict):
                self._json({"ok": False, "error": "body must be a JSON object"}, 400)
                return None
            return body

        def _pet_frame(self, q):
            """The cutout, plus its pose in a header.

            The PNG bytes never touch the disk: `camera_web` writes a file per
            request, which is fine three times a second and is not fine twenty
            times a second. `sim.submit` is in-process, so the bytes come back
            in the reply dict and go straight out the socket.
            """
            req = {"cmd": "pet_frame", "client": "web"}
            for key in ("size_px", "supersample"):
                if key in q:
                    try:
                        req[key] = int(q[key])
                    except ValueError:
                        self._json({"ok": False,
                                    "error": f"{key} must be an integer"}, 400)
                        return
            resp = sim.submit(req)
            png = resp.pop("png", None)
            if not resp.get("ok") or png is None:
                self._json(resp, 500)
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Cache-Control", "no-store")
            # One header, one json.loads on the app side: base pose, upright,
            # policy, inhabited. Small (a few hundred bytes) and it keeps the
            # frame and the pose that produced it inseparable.
            self.send_header("X-Duck-Pet", json.dumps(resp, separators=(",", ":")))
            self.send_header("Content-Length", str(len(png)))
            self.end_headers()
            self.wfile.write(png)

        def _pet_push(self, body):
            """A drag on the duck -> the same `push` intent everything else uses.

            The gesture is in the frame's pixels, in screen orientation: +dx
            is rightwards (world +x) and +dy is DOWNWARDS, so a flick upwards
            lifts the duck off the Dock. Nothing here is a special effect —
            it lands in qvel and the real controller has to deal with it.

            Two things this deliberately does NOT do. It does not shove a
            duck that is not the pet's: `pet_state` answers on any scene by
            design (a pose is a pose), and the pet's port is duck-sim's own
            default, so an overlay pointed at the live MCP daemon would
            otherwise render nothing and still be a remote shove button on
            somebody else's session. And it does not spend a `pet_state`
            submit — a whole 50 Hz tick — for a gesture that already arrived
            in metres, which is every gesture the shipped app sends.
            """
            ppm = None
            if "dx_m" not in body or "dy_m" not in body:
                state = sim.submit({"cmd": "pet_state", "client": "web"})
                if not state.get("ok"):
                    self._json(state, 500)
                    return
                if not _pet_scene(state):
                    self._json(PET_NOT_A_PET_SCENE, 400)
                    return
                ppm = state["config"]["px_per_meter"]
            elif not _pet_scene():
                self._json(PET_NOT_A_PET_SCENE, 400)
                return
            try:
                gain = float(body.get("gain", PET_DRAG_GAIN))
                dx_m = (float(body["dx_m"]) if "dx_m" in body
                        else float(body.get("dx_px", 0.0)) / ppm)
                dy_m = (float(body["dy_m"]) if "dy_m" in body
                        else float(body.get("dy_px", 0.0)) / ppm)
            except (TypeError, ValueError):
                self._json({"ok": False, "error": "dx_px/dy_px (or dx_m/dy_m) "
                            "and gain must be numbers"}, 400)
                return
            vx = max(-PET_PUSH_MAX, min(PET_PUSH_MAX, dx_m * gain))
            vz = max(-PET_PUSH_MAX, min(PET_PUSH_MAX, -dy_m * gain))
            req = {"cmd": "push", "client": "web",
                   "magnitude": abs(vx), "angle_deg": 0.0 if vx >= 0 else 180.0,
                   "vz": vz}
            resp = sim.submit(req)
            self._json(resp, 200 if resp.get("ok") else 500)

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
                elif url.path == "/pet/frame":
                    self._pet_frame(q)
                elif url.path == "/pet/state":
                    self._json(sim.submit({"cmd": "pet_state", "client": "web"}))
                else:
                    self._json({"ok": False, "error": "not found"}, 404)
            except Exception as e:
                try:
                    self._json({"ok": False, "error": repr(e)}, 500)
                except Exception:
                    pass

        def do_POST(self):
            url = urlparse(self.path)
            try:
                if url.path not in ("/pet/config", "/pet/world", "/pet/push"):
                    self._json({"ok": False, "error": "not found"}, 404)
                    return
                body = self._body()
                if body is None:
                    return  # _body already answered
                if url.path == "/pet/push":
                    self._pet_push(body)
                    return
                cmd = "pet_config" if url.path == "/pet/config" else "pet_world"
                resp = sim.submit({**body, "cmd": cmd, "client": "web"})
                self._json(resp, 200 if resp.get("ok") else 400)
            except Exception as e:
                try:
                    self._json({"ok": False, "error": repr(e)}, 500)
                except Exception:
                    pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"AX debug page: http://127.0.0.1:{port}")
    return server
