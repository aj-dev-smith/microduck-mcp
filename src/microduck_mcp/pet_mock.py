"""A fake sim behind the real HTTP contract — for testing the *window*, only.

`duck-pet --mock` boots this instead of talking to a daemon, so the parts that
are hard to get right on macOS — a borderless window that floats above the
Dock, transparent where the duck is not, that travels as the duck walks and
takes a drag — can be built and screenshotted with no sim, no MuJoCo and no
GPU anywhere in the picture.

Read the boundary carefully, because it is the whole point of putting the mock
*here* rather than in the app:

  **This module is a stand-in SIM, not a stand-in duck.** It answers the same
  four routes as `webui.py`, in the same units, with the same header, and the
  app cannot tell the difference. The app contains no animation data and no
  fallback gait; when it is drawing a walk cycle it is because something on
  the other end of a TCP connection drew one and sent it. Delete this file and
  the app still works — it just has nothing to show until a daemon answers.

So the crude sweep and the paper-cutout duck below are honest: they are what a
very bad physics engine looks like through the contract. Anything that made
the *app* generate motion would not be.
"""

import io
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import pet_map

# `--mock` cannot use pet_feed.DEFAULT_PORT: that is duck-sim's own web port,
# and the whole point of the mock is running with a daemon absent — or with a
# real one on the box that this must not fight over. 8419 is the far end of
# the block reserved for this project.
MOCK_PORT = 8419

WALK_MPS = 0.3          # the walk policy's max vx (sim_server.py:470)
STEP_HZ = 2.2           # made up; the real gait comes from the ONNX policy
DRAG_S = 1.4            # how fast a push bleeds off, seconds
PUSH_MAX = 2.0          # sim_server.PET_PUSH_MAX
DRAG_GAIN = 6.0         # webui.PET_DRAG_GAIN


class MockDuck:
    """A duck on rails: paces between the walls, staggers when shoved."""

    def __init__(self, half_span_m=1.05):
        self.half_span_m = half_span_m
        self.x = 0.0
        self.z = 0.116
        self.vx = WALK_MPS
        self.kick_x = 0.0
        self.kick_z = 0.0
        self.t0 = time.time()
        self.last = self.t0
        self.lock = threading.Lock()

    def step(self):
        with self.lock:
            now = time.time()
            dt = min(0.25, now - self.last)
            self.last = now
            decay = math.exp(-dt / DRAG_S)
            self.x += (self.vx + self.kick_x) * dt
            self.z = max(0.116, self.z + self.kick_z * dt - 1.2 * dt * dt)
            self.kick_x *= decay
            self.kick_z = self.kick_z * decay - 4.0 * dt   # gravity, roughly
            if self.z <= 0.116:
                self.z, self.kick_z = 0.116, max(0.0, self.kick_z)
            if self.x > self.half_span_m:
                self.x, self.vx = self.half_span_m, -abs(self.vx)
            elif self.x < -self.half_span_m:
                self.x, self.vx = -self.half_span_m, abs(self.vx)
            return self.x, self.z, now - self.t0

    def push(self, dx_m, dy_m):
        """Same arithmetic webui does: metres of gesture × gain, clamped."""
        with self.lock:
            self.kick_x += max(-PUSH_MAX, min(PUSH_MAX, dx_m * DRAG_GAIN))
            self.kick_z += max(-PUSH_MAX, min(PUSH_MAX, -dy_m * DRAG_GAIN))

    def set_half_span(self, half_span_m):
        with self.lock:
            self.half_span_m = max(0.1, float(half_span_m))


def render(px: int, t: float, view_m: float, floor_pad_px: int,
           facing: int, lift_m: float) -> bytes:
    """A duck-shaped hole in a transparent square, `px` on a side."""
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (px, px), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    m2px = px / max(1e-6, view_m)
    floor = px - floor_pad_px - lift_m * m2px
    h = pet_map.DUCK_HEIGHT_M * m2px
    w = pet_map.DUCK_DEPTH_M * m2px      # side-on, the duck is tall and narrow
    cx = px * 0.5
    phase = 2 * math.pi * STEP_HZ * t
    bob = 0.018 * h * math.sin(2 * phase)
    body_b = floor - 0.17 * h + bob
    body_t = body_b - 0.44 * h
    for i, sign in enumerate((-1, 1)):   # legs first: the body sits over them
        swing = 0.06 * h * math.sin(phase + i * math.pi)
        d.line([(cx + sign * 0.14 * w, body_b - 0.02 * h),
                (cx + sign * 0.14 * w + swing, floor)],
               fill=(214, 138, 46, 255), width=max(2, int(0.05 * h)))
    d.ellipse([cx - 0.5 * w, body_t, cx + 0.5 * w, body_b],
              fill=(242, 199, 92, 255))
    head_r = 0.145 * h
    head_cx = cx + facing * 0.20 * w
    head_cy = body_t - 0.62 * head_r + bob * 0.5
    d.ellipse([head_cx - head_r, head_cy - head_r,
               head_cx + head_r, head_cy + head_r], fill=(246, 209, 110, 255))
    beak = facing * head_r * 1.5
    d.polygon([(head_cx + facing * head_r * 0.5, head_cy - 0.15 * head_r),
               (head_cx + beak, head_cy + 0.10 * head_r),
               (head_cx + facing * head_r * 0.5, head_cy + 0.45 * head_r)],
              fill=(233, 141, 45, 255))
    eye = 0.13 * head_r
    d.ellipse([head_cx + facing * 0.25 * head_r - eye,
               head_cy - 0.42 * head_r - eye,
               head_cx + facing * 0.25 * head_r + eye,
               head_cy - 0.42 * head_r + eye], fill=(30, 26, 22, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


def start_mock(port: int) -> ThreadingHTTPServer:
    """Serve /pet/{frame,state,push,config} on loopback, from a daemon thread."""
    duck = MockDuck()
    cfg = {"px_per_meter": 1312.0, "frame_px": 512, "supersample": 2,
           "screen_width_m": 2.634, "floor_pad_px": pet_map.DEFAULT_FLOOR_PAD_PX,
           "wall_margin_m": 512 / (2 * 1312.0),
           "camera_distance_m": 1.0, "azimuth_deg": 90.0, "elevation_deg": 0.0}

    def wall_m():
        return max(0.05, cfg["screen_width_m"] / 2 - cfg["wall_margin_m"])

    def config_state():
        return {**cfg, "render_px": cfg["frame_px"] * cfg["supersample"],
                "view_height_m": cfg["frame_px"] / cfg["px_per_meter"],
                "walls_m": [-wall_m(), wall_m()],
                "duck_height_m": pet_map.DUCK_HEIGHT_M,
                "duck_height_px": pet_map.DUCK_HEIGHT_M * cfg["px_per_meter"],
                "platform_capacity": 0, "segmentation": True, "mock": True}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"   # keep-alive, like the real route

        def log_message(self, *args):
            pass

        def _send(self, code, body, ctype, extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def _state(self, x, z, t):
            return {"ok": True, "sim_time_s": round(t, 3),
                    "base_x_m": x, "base_y_m": 0.0, "base_z_m": z,
                    "heading_deg": 90.0 if duck.vx >= 0 else -90.0,
                    "vel_world_mps": [duck.vx, 0.0, 0.0],
                    "upright": True, "fallen": False, "sitting": False,
                    "active_policy": "walking", "behavior": "drive",
                    "vel_cmd": [duck.vx, 0.0, 0.0],
                    "machine": {"name": "mock", "armed": True, "node": "wander"},
                    "inhabited": False, "inhabited_age_s": None,
                    "inhabited_cmd": None,
                    "screen": {"center_offset_px": x * cfg["px_per_meter"],
                               "floor_px_from_bottom": cfg["floor_pad_px"],
                               "px_per_meter": cfg["px_per_meter"],
                               "frame_px": cfg["frame_px"]},
                    "config": config_state()}

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                return json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                return {}

        def do_GET(self):
            path, _, query = self.path.partition("?")
            q = dict(p.split("=", 1) for p in query.split("&") if "=" in p)
            x, z, t = duck.step()
            if path == "/pet/frame":
                size = max(32, min(512, int(float(q.get("size_px",
                                                        cfg["frame_px"])))))
                png = render(size, t, size / cfg["px_per_meter"],
                             round(cfg["floor_pad_px"] * size / cfg["frame_px"]),
                             1 if duck.vx >= 0 else -1, z - 0.116)
                state = self._state(x, z, t)
                self._send(200, png, "image/png",
                           {"X-Duck-Pet": json.dumps(state,
                                                     separators=(",", ":"))})
            elif path == "/pet/state":
                self._json(self._state(x, z, t))
            else:
                self._json({"ok": False, "error": "not found"}, 404)

        def do_POST(self):
            path = self.path.partition("?")[0]
            body = self._body()
            if path == "/pet/push":
                duck.push(float(body.get("dx_m", 0.0)),
                          float(body.get("dy_m", 0.0)))
                self._json({"ok": True, "pushed": body})
            elif path == "/pet/config":
                for k in ("px_per_meter", "frame_px", "supersample",
                          "floor_pad_px", "wall_margin_m", "camera_distance_m"):
                    if k in body:
                        cfg[k] = (int(body[k]) if k in ("frame_px", "supersample",
                                                        "floor_pad_px")
                                  else float(body[k]))
                if "screen_width_px" in body:
                    cfg["screen_width_m"] = (float(body["screen_width_px"])
                                             / cfg["px_per_meter"])
                if "wall_margin_m" not in body:
                    # The daemon's rule, and the reason the app overrides it:
                    # left alone the margin tracks the window size, which puts
                    # the walls inside the arc pet.toml turns around in.
                    cfg["wall_margin_m"] = cfg["frame_px"] / (2 * cfg["px_per_meter"])
                duck.set_half_span(wall_m())
                self._json({"ok": True, "config": config_state(),
                            "note": f"walls at ±{wall_m():.3f} m"})
            else:
                self._json({"ok": False, "error": "not found"}, 404)

    # Loopback, always — `host` is where duck-pet CONNECTS, and a user who
    # points --host at a daemon on another machine has not asked to publish
    # a stand-in duck on every interface of this one. Every other listener in
    # this repo is pinned the same way (webui.start_web).
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"mock sim (a stand-in DAEMON, not a stand-in duck): "
          f"http://127.0.0.1:{port}/pet/frame", flush=True)
    return server
