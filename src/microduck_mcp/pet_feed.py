"""The pet's link to the daemon: one background thread, four endpoints.

The overlay window must never block on the network. Cocoa's main thread has
exactly one job — draw the last frame that arrived and move the window to where
that frame says the duck is — and everything with a socket in it happens here,
on a thread, behind a lock. When the daemon goes away the thread keeps trying
and the window keeps drawing the last frame it got: **the duck freezes**. That
freeze is not a failure mode to paper over, it is the proof that nothing in
this app is animated. There is no interpolation, no idle loop, no fallback
gait. If the physics stops, the picture stops.

The endpoints are `webui.py`'s (`GET /pet/frame`, `GET /pet/state`,
`POST /pet/config`, `POST /pet/push`), and the important one is deliberately
one request per frame: `sim.submit()` costs at least one 50 Hz tick, so a
separate pose poll would double the tick cost of every frame for information
the render already had in its hands. The pose rides back on the frame
response's `X-Duck-Pet` header. `/pet/state` is only the fallback for a daemon
that answers frames without it, and is polled at a fraction of the frame rate.

Frame budget: rendering happens **on the sim thread**. Measured, 640×480:
25 fps leaves the sim at 1.00× realtime, 42 fps drops it to 0.012× — there is
no warning between the two. `DEFAULT_FPS` is 20 and `--fps` is clamped below
the cliff on purpose. Never point two frame consumers at one daemon.
"""

import http.client
import json
import queue
import threading
import time
import urllib.parse

# `duck-sim --web` defaults to 8400, and the pet is meant to be a *viewer* onto
# the daemon the MCP already drives (notes/desktop-pet.md: "shared daemon"), not
# a second sim of its own. Same number on both sides, so the shipped pair takes
# no flags at all; dev daemons live on 8410-8419 and get --port.
DEFAULT_PORT = 8400
DEFAULT_FPS = 20.0
MAX_FPS = 25.0          # the measured cliff is at ~40; stay well under it
STATE_EVERY = 6         # frames between fallback /pet/state polls
HTTP_TIMEOUT_S = 2.5
OFFLINE_RETRY_S = 0.75
# What the feed drops to when nobody can see the window — the display asleep,
# the session switched away, the overlay fully occluded. Not zero: one frame a
# second is what notices that the daemon came back, and keeps the pose fresh
# enough that the window is in the right place the instant the screen lights
# up. Each frame costs a ~40 ms render on the SIM thread, so the difference
# between this and 20 fps is a pinned core for nothing.
IDLE_FPS = 1.0
# A queued shove is a gesture, and a gesture goes stale. `_run` raises out
# before it drains while the daemon is down, so without this the drag you made
# while the sim was crashed arrives, minutes later, the moment it comes back.
PUSH_STALE_S = 1.0

# Where the pose rides back with its own frame (webui.py `_pet_frame`).
POSE_HEADER = "X-Duck-Pet"


class PetFeed:
    """Polls the daemon for frames; carries pushes the other way.

    One persistent `HTTPConnection` (loopback keep-alive: reconnecting per
    frame at 20 fps is pure syscall tax), reconnected on any error. The public
    surface is `snapshot()` — a consistent dict the UI thread can read without
    reasoning about tearing — and `push()`, which is fire-and-forget.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT,
                 fps: float = DEFAULT_FPS, config: dict = None,
                 frame_px: int = 512, supersample: int = 2):
        self.host = host
        self.port = int(port)
        self.period = 1.0 / max(1.0, min(float(fps), MAX_FPS))
        self.frame_px = int(frame_px)
        self.supersample = int(supersample)
        self._config = dict(config or {})
        self._config_dirty = True

        self._lock = threading.Lock()
        self._png = None            # bytes of the last frame that arrived
        self._pose = {}             # the pet_state dict that frame came with
        self._seq = 0               # bumped per new frame; the UI's redraw cue
        self._online = False
        self._error = "starting"
        self._frames = 0
        self._first_frame_at = None

        self._pushes = queue.Queue(maxsize=8)
        self._conn = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="pet-feed",
                                        daemon=True)

    # ---------- lifecycle ----------

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._close()

    def set_fps(self, fps: float):
        """Change the ask rate live — the throttle for a window nobody sees.

        `_run` re-reads `self.period` on every turn, so this takes effect on
        the next one; a float store is atomic under the GIL, and the lock is
        here only to keep every mutation of this object on one discipline.
        """
        with self._lock:
            self.period = 1.0 / max(0.05, min(float(fps), MAX_FPS))

    def set_config(self, config: dict, frame_px: int = None,
                   supersample: int = None):
        """Screen changed under us — renegotiate on the next loop turn."""
        with self._lock:
            self._config = dict(config)
            if frame_px:
                self.frame_px = int(frame_px)
            if supersample:
                self.supersample = int(supersample)
            self._config_dirty = True

    # ---------- what the UI thread reads ----------

    def snapshot(self) -> dict:
        with self._lock:
            return {"seq": self._seq, "png": self._png, "pose": dict(self._pose),
                    "online": self._online, "error": self._error,
                    "frames": self._frames, "fps": self._measured_fps()}

    def _measured_fps(self) -> float:
        if not self._first_frame_at or self._frames < 2:
            return 0.0
        dt = time.time() - self._first_frame_at
        return (self._frames - 1) / dt if dt > 0 else 0.0

    def push(self, body: dict):
        """Queue a shove. Dropped rather than blocked if the feed is wedged —
        a UI thread that waits on a socket is a beachball."""
        try:
            self._pushes.put_nowait((time.monotonic(), dict(body)))
        except queue.Full:
            pass

    # ---------- the thread ----------

    def _run(self):
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                self._send_config_if_dirty()
                self._drain_pushes()
                self._fetch_frame()
                self._mark_online()
            except Exception as e:                      # noqa: BLE001 — any
                self._mark_offline(e)                   # failure is "offline"
                self._close()
                self._stop.wait(OFFLINE_RETRY_S)
                continue
            slack = self.period - (time.monotonic() - started)
            if slack > 0:
                self._stop.wait(slack)

    # ---------- HTTP ----------

    def _connection(self):
        if self._conn is None:
            self._conn = http.client.HTTPConnection(self.host, self.port,
                                                    timeout=HTTP_TIMEOUT_S)
        return self._conn

    def _close(self):
        conn, self._conn = self._conn, None
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    def _request(self, method: str, path: str, body: dict = None):
        conn = self._connection()
        headers = {"Accept": "*/*"}
        payload = None
        if body is not None:
            payload = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        conn.request(method, path, body=payload, headers=headers)
        resp = conn.getresponse()
        data = resp.read()          # always drain, or keep-alive desynchronises
        return resp.status, dict(resp.getheaders()), data

    def _send_config_if_dirty(self):
        with self._lock:
            if not self._config_dirty:
                return
            config = dict(self._config)
        status, _headers, data = self._request("POST", "/pet/config", config)
        with self._lock:
            # One attempt per reconnect: a daemon that refuses the config (a
            # non-pet scene, say) will refuse it again next frame too, and
            # hammering it would only cost the sim ticks. The error text is
            # kept — it is the honest reason the window is empty.
            self._config_dirty = False
        obj = _loads(data)
        if status == 200:
            self._absorb_state(obj)
        else:
            self._note_error("config", status, obj)

    def _drain_pushes(self):
        while True:
            try:
                queued_at, shove = self._pushes.get_nowait()
            except queue.Empty:
                return
            if time.monotonic() - queued_at > PUSH_STALE_S:
                # The daemon was away when this gesture was made (the loop
                # raises before it gets here while offline). Landing it now
                # would be a shove out of nowhere, unrelated to anything the
                # hand on the mouse is doing at this moment.
                continue
            status, _h, data = self._request("POST", "/pet/push", shove)
            if status != 200:
                self._note_error("push", status, _loads(data))

    def _fetch_frame(self):
        path = "/pet/frame?" + urllib.parse.urlencode(
            {"size_px": self.frame_px, "supersample": self.supersample})
        status, headers, data = self._request("GET", path)
        if status != 200 or not data:
            obj = _loads(data)
            self._note_error("frame", status, obj)
            raise RuntimeError(_message(obj) or f"/pet/frame -> HTTP {status}")
        pose = _loads(_header(headers, POSE_HEADER))
        with self._lock:
            self._png = data
            self._seq += 1
            self._frames += 1
            if self._first_frame_at is None:
                self._first_frame_at = time.time()
            if pose:
                self._pose = pose
            need_state = not pose and (self._frames % STATE_EVERY == 1)
        if need_state:
            # A daemon that ships frames without the header: pay the extra
            # tick, but only occasionally.
            s, _h, body = self._request("GET", "/pet/state")
            if s == 200:
                self._absorb_state(_loads(body))

    def _absorb_state(self, obj):
        if not isinstance(obj, dict):
            return
        with self._lock:
            self._pose.update(obj)

    # ---------- online/offline bookkeeping ----------

    def _mark_online(self):
        with self._lock:
            self._online, self._error = True, ""

    def _mark_offline(self, exc):
        with self._lock:
            self._online = False
            # `_fetch_frame` already folded the daemon's own words into the
            # exception, so this is the whole story either way.
            self._error = f"{type(exc).__name__}: {exc}"
            # The daemon we come back to may not be the daemon we left — a
            # restarted sim has default walls and no idea what screen it is
            # standing on. Renegotiate on reconnect, every time.
            self._config_dirty = True

    def _note_error(self, where, status, obj):
        with self._lock:
            self._error = f"{where}: HTTP {status} {_message(obj) or ''}".strip()


def _header(headers: dict, name: str):
    """HTTP header names are case-insensitive; `getheaders()` is not."""
    lowered = name.lower()
    for k, v in headers.items():
        if k.lower() == lowered:
            return v
    return None


def _message(obj):
    if isinstance(obj, dict):
        return obj.get("error") or obj.get("note")
    return None


def _loads(raw):
    """JSON that never raises — a malformed header must not kill the feed."""
    if not raw:
        return None
    if isinstance(raw, bytes):
        try:
            raw = raw.decode()
        except UnicodeDecodeError:
            return None
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return None
