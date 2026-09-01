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
`POST /pet/config`, `POST /pet/push`, `POST /pet/touch`, `POST /pet/sense`,
`POST /pet/carry`), and they travel over that one connection in four different
ways, because they are four different kinds of traffic:

  * **gestures** (`push`, `touch`) are queued and never coalesced — every
    shove and every stroke happened, and dropping one loses an event;
  * **the cursor** is a single coalescing slot, because the pointer's history
    is worthless and only its latest position means anything;
  * **the carry** is a queue of its own with a token in it: the `start` and
    the `end` are structural and must both land, the `move`s between them are
    droppable, and the token that ties them together is MINTED BY THE DAEMON
    and never leaves this class — the app says `carry_start(x, z)` and never
    sees one;
  * **the frame** is the poll everything else rides alongside.

and the important one is deliberately
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
# It covers a pet for the same reason: a stroke made while the daemon was
# crashed is as meaningless as a shove was, and a coo arriving out of nowhere
# a minute later is worse than one that never came.
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

        # (queued_at, path, body) — one queue for every gesture the hand
        # makes, so a shove and a stroke cannot overtake each other.
        self._gestures = queue.Queue(maxsize=8)
        # ...and one SLOT for the pointer, which is not a gesture. Overwritten
        # rather than queued: where the mouse was three samples ago is not
        # information anybody wants delivered late.
        self._sense = None
        self._sense_dirty = False
        # ...and one queue for the pick-up, kept apart from the gestures for
        # the reason a carry is not a gesture: it is a session with a
        # beginning, a middle and an end, and only the middle is droppable.
        # `_carry_token` is the daemon's answer to the `start`, stamped onto
        # every `move` and the `end` — the app never handles one.
        self._carry = queue.Queue(maxsize=16)
        self._carry_token = None
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
                    "frames": self._frames, "fps": self._measured_fps(),
                    # Is there still a grip? The UI thread cannot ask the
                    # daemon and must not assume: a reconnect, or the daemon's
                    # own deadman, ends a carry without the mouse button ever
                    # coming up, and a window that went on believing it was
                    # holding a duck would send an `end` for somebody else's
                    # grab later.
                    "carrying": bool(self._carry_token)}

    def _measured_fps(self) -> float:
        if not self._first_frame_at or self._frames < 2:
            return 0.0
        dt = time.time() - self._first_frame_at
        return (self._frames - 1) / dt if dt > 0 else 0.0

    def push(self, body: dict):
        """Queue a shove — of the duck, or of its toy if the body carries
        `target: "ball"`. Dropped rather than blocked if the feed is wedged —
        a UI thread that waits on a socket is a beachball.

        The body is passed through untouched, which is why the ball cost this
        class nothing: a target is a field on a gesture, not a new kind of
        traffic, and it must keep its place in the queue behind whatever the
        hand did before it.
        """
        self._gesture("/pet/push", body)

    def touch(self, body: dict):
        """Queue a pet. Same queue as a shove, and that is the point: they are
        two answers to the same button going up, and they must arrive in the
        order the hand made them."""
        self._gesture("/pet/touch", body)

    def sense(self, body: dict):
        """Where the pointer is now. Overwrites whatever was waiting.

        Never queued and never retried: this is sampled up to five times a
        second and the only sample worth sending is the newest one. A pointer
        position that arrives late is not late information, it is *wrong*
        information — the hand is somewhere else by now.
        """
        with self._lock:
            self._sense = dict(body)
            self._sense_dirty = True

    def _gesture(self, path: str, body: dict):
        try:
            self._gestures.put_nowait((time.monotonic(), path, dict(body)))
        except queue.Full:
            pass

    # ---------- the pick-up ----------

    def carry_start(self, x_m: float, z_m: float):
        """Close an invisible hand on the duck, wherever the pointer is."""
        self._carry_put("start", {"x_m": float(x_m), "z_m": float(z_m)})

    def carry_move(self, x_m: float, z_m: float):
        """Restate where the hand is. Also the heartbeat the daemon's deadman
        listens for, which is why the app keeps sending these even when the
        mouse is not moving."""
        self._carry_put("move", {"x_m": float(x_m), "z_m": float(z_m)})

    def carry_end(self):
        """Let go. The daemon hands the duck the hand's own velocity."""
        self._carry_put("end", {})

    def _carry_put(self, action: str, body: dict):
        """Queue one carry message, dropping a `move` before anything else.

        `start` and `end` are structural — a dropped `start` is a pick-up that
        never happened, and a dropped `end` is a duck left hanging until the
        deadman notices — so if the queue is somehow full they evict the
        oldest thing in it rather than themselves. Only `move`s can be in
        there in any number, so what gets evicted is always a stale position.
        """
        item = (time.monotonic(), action, dict(body))
        for _ in range(self._carry.maxsize + 1):
            try:
                self._carry.put_nowait(item)
                return
            except queue.Full:
                if action == "move":
                    return          # a position nobody will miss
                try:
                    self._carry.get_nowait()
                except queue.Empty:
                    pass

    # ---------- the thread ----------

    def _run(self):
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                self._send_config_if_dirty()
                self._drain_gestures()
                self._drain_carry()
                self._send_sense_if_due()
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

    def _drain_gestures(self):
        """Every shove and stroke the hand made, in the order it made them."""
        while True:
            try:
                queued_at, path, body = self._gestures.get_nowait()
            except queue.Empty:
                return
            if time.monotonic() - queued_at > PUSH_STALE_S:
                # The daemon was away when this gesture was made (the loop
                # raises before it gets here while offline). Landing it now
                # would be a shove — or a coo — out of nowhere, unrelated to
                # anything the hand on the mouse is doing at this moment.
                continue
            status, _h, data = self._request("POST", path, body)
            if status != 200:
                self._note_error(path.rsplit("/", 1)[1], status, _loads(data))

    def _drain_carry(self):
        """The pick-up, in order, with the daemon's token stamped on.

        Three rules, and each of them is a bug that would otherwise be very
        hard to see:

          * **The token is the daemon's.** It comes back on the `start` reply
            and goes onto every `move` and the `end`. A `move` or an `end`
            with a token that is no longer current gets a 409 and changes
            nothing — which is what stops a stale `end`, from a gesture the
            window server abandoned when a Space switched, releasing a grab
            the human started afterwards.
          * **`PUSH_STALE_S` applies to `move` only.** A shove made while the
            daemon was crashed is meaningless a minute later; an `end` is not
            optional at any age, and a `start` that was queued while the
            daemon was away is still describing a hand that is still down.
          * **A `move` with no token is dropped, not sent.** Between a failed
            `start` and the mouse coming up there is nothing to move.
        """
        while True:
            try:
                queued_at, action, body = self._carry.get_nowait()
            except queue.Empty:
                return
            with self._lock:
                token = self._carry_token
            if action != "start":
                if token is None:
                    continue        # no grip to talk about
                if action == "move" and time.monotonic() - queued_at > PUSH_STALE_S:
                    continue
                body = {**body, "token": token}
            body = {**body, "action": action}
            status, _h, data = self._request("POST", "/pet/carry", body)
            obj = _loads(data)
            if status != 200:
                self._note_error("carry", status, obj)
            if action == "start":
                with self._lock:
                    self._carry_token = (obj or {}).get("token")
            elif action == "end" or status == 409:
                # Let go, or told the grip was already gone. Either way this
                # feed is not holding a duck any more.
                with self._lock:
                    self._carry_token = None

    def _send_sense_if_due(self):
        """One pointer sample, if a fresh one is waiting.

        Read-and-clear under the lock, then post outside it: the UI thread
        must never block behind a socket, and a sample overwritten while this
        one is in flight simply goes next turn. No staleness check either —
        the app's own rate limit is the whole policy, and a sample that is one
        loop turn old is still the truth about where the mouse is.
        """
        with self._lock:
            if not self._sense_dirty:
                return
            body, self._sense_dirty = dict(self._sense or {}), False
        status, _h, data = self._request("POST", "/pet/sense", body)
        if status != 200:
            self._note_error("sense", status, _loads(data))

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
            # ...and it is certainly not holding the duck. The daemon's own
            # deadman released the weld 1.5 s into the outage, so a token kept
            # across it would be a key to a grip that no longer exists — and
            # the mouse-up that eventually comes would send an `end` at
            # whatever grab happened to be live by then.
            self._carry_token = None

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
