"""A fake sim behind the real HTTP contract — for testing the *window*, only.

`duck-pet --mock` boots this instead of talking to a daemon, so the parts that
are hard to get right on macOS — a borderless window that floats above the
Dock, transparent where the duck is not, that travels as the duck walks and
takes a drag — can be built and screenshotted with no sim, no MuJoCo and no
GPU anywhere in the picture.

Read the boundary carefully, because it is the whole point of putting the mock
*here* rather than in the app:

  **This module is a stand-in SIM, not a stand-in duck.** It answers the same
  routes as `webui.py`, in the same units, with the same header, and the app
  cannot tell the difference. The app contains no animation data and no
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
CURSOR_STALE_S = 2.0    # sim_server.PET_CURSOR_STALE_S
CURSOR_FLOOR_M = 0.35   # sim_server.PET_CURSOR_FLOOR_M
TOUCH_RECENT_S = 3.0    # sim_server.PET_TOUCH_RECENT_S
TOUCH_COOLDOWN_S = 2.5  # sim_server.PET_TOUCH_ACK_COOLDOWN_S
BALL_RADIUS_M = 0.035   # sim_server.BALL_RADIUS_M — the 70 mm floorball
BALL_ROLL_S = 2.2       # how fast a rolled ball bleeds off, seconds. Longer
                        # than the duck's own DRAG_S: a ball keeps going, and
                        # a mock where it did not would make the window's
                        # "chase the toy off the edge of the frame" case
                        # impossible to look at.
CARRY_TIMEOUT_S = 1.5   # sim_server.PET_CARRY_TIMEOUT_S — the deadman
CARRY_MIN_Z_M = 0.05    # sim_server.PET_CARRY_MIN_Z_M
CARRY_MAX_Z_M = 0.55    # sim_server.PET_CARRY_MAX_Z_M
CARRY_HAND_MPS = 1.5    # sim_server.PET_CARRY_HAND_SPEED_MPS
LIFT_TRIGGER_M = 0.20   # sim_server.PET_LIFT_TRIGGER_M — above this the pet
                        # camera follows the duck up and the WINDOW climbs
STAND_Z_M = 0.116       # where this crude duck's trunk sits when standing


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
        # The human. A mock duck cannot walk towards a pointer — that is a
        # machine's decision and a stand-in sim has no machine — but it has to
        # be able to say where the pointer WAS and when it was last touched,
        # because those are the two blocks the app reads back out of
        # `/pet/state`. Wall clock here rather than the sim clock the daemon
        # uses: this duck's `t0` is a wall clock too.
        self.cursor = None
        self.touches = 0
        self.touched_at = None
        self.ack_at = None
        # The toy, on rails of its own: a position, a velocity that bleeds
        # off, and a wall to stop at. It is here for the same reason the duck
        # is — the app has to be able to hit-test a ball, draw one, and shove
        # one with no MuJoCo in the room — and it is deliberately as crude as
        # the duck's sweep. What has to be RIGHT is the contract: the metres,
        # the `ball` block, and `ball_bbox` going null the instant the ball
        # leaves the frame, because that null is the case pet_app's fallback
        # arithmetic exists for.
        self.ball_x = 0.45
        self.ball_vx = 0.0
        # The hand, when there is one: `{token, t0, last, x, z}`. A stand-in
        # sim has no weld and no solver, so what it stands in for is the
        # CONTRACT — a token minted here and demanded back, a hand the duck
        # follows at a finite speed, a deadman, and a release that hands over
        # the hand's own velocity. Every one of those is a code path in the
        # app, and none of them needs MuJoCo to exercise.
        self.carry = None

    def step(self):
        with self.lock:
            now = time.time()
            dt = min(0.25, now - self.last)
            self.last = now
            # The deadman first: a hand that stopped talking is a hand that is
            # not there, and the daemon it stands in for makes exactly the
            # same call before it moves anything.
            if self.carry is not None and now - self.carry["last"] > CARRY_TIMEOUT_S:
                self._release()
            if self.carry is not None:
                self._follow_hand(dt)
            else:
                decay = math.exp(-dt / DRAG_S)
                self.x += (self.vx + self.kick_x) * dt
                self.z = max(STAND_Z_M,
                             self.z + self.kick_z * dt - 1.2 * dt * dt)
                self.kick_x *= decay
                self.kick_z = self.kick_z * decay - 4.0 * dt  # gravity, roughly
                if self.z <= STAND_Z_M:
                    self.z, self.kick_z = STAND_Z_M, max(0.0, self.kick_z)
                if self.x > self.half_span_m:
                    self.x, self.vx = self.half_span_m, -abs(self.vx)
                elif self.x < -self.half_span_m:
                    self.x, self.vx = -self.half_span_m, abs(self.vx)
            self.ball_x += self.ball_vx * dt
            self.ball_vx *= math.exp(-dt / BALL_ROLL_S)
            if abs(self.ball_x) > self.half_span_m:
                self.ball_x = math.copysign(self.half_span_m, self.ball_x)
                self.ball_vx = -0.4 * self.ball_vx      # a wall, softly
            return self.x, self.z, now - self.t0

    def push(self, dx_m, dy_m, target="duck"):
        """Same arithmetic webui does: metres of gesture × gain, clamped.

        Same gain for both targets, which is the daemon's rule too: one
        gesture vocabulary, and the difference between shoving a 700 g duck
        and a 15 g ball is the physics', not the mouse's.
        """
        with self.lock:
            vx = max(-PUSH_MAX, min(PUSH_MAX, dx_m * DRAG_GAIN))
            if target == "ball":
                self.ball_vx += vx
                return
            self.kick_x += vx
            self.kick_z += max(-PUSH_MAX, min(PUSH_MAX, -dy_m * DRAG_GAIN))

    def ball_state(self, half_frame_m, z_lo, z_hi):
        """The `ball` block, exactly the shape `_pet_ball_state` returns."""
        with self.lock:
            dx = self.ball_x - self.x
            return {"present": True, "x_m": self.ball_x, "y_m": 0.0,
                    "z_m": BALL_RADIUS_M, "dx_m": dx,
                    "radius_m": BALL_RADIUS_M,
                    "in_frame": bool(abs(dx) <= half_frame_m + BALL_RADIUS_M
                                     and z_lo - BALL_RADIUS_M <= BALL_RADIUS_M
                                     <= z_hi + BALL_RADIUS_M),
                    "vel_mps": [self.ball_vx, 0.0, 0.0]}

    def set_half_span(self, half_span_m):
        with self.lock:
            self.half_span_m = max(0.1, float(half_span_m))

    # ----- the human, answered exactly the way the daemon answers -----
    # The mock duck does not walk towards a cursor and never will: that is a
    # machine's decision, and a machine is not a thing a stand-in SIM has. But
    # the ARITHMETIC has to match, because the app rate-limits, converts and
    # rounds against these replies, and a mock that answered a different shape
    # would verify the window against a contract nothing implements.

    def sense(self, present, x_m, z_m):
        """Take a cursor sample; report what the duck would make of it."""
        with self.lock:
            now = time.time()
            if not present:
                self.cursor = None
            else:
                prev = self.cursor
                speed = None
                if prev is not None and now > prev["t"]:
                    speed = abs(x_m - prev["x_m"]) / (now - prev["t"])
                self.cursor = {"x_m": x_m, "z_m": z_m, "t": now,
                               "speed_mps": speed}
            return self._cursor_state()

    def _cursor_state(self):
        blank = {"present": False, "x_m": None, "z_m": None, "dx_m": None,
                 "dist_m": None, "age_s": 999.0, "near_floor": False,
                 "speed_mps": None}
        c = self.cursor
        if c is None:
            return blank
        age = round(time.time() - c["t"], 3)
        if age > CURSOR_STALE_S:
            return {**blank, "age_s": age}
        dx = c["x_m"] - self.x
        return {"present": True, "x_m": c["x_m"], "z_m": c["z_m"],
                "dx_m": dx, "dist_m": abs(dx), "age_s": age,
                "near_floor": bool(c["z_m"] <= CURSOR_FLOOR_M),
                "speed_mps": c["speed_mps"]}

    def touch(self):
        """A pet: tally it, and answer once per cooldown the way the sim does."""
        with self.lock:
            now = time.time()
            self.touches += 1
            self.touched_at = now
            if self.ack_at is not None and now - self.ack_at < TOUCH_COOLDOWN_S:
                return {"acknowledged": False, "emote": None, "sound": None,
                        "note": "still enjoying the last one",
                        "count": self.touches, "cooldown_s": TOUCH_COOLDOWN_S}
            self.ack_at = now
            return {"acknowledged": True, "emote": "nuzzle", "sound": "coo",
                    "note": "", "count": self.touches,
                    "cooldown_s": TOUCH_COOLDOWN_S}

    def _touch_state(self):
        t = self.touched_at
        age = 999.0 if t is None else round(time.time() - t, 3)
        return {"petted": bool(t is not None and age < TOUCH_RECENT_S),
                "age_s": age, "count": self.touches}

    # ----- the pick-up, kept to the contract and nothing else -----
    # No weld and no solver here. What a stand-in sim owes the app is the
    # SHAPE of a carry: a token it did not choose, a hand the duck follows at
    # a finite speed rather than teleporting to, a clamp, a deadman, and a
    # release that hands the duck the hand's own velocity. The flailing
    # standing policy that makes the real thing look alive is not something a
    # paper cutout can have, and pretending otherwise would be the one kind of
    # lie this file exists to avoid.

    def carry_start(self, x_m, z_m):
        with self.lock:
            if self.carry is not None:
                return None                       # 409: somebody has it
            now = time.time()
            x, z = self._carry_clamp(x_m, z_m)
            self.carry = {"token": f"{int(now * 1000) % 0xFFFFFFFF:08x}",
                          "t0": now, "last": now, "x": x, "z": z,
                          "vx": 0.0, "vz": 0.0}
            return dict(self.carry)

    def carry_move(self, token, x_m, z_m):
        with self.lock:
            if self.carry is None or self.carry["token"] != token:
                return None                       # 409: not the current grip
            if x_m is not None and z_m is not None:
                self.carry["x"], self.carry["z"] = self._carry_clamp(x_m, z_m)
            self.carry["last"] = time.time()
            return dict(self.carry)

    def carry_end(self, token):
        with self.lock:
            if self.carry is None or self.carry["token"] != token:
                return None
            return self._release()

    def _carry_clamp(self, x_m, z_m):
        return (max(-self.half_span_m, min(self.half_span_m, float(x_m))),
                max(CARRY_MIN_Z_M, min(CARRY_MAX_Z_M, float(z_m))))

    def _follow_hand(self, dt):
        """Chase the hand at a finite speed — never teleport to it."""
        c = self.carry
        step = CARRY_HAND_MPS * dt
        for axis, want in (("x", c["x"]), ("z", c["z"])):
            here = getattr(self, axis)
            delta = want - here
            moved = math.copysign(min(abs(delta), step), delta)
            setattr(self, axis, here + moved)
            c["v" + axis] = moved / dt if dt > 0 else 0.0

    def _release(self):
        """Open the hand; the duck keeps whatever the hand was doing."""
        c, self.carry = self.carry, None
        if c is None:
            return None
        self.kick_x += max(-PUSH_MAX, min(PUSH_MAX, c["vx"]))
        self.kick_z += max(-PUSH_MAX, min(PUSH_MAX, c["vz"]))
        return {**c, "released_vel_mps": [c["vx"], 0.0, c["vz"]]}

    def _carry_state(self):
        c = self.carry
        if c is None:
            return {"carried": False, "token": None, "held_s": 0.0,
                    "hand_m": None}
        return {"carried": True, "token": c["token"],
                "held_s": round(time.time() - c["t0"], 3),
                "hand_m": [self.x, 0.0, self.z]}

    def frame_floor_z_m(self):
        """What world height the frame's floor row is showing — the daemon's
        `_pet_lift_m`, and 0 for every duck that is on the Dock."""
        with self.lock:
            return max(0.0, self.z - LIFT_TRIGGER_M)


def render(px: int, t: float, view_m: float, floor_pad_px: int,
           facing: int, lift_m: float, ball_dx_m: float = None) -> bytes:
    """A duck-shaped hole in a transparent square, `px` on a side.

    Returns `(png, ball_bbox)` — the toy's box in frame pixels, top-left
    origin, half-open, or None when the ball is not in this picture at all.
    That second value is the mock keeping the daemon's contract rather than
    the duck's: `_handle_pet_frame` sends `bbox` and `ball_bbox` separately
    because the overlay hit-tests them separately, and a mock that only ever
    said "there is no ball box" would leave that path unexercised.
    """
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
    # The toy, if it is close enough to be in this picture. `ball_dx_m` is
    # the ball's offset from the DUCK, because the duck is what the frame is
    # centred on — which is also why the box goes null rather than clamping:
    # the daemon's segmentation mask has nothing to report about a ball that
    # is not drawn, and neither has this.
    ball_bbox = None
    if ball_dx_m is not None:
        br = BALL_RADIUS_M * m2px
        bcx = cx + ball_dx_m * m2px
        bcy = px - floor_pad_px - BALL_RADIUS_M * m2px
        if -br <= bcx <= px + br:
            d.ellipse([bcx - br, bcy - br, bcx + br, bcy + br],
                      fill=(255, 140, 0, 255))
            x0 = max(0, int(math.floor(bcx - br)))
            y0 = max(0, int(math.floor(bcy - br)))
            x1 = min(px, int(math.ceil(bcx + br)))
            y1 = min(px, int(math.ceil(bcy + br)))
            if x1 > x0 and y1 > y0:
                ball_bbox = [x0, y0, x1, y1]
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=1)
    return buf.getvalue(), ball_bbox


def start_mock(port: int) -> ThreadingHTTPServer:
    """Serve /pet/{frame,state,push,config,sense,touch,carry} on loopback,
    from a daemon thread."""
    duck = MockDuck()
    cfg = {"px_per_meter": 1312.0, "frame_px": 512, "supersample": 2,
           "screen_width_m": 2.634, "floor_pad_px": pet_map.DEFAULT_FLOOR_PAD_PX,
           "wall_margin_m": 512 / (2 * 1312.0),
           "ball_radius_m": BALL_RADIUS_M,
           "carry_max_z_m": CARRY_MAX_Z_M,
           "carry_hand_speed_mps": CARRY_HAND_MPS,
           "lift_trigger_m": LIFT_TRIGGER_M,
           "camera_distance_m": 1.0, "azimuth_deg": 90.0, "elevation_deg": 0.0}

    def wall_m():
        return max(0.05, cfg["screen_width_m"] / 2 - cfg["wall_margin_m"])

    def frame_span():
        """(half the frame's width, its bottom edge, its top edge) in metres —
        the daemon's `_pet_frame_z_span` plus the horizontal half, which is
        what `ball.in_frame` is decided against on both sides. Both vertical
        edges carry the camera's lift, because the frame's floor is not world
        z=0 once the duck has been picked up off it."""
        lift = duck.frame_floor_z_m()
        return (0.5 * cfg["frame_px"] / cfg["px_per_meter"],
                lift - cfg["floor_pad_px"] / cfg["px_per_meter"],
                lift + (cfg["frame_px"] - cfg["floor_pad_px"])
                / cfg["px_per_meter"])

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
                    "cursor": duck._cursor_state(),
                    "touch": duck._touch_state(),
                    "carry": duck._carry_state(),
                    "ball": duck.ball_state(*frame_span()),
                    "screen": {"center_offset_px": x * cfg["px_per_meter"],
                               "floor_px_from_bottom": cfg["floor_pad_px"],
                               "px_per_meter": cfg["px_per_meter"],
                               # Where the frame's floor row is, in world z:
                               # 0 for a duck on the Dock, and how far the
                               # WINDOW has to climb once one is lifted.
                               "frame_floor_z_m": duck.frame_floor_z_m(),
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
                state = self._state(x, z, t)
                # How high the duck is drawn INSIDE its frame: its height off
                # the floor, less whatever the camera already took by lifting
                # the frame. Past the trigger the two cancel and the duck
                # stays put in the picture while the window climbs — which is
                # exactly what the daemon's camera does.
                png, ball_bbox = render(
                    size, t, size / cfg["px_per_meter"],
                    round(cfg["floor_pad_px"] * size / cfg["frame_px"]),
                    1 if duck.vx >= 0 else -1,
                    z - STAND_Z_M - state["screen"]["frame_floor_z_m"],
                    ball_dx_m=state["ball"]["dx_m"])
                # The frame's two boxes ride the header beside the pose, the
                # way the daemon sends them. `bbox` is left absent rather than
                # faked: the app falls back to its nominal standing box, which
                # is a path worth exercising too.
                state["ball_bbox"] = ball_bbox
                self._send(200, png, "image/png",
                           {"X-Duck-Pet": json.dumps(state,
                                                     separators=(",", ":"))})
            elif path == "/pet/state":
                self._json(self._state(x, z, t))
            else:
                self._json({"ok": False, "error": "not found"}, 404)

        def _carry(self, body):
            """`/pet/carry`, refused the same three ways the daemon refuses it.

            The 409 is the one that matters here: `pet_feed` clears its token
            on one, and the app tears its gesture down when the feed says it
            is no longer carrying anything. Neither of those paths would ever
            run against a mock that answered 200 to everything.
            """
            action = str(body.get("action", ""))
            if action not in ("start", "move", "end"):
                self._json({"ok": False, "error":
                            f"unknown carry action {action!r} — choose from "
                            "start, move, end"}, 400)
                return
            x_m, z_m = body.get("x_m"), body.get("z_m")
            if action == "start":
                got = duck.carry_start(0.0 if x_m is None else float(x_m),
                                       STAND_Z_M if z_m is None else float(z_m))
                if got is None:
                    self._json({"ok": False, "conflict": True, "error":
                                "the duck is already in somebody's hand"}, 409)
                    return
                released = None
            elif action == "move":
                got = duck.carry_move(body.get("token"), x_m, z_m)
                released = None
            else:
                got = duck.carry_end(body.get("token"))
                released = (got or {}).get("released_vel_mps")
            if got is None:
                self._json({"ok": False, "conflict": True, "error":
                            "not the current carry (token expired — the hand "
                            "was released)"}, 409)
                return
            state = duck._carry_state()
            self._json({"ok": True, "action": action, "note": "",
                        "token": got["token"] if action != "end" else None,
                        "carried": state["carried"], "held_s": state["held_s"],
                        "hand_m": state["hand_m"],
                        "target_m": [got["x"], got["z"]],
                        "limits": {"x_m": [-duck.half_span_m, duck.half_span_m],
                                   "z_m": [CARRY_MIN_Z_M, CARRY_MAX_Z_M],
                                   "hand_speed_mps": CARRY_HAND_MPS,
                                   "timeout_s": CARRY_TIMEOUT_S},
                        "released_vel_mps": released})

        def do_POST(self):
            path = self.path.partition("?")[0]
            body = self._body()
            if path == "/pet/push":
                target = str(body.get("target", "duck"))
                if target not in ("duck", "ball"):
                    self._json({"ok": False, "error":
                                f"unknown push target {target!r} — choose "
                                "from duck, ball"}, 400)
                    return
                duck.push(float(body.get("dx_m", 0.0)),
                          float(body.get("dy_m", 0.0)), target)
                self._json({"ok": True, "target": target, "pushed": body})
            elif path == "/pet/sense":
                # Validated the way webui does it, and refused the same way:
                # the app's rate limiter and its `present: False` message are
                # both exercised against this reply, so a mock that accepted
                # nonsense would hide the one bug worth catching here.
                if not body.get("present", True):
                    self._json({"ok": True, "cursor": duck.sense(False, 0, 0)})
                    return
                try:
                    x_m, z_m = float(body["x_m"]), float(body["z_m"])
                except (KeyError, TypeError, ValueError):
                    self._json({"ok": False,
                                "error": "x_m and z_m must be numbers"}, 400)
                    return
                self._json({"ok": True, "cursor": duck.sense(True, x_m, z_m)})
            elif path == "/pet/touch":
                kind = str(body.get("kind", "pet"))
                if kind != "pet":
                    self._json({"ok": False,
                                "error": f"unknown touch {kind!r} — this duck "
                                         f"understands pet"}, 400)
                    return
                # A pet never reaches the physics — no `duck.push`, no kick,
                # nothing. That absence is the contract, and the mock keeps it
                # for the same reason the daemon does.
                self._json({"ok": True, "kind": kind, **duck.touch()})
            elif path == "/pet/carry":
                self._carry(body)
            elif path == "/pet/config":
                for k in ("px_per_meter", "frame_px", "supersample",
                          "floor_pad_px", "wall_margin_m", "camera_distance_m",
                          "carry_max_z_m"):
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
