"""The screen ↔ sim mapping for the desktop pet — pure arithmetic, no Cocoa.

Everything the overlay knows about *where the duck is* lives here, in plain
numbers, so it can be tested on any machine (`tests/test_pet_app.py`) and read
by anyone arguing about the geometry without booting a window server.

The whole trick, in one paragraph. The sim floor is the ordinary infinite
ground plane; we pin it to the Dock's top edge. An orthographic side-on camera
tracks the duck, so the duck sits dead centre of every rendered frame and never
moves within it — **the window travels instead**, at `px_per_meter` points per
metre of sim x. Two invisible walls at the mapped screen edges are what keep it
on screen, and the sim places them from the numbers this module sends.

    sim x = 0   ->  centre of the usable screen band
    sim x = +X  ->  window centre at  centre + X * px_per_meter
    sim floor   ->  Cocoa y = visibleFrame.origin.y   (the Dock's top edge)

`px_per_meter` is chosen from a *desired duck height in points* rather than
typed in: a pet is 180 pt tall, and 180 / 0.2744 m = 656 pt/m falls out. Say
the size you want and the scale follows.

**Two units, kept honest.** Cocoa places windows in points; the sim renders
device pixels. Every number this module sends the daemon is in device pixels
(`px_per_meter` included), because `frame_px` is the *output image* and a frame
that is not device-pixel sized is a frame macOS has to resample. Every number
it hands Cocoa is in points. `backing_scale` is the only bridge, and it appears
in exactly the handful of places named below. Gestures dodge the question
entirely by travelling in metres — `dx_m`/`dy_m`, which mean the same thing on
both sides of the socket.
"""

import math
from dataclasses import dataclass, replace

# Measured off the model at the STAND keyframe (every group-0/2 mesh vertex
# transformed to world, floor and the unposed mouth mocap plate excluded):
# floor -> crown 0.2744 m, body 0.1421 m wide, 0.1227 m deep.
DUCK_HEIGHT_M = 0.2744
DUCK_WIDTH_M = 0.1421
# The one that matters for the overlay: the pet camera looks along world y, so
# what the screen sees across is the duck's **x depth**, not its y width. A
# Microduck read side-on is tall and narrow — 0.123 m across, 0.274 m up.
DUCK_DEPTH_M = 0.1227

# A pet, not a kaiju.
DEFAULT_DUCK_PT = 180.0
DEFAULT_WINDOW_PT = 300.0

# The toy, mirrored from sim_server.BALL_RADIUS_M. Only a fallback: every pose
# the daemon sends carries `ball.radius_m`, and `POST /pet/config` echoes
# `ball_radius_m` besides. It is here so the app can size a grab box before
# the first pose lands, and it is pinned to the daemon by the mirror test in
# tests/test_pet_app.py — a ball drawn at the wrong size would be a ball you
# reach for and miss.
BALL_RADIUS_M = 0.035

# How far in from the edge of the band the invisible walls stand, in metres of
# sim floor. One duck-depth, and both halves of that are load-bearing:
#
#   NOT LESS, or the duck leans on a wall it never meant to reach.
#     machines/pet.toml turns itself around at |x| = 1.00 m and then *coasts*
#     through a 4 s point turn to a measured worst case of 1.063 m. A wall
#     inside 1.063 + half a duck (0.061) = 1.124 m gets leaned on every lap.
#     The daemon's own default — half a window, 1.088 m here — is exactly
#     inside that, which is why the app pins the margin instead of letting it
#     track the window size.
#   NOT MORE, or a shove into a wall parks the duck half off the screen.
#     The duck's beak reaches 0.075 m past its base, so a wall at
#     half_span - DUCK_DEPTH_M leaves the whole silhouette inside the band
#     with room to spare (1.194 + 0.075 = 1.269 < 1.317 m on this display).
#
# The *window* may still overhang the bezel at the wall; it is transparent and
# nobody can tell. The duck may not.
WALL_MARGIN_M = DUCK_DEPTH_M

# Mirrored from sim_server.py (PET_FRAME_MIN_PX / PET_FRAME_MAX_PX /
# PET_OFFSCREEN_MAX_PX / PET_FLOOR_PAD_PX). Duplicated on purpose: the app has
# to know what the daemon will silently clamp *before* it sizes a window around
# it, and a window that does not match the frame inside it is the one bug you
# cannot see in a screenshot. The daemon still echoes back what it accepted
# (`adopt`), so if these ever drift the daemon wins on the next config round.
PET_FRAME_MIN_PX = 32
PET_FRAME_MAX_PX = 512
PET_OFFSCREEN_MAX_PX = 1024
# Output pixels between the sim floor line and the bottom edge of the frame: a
# little landing room under the feet so a stumble is not clipped by the window.
DEFAULT_FLOOR_PAD_PX = 26

# What the daemon turns a drag into: metres of gesture × PET_DRAG_GAIN =
# metres per second of shove, clamped to ±2 (webui.PET_DRAG_GAIN,
# sim_server.PET_PUSH_MAX). Mirrored only to size the poke; the app sends
# metres and lets the daemon own the conversion.
PET_DRAG_GAIN = 6.0

# A click with no drag in it. 0.075 m of gesture ≈ 0.45 m/s of shove: enough to
# make the walk policy visibly catch itself, nowhere near enough to topple it.
POKE_M = 0.075
# Slop below which a drag is really a click (points).
CLICK_SLOP_PT = 4.0
# A flick is measured over the tail of the gesture, not its whole length —
# otherwise a slow drag with a fast finish reads as slow.
FLICK_WINDOW_S = 0.12

# ---------- what separates a hand from a shove ----------
# Several gestures share one mouse button, and the split is entirely in the
# numbers below. All of them are in POINTS and SECONDS, because that is what a
# hand does; the metres only appear once the gesture has been named.
#
#   click   travel < CLICK_SLOP_PT              -> poke  (a nudge away)
#   stroke  slow, small, ends on the duck       -> pet   (no insult at all)
#   flick   anything faster                     -> push  (a real shove)
#
# The stroke thresholds are derived from the poke rather than guessed. The
# daemon turns the last FLICK_WINDOW_S of a drag into metres and multiplies by
# PET_DRAG_GAIN, so a 49 pt tail (0.075 m at 656 pt/m) is exactly a poke's
# 0.45 m/s. A gentle hand must land UNDER that, hence 40 pt.
PET_MIN_S = 0.15            # a stroke takes time; anything quicker is a flick
PET_MAX_TAIL_PT = 40.0      # travel in the last FLICK_WINDOW_S -> < 0.37 m/s
# NET DISPLACEMENT, start to finish — not path length, and the difference is
# the whole point: a stroke goes back and forth over the same duck and its net
# displacement is near zero, which is exactly what a petting hand should look
# like. (Path length is not available to measure anyway: `pet_app.sampleDrag_`
# keeps only the last 24 samples, so a long gesture's early travel is gone by
# the time anyone asks.) About two thirds of a duck.
PET_MAX_TRAVEL_PT = 120.0

# A press that never moves is a fourth gesture — a hold, which the pick-up
# lane turns into a carry. Six points is one Retina pixel-pair of tremor; more
# than that and the hand is dragging, and this press can never become a hold
# no matter how long it lasts. `pet_app.sampleDrag_` latches that the moment
# it happens, because the sampler is the only thing that sees every mouse
# move — a promotion decided from anywhere else would be deciding about a hand
# it did not watch.
CARRY_SLOP_PT = 6.0
# ...and how long it has to stay still. Long enough that a slow click is not a
# pick-up, short enough that nobody has to wonder whether it worked. The
# promotion runs off the UI tick and NOT off the drag callback, which is the
# single easiest thing in this feature to get wrong: `mouseDragged_` does not
# fire when the mouse does not move, so a perfectly still hold is a stream of
# exactly zero events.
CARRY_HOLD_S = 0.30
# A hold that went nowhere and let go straight away was never a pick-up — it
# was a hand resting on a duck. Those get a pet instead, which is why these
# two are here rather than in the classifier: by the time they are asked, the
# gesture is already known to have been a carry.
CARRY_STILL_PT = 8.0
CARRY_TAP_S = 0.80
# How often the hand's position is restated while the duck hangs off it. Also
# the heartbeat the daemon's deadman listens for (sim_server's
# PET_CARRY_TIMEOUT_S is 1.5 s), so it has to keep firing even when the mouse
# is perfectly still — silence means the overlay died, and a dead overlay must
# not leave a duck in mid-air.
CARRY_HZ = 20.0

# The cursor channel. Five a second while moving, one a second at rest — the
# duck has to notice a hand that has PARKED next to it, not just one in
# flight, and a sample that goes stale (sim_server.PET_CURSOR_STALE_S, 2 s)
# would make it lose interest in a pointer that is plainly still there.
SENSE_HZ = 5.0
SENSE_MIN_M = 0.02
SENSE_HEARTBEAT_S = 1.0

# Mirrored from sim_server.PET_LIFT_TRIGGER_M — the trunk height above which
# the daemon's camera starts following the duck upwards, and therefore above
# which the WINDOW starts climbing the screen (see `ScreenMap.window_origin`).
# The app never computes it: the daemon reports the lift it actually applied
# as `pet_state.screen.frame_floor_z_m`, and this is here so the mirror test
# catches the day the two stop agreeing.
PET_LIFT_TRIGGER_M = 0.20


def ppm_for_duck_pt(duck_pt: float = DEFAULT_DUCK_PT) -> float:
    """Points per metre that makes the duck stand `duck_pt` points tall."""
    return float(duck_pt) / DUCK_HEIGHT_M


def frame_px_for(window_pt: float, backing_scale: float) -> int:
    """The device-pixel frame that fills `window_pt`, within the daemon's cap.

    512 px is the daemon's ceiling, so on a 2× display the largest crisp window
    is 256 pt. That trades window *box* for window *sharpness*, which is the
    right way round: the duck's own size is set by `px_per_meter` and does not
    change — only how much air there is around it.
    """
    want = int(round(float(window_pt) * float(backing_scale)))
    return max(PET_FRAME_MIN_PX, min(PET_FRAME_MAX_PX, want))


def supersample_for(frame_px: int) -> int:
    """Most antialiasing the daemon's 1024 px offscreen buffer will allow."""
    for ss in (3, 2, 1):
        if frame_px * ss <= PET_OFFSCREEN_MAX_PX:
            return ss
    return 1


@dataclass(frozen=True)
class ScreenMap:
    """One screen's worth of mapping. Immutable; rebuild it on screen changes.

    `left_pt` / `width_pt` come from the screen's **visibleFrame**, not its
    frame: a Dock on the left or right eats horizontal room and the duck should
    walk the band that is actually free. `floor_y_pt` is `visibleFrame.origin.y`
    — the Dock's top edge, which is ~0 when the Dock is auto-hidden, and the
    blueprint's fallback (walk the absolute bottom edge) then falls out for
    free rather than needing a branch.
    """

    left_pt: float
    width_pt: float
    floor_y_pt: float
    screen_h_pt: float
    px_per_meter: float           # POINTS per metre — the Cocoa-side scale
    backing_scale: float = 2.0
    frame_px: int = 512           # DEVICE pixels: the image the daemon renders
    supersample: int = 2
    floor_pad_px: int = DEFAULT_FLOOR_PAD_PX   # device px, floor above bottom

    # ---------- construction ----------

    @classmethod
    def from_screen(cls, frame, visible, *, duck_pt=DEFAULT_DUCK_PT,
                    px_per_meter=None, window_pt=DEFAULT_WINDOW_PT,
                    frame_px=None, floor_pad_px=DEFAULT_FLOOR_PAD_PX,
                    backing_scale=2.0) -> "ScreenMap":
        """Build from `(x, y, w, h)` tuples of NSScreen frame and visibleFrame."""
        ppm = float(px_per_meter) if px_per_meter else ppm_for_duck_pt(duck_pt)
        px = int(frame_px) if frame_px else frame_px_for(window_pt, backing_scale)
        px = max(PET_FRAME_MIN_PX, min(PET_FRAME_MAX_PX, px))
        return cls(left_pt=float(visible[0]), width_pt=float(visible[2]),
                   floor_y_pt=float(visible[1]), screen_h_pt=float(frame[3]),
                   px_per_meter=ppm, backing_scale=float(backing_scale),
                   frame_px=px, supersample=supersample_for(px),
                   floor_pad_px=int(floor_pad_px))

    def adopt(self, cfg) -> "ScreenMap":
        """Take the daemon's word for what it actually accepted.

        `POST /pet/config` echoes a `config` block, and so does every pose the
        frame stream carries, so the app is re-synchronised on every frame for
        free. It matters: the daemon clamps `frame_px`, may drop `supersample`
        to fit its offscreen buffer, and owns `floor_pad_px` — and a window
        sized for numbers the daemon rejected draws the duck in the wrong place
        forever. Its `px_per_meter` is in device pixels; ours is in points.
        """
        if not isinstance(cfg, dict):
            return self
        out = self
        px = _num(cfg.get("frame_px"))
        if px and PET_FRAME_MIN_PX <= px <= PET_FRAME_MAX_PX:
            out = replace(out, frame_px=int(px))
        ss = _num(cfg.get("supersample"))
        if ss and 1 <= ss <= 3:
            out = replace(out, supersample=int(ss))
        pad = _num(cfg.get("floor_pad_px"))
        if pad is not None and 0 <= pad <= PET_FRAME_MAX_PX:
            out = replace(out, floor_pad_px=int(pad))
        ppm = _num(cfg.get("px_per_meter"))
        if ppm and ppm > 0:
            out = replace(out, px_per_meter=ppm / out.backing_scale)
        return out

    # ---------- the mapping ----------

    @property
    def center_x_pt(self) -> float:
        return self.left_pt + 0.5 * self.width_pt

    @property
    def window_pt(self) -> float:
        """The overlay window's edge, in points: the frame at native density."""
        return self.frame_px / self.backing_scale

    @property
    def ground_pt(self) -> float:
        """Points of frame below the floor line — how far the window hangs."""
        return self.floor_pad_px / self.backing_scale

    @property
    def device_ppm(self) -> float:
        """Device pixels per metre: what the daemon is told, and only that."""
        return self.px_per_meter * self.backing_scale

    @property
    def duck_pt(self) -> float:
        """How tall the duck stands, in points."""
        return DUCK_HEIGHT_M * self.px_per_meter

    @property
    def span_m(self) -> float:
        """How much sim floor the usable screen band is worth."""
        return self.width_pt / self.px_per_meter

    @property
    def half_span_m(self) -> float:
        """Where the invisible walls stand: one duck-depth in from the edge of
        the band, for the two reasons spelled out at WALL_MARGIN_M. The daemon
        reports the truth in `config.walls_m`; this is what the app asks for,
        and what it draws with until it has heard back."""
        return max(0.05, 0.5 * self.span_m - WALL_MARGIN_M)

    def screen_x_pt(self, x_m: float) -> float:
        """Sim x (metres) -> the Cocoa x the duck's centre line should be at."""
        return self.center_x_pt + float(x_m) * self.px_per_meter

    def x_m_from_screen_pt(self, x_pt: float) -> float:
        """The inverse, for turning a cursor position back into sim space."""
        return (float(x_pt) - self.center_x_pt) / self.px_per_meter

    def screen_y_pt(self, z_m: float) -> float:
        """Sim height above the floor (metres) -> a Cocoa y on this screen."""
        return self.floor_y_pt + float(z_m) * self.px_per_meter

    def z_m_from_screen_pt(self, y_pt: float) -> float:
        """A Cocoa y -> how high that is above the duck's floor, in metres.

        The inverse of `screen_y_pt`, and exact for the same reason the x
        mapping is: the pet camera is orthographic, so metres-per-point is one
        number at every depth. The Dock's top edge is z = 0 — negative is
        *inside* the Dock, which is where the pointer is whenever it is on a
        Dock icon, and the duck is entitled to know that.
        """
        return (float(y_pt) - self.floor_y_pt) / self.px_per_meter

    def window_origin(self, x_m: float, floor_z_m: float = 0.0) -> tuple:
        """Bottom-left of the pet window for a duck at sim x, in Cocoa points.

        The window is centred on the duck horizontally and hung so that the
        frame's floor row lands exactly on the Dock's top edge. Deliberately
        *not* clamped to the screen: at the walls the window may overhang a
        little, the duck inside it does not, and clamping would slide the duck
        off its own centre line and break the illusion that the window is a
        camera rather than a box.

        `floor_z_m` is what world height that floor row is showing, and it is
        0 for every duck standing on the Dock — which is every duck that is
        not being carried. Past `PET_LIFT_TRIGGER_M` the daemon's camera
        follows the duck upwards and reports the lift it took
        (`pet_state.screen.frame_floor_z_m`); the window then has to climb the
        screen by exactly the same amount, or the picture rises inside a
        window that stayed put and the duck appears to shrink into its own
        frame. The default keeps every caller written before the pick-up
        existed byte-identical.
        """
        return (self.screen_x_pt(x_m) - 0.5 * self.window_pt,
                self.floor_y_pt + float(floor_z_m) * self.px_per_meter
                - self.ground_pt)

    # ---------- what the daemon needs to hear ----------

    def config_payload(self) -> dict:
        """The body of `POST /pet/config` — all of it in device pixels.

        This is the whole negotiation: the app knows the screen, the daemon
        knows the physics. `wall_margin_m` is the one number the app pins
        rather than leaves to the daemon: left out, the margin tracks the
        window size, and a window big enough to be worth looking at puts the
        walls inside the arc the machine turns around in (see WALL_MARGIN_M).
        `carry_max_z_m` is the second number the app pins, and for the same
        shape of reason: the daemon knows the physics of a weld and has no
        idea how tall the display is. A duck carried above the menu bar has
        left the world it lives in, so the ceiling is "the screen above the
        walk line, less a duck" — measured from `screen_h_pt`, which is the
        one thing only this side can see. Clamped into the daemon's own bounds
        so a silly screen is a smaller carry rather than a refused config.

        Deliberately *not* sent: every camera key — azimuth, elevation,
        distance and the orthographic view height are the renderer's business,
        and the app has no opinion it could defend. `lift_trigger_m` is on
        that list too: how high a duck has to be before the camera follows it
        is a claim about the FRAME, and the frame is the daemon's.
        """
        headroom = (self.screen_h_pt - self.floor_y_pt) / self.px_per_meter
        return {
            "px_per_meter": round(self.device_ppm, 4),
            "frame_px": self.frame_px,
            "supersample": self.supersample,
            "screen_width_px": round(self.width_pt * self.backing_scale, 2),
            "floor_pad_px": self.floor_pad_px,
            "wall_margin_m": round(WALL_MARGIN_M, 4),
            "carry_max_z_m": round(max(0.10, min(3.0,
                                                 headroom - DUCK_HEIGHT_M)), 4),
        }


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------- pointing device -> a real shove ----------

def drag_to_push(dx_pt: float, dy_pt: float, px_per_meter: float) -> dict:
    """A drag in Cocoa points -> the body of `POST /pet/push`.

    Sent in **metres**, not pixels: the daemon accepts either, and metres are
    the one unit that cannot be wrong about whose pixels we meant. It multiplies
    by `PET_DRAG_GAIN` and clamps to ±2 m/s, so the app does no clamping of its
    own — asking for a shove and being told how much of it landed is more
    honest than pre-trimming the gesture.

    Sign convention is the endpoint's: **+dy is DOWNWARDS**. Cocoa's +y is up,
    hence the flip. Up the screen is world +z in the side view, so a flick
    upwards genuinely lifts the duck off the Dock — no depth fudge, no third
    axis invented to make the mouse feel three-dimensional.
    """
    return {"dx_m": float(dx_pt) / px_per_meter,
            "dy_m": -float(dy_pt) / px_per_meter}


def poke_to_push(x_in_window_pt: float, window_pt: float,
                 center_pt: float = None) -> dict:
    """A click with no drag -> the body of `POST /pet/push`.

    Pokes push the thing you poked *away from your finger*: hit its left flank
    and it scoots right. Which side of the THING's centre you clicked is the
    whole question — and for the duck that is the middle of the window, since
    the duck is always centred in its own frame, which is why `center_pt`
    defaults to exactly that.

    The ball is not centred in anything: it is wherever it rolled to. So a
    poke at the toy passes the ball's own centre and the same rule then reads
    correctly for it — poke its right side and it rolls left, which is what a
    poke means when you are looking at it.
    """
    mid = 0.5 * float(window_pt) if center_pt is None else float(center_pt)
    left = float(x_in_window_pt) < mid
    return {"dx_m": POKE_M if left else -POKE_M, "dy_m": 0.0}


def is_click(dx_pt: float, dy_pt: float) -> bool:
    """Did the hand move enough for this to have been a drag at all?"""
    return math.hypot(float(dx_pt), float(dy_pt)) < CLICK_SLOP_PT


def push_speed_mps(push: dict, gain: float = PET_DRAG_GAIN) -> float:
    """What the daemon will make of that gesture, for the verbose log."""
    speed = math.hypot(push.get("dx_m", 0.0), push.get("dy_m", 0.0)) * gain
    return min(speed, 2.0)


# ---------- pointing device -> a hand the duck can notice ----------

def screen_to_sim_m(smap: ScreenMap, x_pt: float, y_pt: float) -> dict:
    """A point on the screen, in the duck's own metres.

    Both axes, and both exact: sim x from the centre of the usable band, sim z
    up from the Dock's top edge. This is the one conversion the cursor channel
    is made of, kept out of the app so it can be tested without a mouse.
    """
    return {"x_m": smap.x_m_from_screen_pt(x_pt),
            "z_m": smap.z_m_from_screen_pt(y_pt)}


def cursor_payload(smap: ScreenMap, x_pt: float, y_pt: float) -> dict:
    """The body of `POST /pet/sense` for a pointer at this screen point.

    `present` is always True here — the app posts `{"present": False}` on its
    own when the pointer leaves this screen, which is a statement about the
    *absence* of a position rather than a position.
    """
    return {**screen_to_sim_m(smap, x_pt, y_pt), "present": True}


def touch_payload(smap: ScreenMap, x_pt: float, y_pt: float,
                  base_x_m: float = 0.0, duration_s: float = 0.0,
                  travel_pt: float = 0.0) -> dict:
    """The body of `POST /pet/touch` — where on the animal the hand was.

    `x_m` is relative to the duck's own base rather than to the screen: a pet
    on the beak and a pet on the tail are different facts about the duck, and
    the same two screen points mean different things a second later when it
    has walked on. `z_m` stays absolute (height above the floor), because the
    duck's feet are always on the floor line and there is nothing to subtract.
    Both are recorded rather than acted on today; they are what a future
    "scratch under the chin" would read.
    """
    here = screen_to_sim_m(smap, x_pt, y_pt)
    return {"kind": "pet",
            "x_m": here["x_m"] - float(base_x_m),
            "z_m": here["z_m"],
            "duration_s": round(float(duration_s), 3),
            "travel_m": abs(float(travel_pt)) / smap.px_per_meter}


def classify_release(dt_s: float, total_dx_pt: float, total_dy_pt: float,
                     tail_dx_pt: float, tail_dy_pt: float) -> str:
    """What that gesture was: `'poke'`, `'pet'` or `'push'`.

    The ONLY place the split is decided, deliberately: it is pure arithmetic
    over five numbers, so the whole vocabulary of the mouse can be tested
    without a window server, and the app is left holding nothing but "when did
    the button go down and where has it been since". Every gesture judged here
    BEGAN on the duck — `pet_app.release_kind` answers for the ball and the
    carry before this is consulted.

    The three cases, in the order they are asked:

      * A gesture that was BRIEF and never went anywhere is a **poke** — a
        click, and the one gesture whose direction comes from where in the
        window it landed rather than from any movement. Both halves are
        load-bearing, and the brevity half was learned the hard way: the
        distance is a NET displacement, and the one gesture in this app that
        reliably ends where it started is a petting stroke going back and
        forth over the same duck. Without the clock, a 0.9 s stroke that
        happened to finish within 4 pt of its origin was answered with a
        0.45 m/s shove of the duck the human was stroking.
      * A gesture that took its time (`PET_MIN_S`), stayed small
        (`PET_MAX_TRAVEL_PT`) and finished slowly (`PET_MAX_TAIL_PT` over the
        flick window) is a **pet**. A slow press that never moved at all lands
        here too, which is the right reading of a hand resting on a duck (and
        the same one `pet_app.endDrag_` gives a carry that let go straight
        away). Where the hand was at RELEASE is deliberately not asked: the
        duck keeps walking while it is stroked, and on the first live test
        every honest pet ended a few points off a silhouette that had walked
        on — demoted to a 0.02 m/s shove of the very animal being petted. A
        hand that is actually throwing the duck fails the tail test instead.
      * Everything else is a **push** — it moved far, or it finished fast.
    """
    if (float(dt_s) < PET_MIN_S
            and math.hypot(float(total_dx_pt),
                           float(total_dy_pt)) < CLICK_SLOP_PT):
        return "poke"
    slow_enough = float(dt_s) >= PET_MIN_S
    short_enough = math.hypot(float(total_dx_pt),
                              float(total_dy_pt)) <= PET_MAX_TRAVEL_PT
    gentle_finish = math.hypot(float(tail_dx_pt),
                               float(tail_dy_pt)) <= PET_MAX_TAIL_PT
    if slow_enough and short_enough and gentle_finish:
        return "pet"
    return "push"
