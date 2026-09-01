"""The behavior machine: guarded states over the sensed digest, 50 Hz.

Ocarina's pattern (SURFACE.md/MACHINE.md in ~/Code/ocarina) ported to a robot:
the agent never streams commands — it arms a machine whose leaves are
deterministic behaviors executed at the control rate, with transitions guarded
by expressions over the SENSED digest. Fairness by construction: the guard
vocabulary contains only what the robot could genuinely know — camera-derived
ball sighting, proprioception, its own machine state — never the simulator's
ground-truth ball position.

Machine source is TOML (stdlib tomllib; no new dependencies), living in a git
repo the agent edits and hot-swaps via the `machine` socket command family.
Guards are expression strings compiled through the ast module against a strict
whitelist: dotted digest paths, numeric/string/bool literals, comparisons,
and/or/not. Nothing else parses, so nothing else runs.

WAKE NODES (ocarina's wake, robot-adapted): a node may declare
`wake = "reason"` — entering it parks a wake pack on the sim server, where a
blocked `machine wait` picks it up (the mind's interrupt line). A robot cannot
freeze the world like a paused game, so the wake node's own behavior is the
holding pattern while the mind thinks, and ocarina's mandatory non-wake
default becomes source you can read: every wake node must either carry a
transition guarded on elapsed_s (the deadline default that runs if no answer
ever comes) or declare `wake_hold = "why parking here forever is safe"` (the
explicit hold — a fallen duck with no stand-up policy has nothing better to
do). The machine stays autonomous-first, mind-optional, by construction.

SPEAKING NODES: a node may also declare `say = "..."` — entering it puts that
line on the control surface through the sim's existing `say` annotation verb,
and the server speaks it host-side if it can (the robot has a mouth servo, not
a speaker). It is an annotation in the same sense `wake` is: the machine says
what it is doing, and whether anything is listening is somebody else's
problem. Nothing about the behavior, the guards or the physics depends on it,
and a server too old to know the key simply ignores it — an unknown node key
has never been an error here, which is what makes a machine with a voice safe
to hot-reload onto a server without one.

A speaking node may add `say_mood = "excited"`: the weather on that line (see
voice.MOODS). It is a separate key rather than a table-valued `say` for the
degradation it buys — a server too old to know the mood speaks the line
neutral, which is exactly right, where a `say = { ... }` table would have made
the whole line unreadable to it. A mood with nothing to say is an error,
though: that is a line the author lost, not a style choice.

EMOTING NODES: `emote = "name"` names an authored gesture (emotes/*.toml, see
emote.py) to play on entry, and it is `say`'s exact parallel — same code site,
same annotation status, same indifference to whether anything is listening.
A node may carry both, and that is the point rather than a conflict: the mouth
says the line, the body plays the gesture. The grammar knows only that this is
a string. It does not know what emotes exist, what they do, or whether the
server it lands on has any — a machine naming a gesture nobody has is a lint
warning at load and a note at fire time, never a rejection, for the same
reason a machine with a voice loads onto a mute server.

ODOMETRY PATHS: guards may read `base.x`, `base.y`, `base.yaw_deg` and
`base.heading_x`/`base.heading_y` — the duck's own pose, filled in by the
executor rather than by the server's sensed digest. See _pose_digest.

THE ROLL: `roll` is a fresh number in [0, 1) drawn on every node entry and
held constant for the whole stay, so `roll < 0.15` is exactly "one time in
seven, decided on arrival" — and it does not flicker mid-node the way a
per-tick draw would, which would make a guard fire the instant it felt like
it rather than at its deadline. It exists because a machine that runs in
front of a person for hours has to be able to be UNpredictable, and nothing
else in this vocabulary can do it: everything sensed here is downstream of
the same walk, so a "coin" mined out of, say, which way the gait happened to
leave the duck pointing turns out to be an attractor, not a coin (measured:
it moved a wake rate from every-time to never with nothing in between). A
robot may flip a coin; that is not a fairness leak, and unlike the sensed
paths this one is the machine's own, like elapsed_s and node. Seed it for a
test by passing `rng=` to Machine.
"""

import ast
import math
import random
import tomllib

import numpy as np

# Every path a guard may read. The digest is built each tick by the executor;
# ball fields come from the camera detector (fake mediad), the rest from
# proprioception and the machine itself.
GUARD_PATHS = {
    "ball_seen.visible", "ball_seen.distance_m", "ball_seen.ground_distance_m",
    "ball_seen.bearing_deg", "ball_seen.elevation_deg", "ball_seen.age_s",
    "ball_seen.est_forward_m", "ball_seen.est_left_m", "ball_seen.speed_mps",
    # The goal sighting (fake mediad part 2: white frame in the horizon band
    # of the duck's own camera). est_* are dead-reckoned from the last
    # sighting via own odometry — the goal is world-fixed, so they stay live
    # while the head is down tracking the ball; null until first sighted.
    "goal_seen.visible", "goal_seen.bearing_deg", "goal_seen.width_deg",
    "goal_seen.distance_m", "goal_seen.age_s",
    "goal_seen.est_bearing_deg", "goal_seen.est_distance_m",
    "upright", "sitting", "active_policy", "behavior",
    # The referee's scoreboard (goal scenes only; False/0 elsewhere). Not a
    # fairness leak: a real pitch has a goal-line sensor, and "someone scored"
    # is not "where the ball is".
    "goal.scored", "goal.count",
    # Own odometry: where the duck is standing and which way it points. Filled
    # by the executor (_pose_digest), not by the server's sensed digest, and
    # deliberately so — it is derived from the robot's own state and needs no
    # sensor model, which means a machine that steers by its own position runs
    # on any server old enough to load it, with no protocol to negotiate. Not
    # a fairness leak either: dead reckoning is what _own_pose has fed the ball
    # approach since the striker. The ground truth this vocabulary still
    # withholds is where anything ELSE is.
    #
    # heading_x/heading_y are cos/sin of the yaw, and they are here because
    # yaw_deg wraps at ±180 — exactly where a duck walking the -x direction
    # lives — while the grammar has no abs() to paper over it. The components
    # are wrap-free and read plainly: `base.heading_x > 0.0` is "pointing +x".
    "base.x", "base.y", "base.yaw_deg", "base.heading_x", "base.heading_y",
    # The human, as the desktop pet sees one (sim_server's _pet_cursor_state /
    # _pet_touch_state). The overlay is the only thing on either side of the
    # socket that can see a mouse pointer, so unlike base.* these come from
    # the SERVER's digest — but they are shaped the same way ball_seen is, and
    # for the same reason: a machine asks where something is and how long ago
    # it was true, and a null answers every question about a hand that is not
    # there. `dx_m` is signed (which way to walk), `dist_m` is horizontal only
    # (a cursor held overhead is 0 m away — the duck cannot walk upwards).
    "cursor.present", "cursor.x_m", "cursor.z_m", "cursor.dx_m",
    "cursor.dist_m", "cursor.age_s", "cursor.near_floor", "cursor.speed_mps",
    "touch.petted", "touch.age_s", "touch.count",
    # Being held. False on every daemon that has no weld to be held by, which
    # is what lets a machine carve `not carried` out of its fall reflex — a
    # duck dangling from a hand reads `not upright` within a second, and
    # without the carve-out every pick-up is a fall.
    "carried", "carried_s",
    "elapsed_s", "sim_time_s", "node", "roll",
}

# A node's `say` line. Same ceiling as voice.MAX_SAY_CHARS, spelled out rather
# than imported: the grammar deliberately knows nothing about audio, and this
# module's imports stay light enough to validate a machine anywhere.
MAX_SAY_CHARS = 400

# The moods a `say_mood` may name — voice.MOODS' keys, spelled out for the
# same reason MAX_SAY_CHARS is: validating a machine must not need numpy, an
# ffmpeg, or an opinion about audio. The roster is closed on purpose. A mood
# is not free-form content like an emote name (which the server may or may not
# have): it is a fixed vocabulary the renderer implements, so a typo here is a
# load-time error rather than a line that quietly comes out flat.
MOOD_NAMES = frozenset({"neutral", "excited", "sad", "alarmed", "smug"})

_ALLOWED_NODES = (
    ast.Expression, ast.BoolOp, ast.UnaryOp, ast.Not, ast.USub, ast.And, ast.Or,
    ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.Name, ast.Attribute, ast.Constant, ast.Load,
)


def _path_of(node):
    """Name/Attribute chain -> dotted path string, or None."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


class GuardError(ValueError):
    pass


def compile_guard(expr: str):
    """Validate a guard expression; returns an evaluator digest -> bool.

    Comparisons involving a null field (e.g. distance_m while not visible)
    evaluate False rather than raising — 'not visible' is the honest answer
    to any question about where an unseen ball is.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise GuardError(f"guard does not parse: {expr!r} ({e.msg})") from e
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise GuardError(
                f"guard {expr!r}: {type(node).__name__} is not in the grammar")
        if isinstance(node, ast.Constant) and not isinstance(
                node.value, (bool, int, float, str)):
            raise GuardError(f"guard {expr!r}: literal {node.value!r} not allowed")
    # Check every complete Name/Attribute chain against the whitelist.
    class PathCheck(ast.NodeVisitor):
        def visit_Attribute(self, node):
            p = _path_of(node)
            if p is None:
                raise GuardError(f"guard {expr!r}: unsupported attribute base")
            if p not in GUARD_PATHS:
                raise GuardError(f"guard {expr!r}: unknown path {p!r}")
            # don't recurse into the chain

        def visit_Name(self, node):
            if node.id not in GUARD_PATHS:
                raise GuardError(f"guard {expr!r}: unknown path {node.id!r}")

    PathCheck().visit(tree)

    def evaluate(digest: dict) -> bool:
        def ev(node):
            if isinstance(node, ast.Expression):
                return ev(node.body)
            if isinstance(node, ast.BoolOp):
                vals = [ev(v) for v in node.values]
                return all(vals) if isinstance(node.op, ast.And) else any(vals)
            if isinstance(node, ast.UnaryOp):
                v = ev(node.operand)
                if isinstance(node.op, ast.USub):
                    return None if v is None else -v
                return not v
            if isinstance(node, ast.Compare):
                left = ev(node.left)
                for op, comp in zip(node.ops, node.comparators):
                    right = ev(comp)
                    if left is None or right is None:
                        return False
                    if isinstance(op, ast.Eq):
                        ok = left == right
                    elif isinstance(op, ast.NotEq):
                        ok = left != right
                    elif isinstance(op, ast.Lt):
                        ok = left < right
                    elif isinstance(op, ast.LtE):
                        ok = left <= right
                    elif isinstance(op, ast.Gt):
                        ok = left > right
                    else:
                        ok = left >= right
                    if not ok:
                        return False
                    left = right
                return True
            if isinstance(node, (ast.Name, ast.Attribute)):
                return digest.get(_path_of(node))
            if isinstance(node, ast.Constant):
                return node.value
            raise GuardError(f"unreachable node {type(node).__name__}")
        return bool(ev(tree))

    return evaluate


def guard_paths(expr: str) -> set:
    """The digest paths a guard reads. Parse-only — call after (or alongside)
    compile_guard, which owns validation."""
    tree = ast.parse(expr, mode="eval")
    paths = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Name, ast.Attribute)):
            p = _path_of(node)
            if p in GUARD_PATHS:
                paths.add(p)
    return paths


# ---------- behaviors (the leaves) ----------
# Each is step(sim, params, mem, digest) called at 50 Hz while its node is
# current; `mem` is a per-entry dict (cleared on every node entry). Behaviors
# read the sensed digest — the same values guards see — plus proprioception.

def _clip(v, lo, hi):
    return max(lo, min(hi, v))


def _set_head(sim, pitch: float):
    """Tilt the head (and with it the camera) via the policy's gaze command —
    the same path the look intent uses, so the balance policy compensates."""
    p = sim.policy
    p.head_offset[:] = np.clip(
        np.array([0.0, pitch, 0.0, 0.0], dtype=np.float32),
        -p.head_max, p.head_max)
    p._update_command()


def bhv_idle(sim, params, mem, digest):
    if not mem.get("zeroed"):
        sim.policy.set_vel_cmd(0.0, 0.0, 0.0)
        _set_head(sim, 0.0)
        mem["zeroed"] = True


def bhv_drive(sim, params, mem, digest):
    """Constant velocity intent (params vx, vy, wz) — the building block for
    open-loop maneuvers. Example: a backoff node that walks backward for a
    couple of seconds after a failed kick, because the gait needs distance to
    rebuild real momentum — commanded creep from a standstill moves the duck
    millimeters, not centimeters."""
    if not mem.get("set"):
        sim.policy.set_vel_cmd(float(params.get("vx", 0.0)),
                               float(params.get("vy", 0.0)),
                               float(params.get("wz", 0.0)))
        mem["set"] = True


def bhv_search_ball(sim, params, mem, digest):
    """Rotate in place until a guard sees the ball. wz below ~1.2 is a dead
    zone, so search always turns at full rate. Head level: a level camera
    sees the far field; the approach looks down as it closes."""
    if not mem.get("head"):
        _set_head(sim, 0.0)
        mem["head"] = True
    wz = float(params.get("wz", 1.5))
    sim.policy.set_vel_cmd(0.0, 0.0, wz)


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _own_pose(sim):
    """Own world pose (x, y, yaw) — odometry, the robot's own knowledge."""
    adr = sim.qpos_adr
    x, y = float(sim.data.qpos[adr]), float(sim.data.qpos[adr + 1])
    q = sim.data.qpos[adr + 3:adr + 7]
    w, qx, qy, qz = (float(v) for v in q)
    yaw = math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return x, y, yaw


# What "I don't know where I am" looks like in the digest. Every comparison
# over a null field is False (see compile_guard), so a machine that steers by
# position simply stops steering rather than steering on a guess.
_NO_POSE = {"base.x": None, "base.y": None, "base.yaw_deg": None,
            "base.heading_x": None, "base.heading_y": None}

# The sim shapes that legitimately have no pose to give: a bare stub in a test,
# a hardware feed whose telemetry has not started, a server whose model has no
# freejoint. All of them fail inside _own_pose on attribute or index access.
_POSE_MISSING = (AttributeError, TypeError, IndexError, ValueError)


def _pose_digest(sim) -> dict:
    """The odometry half of the digest — see GUARD_PATHS' `base.*` note.

    The executor adds this each tick instead of the server building it into
    the sensed digest, because it is the one part of the vocabulary that needs
    nothing from the server but the state the robot already carries. That is
    what lets a position-steering machine hot-load onto a running daemon.
    """
    try:
        x, y, yaw = _own_pose(sim)
    except _POSE_MISSING:
        return dict(_NO_POSE)
    return {"base.x": round(x, 4), "base.y": round(y, 4),
            "base.yaw_deg": round(math.degrees(yaw), 2),
            "base.heading_x": round(math.cos(yaw), 4),
            "base.heading_y": round(math.sin(yaw), 4)}


def bhv_stroll(sim, params, mem, digest):
    """Walk a compass heading and hold it — the pet's gait.

    `drive` sets one velocity intent and lets it ride, which is right for a
    two-second backoff after a failed kick and wrong for a two-metre walk: a
    hundred steps of gait noise bends an open-loop vx into a slow arc, and a
    duck that is supposed to be pacing the Dock ends up facing the wallpaper.
    So this is drive with a compass — proportional steering onto
    `heading_deg` in the WORLD frame (own odometry, the same knowledge the
    ball approach has always steered on), at `vx`.

    Two bands, and the split is the gait's, not a preference. Measured on the
    shipped walk policy (plain scene, commanded vs achieved):

        vx  0.22 -> 0.096 m/s   0.26 -> 0.114   0.30 -> 0.131
        vx <= 0.21 -> the duck does not move at all, and neither does any
                      negative vx: there is no reverse
        wz  1.5 -> 0.76-0.92 rad/s in place (180 deg in ~3.5-4 s)
        wz  1.2 while walking -> the walk collapses to 0.008 m/s

    So a big wz is a point turn whether you meant one or not, and a small one
    is free steering: at vx 0.30, wz 0.2/0.3/0.4 buys 5/11/15 deg per second
    while still covering 0.124/0.118/0.105 m/s of ground. Hence `wz_max`
    defaulting to 0.4 — winding it up does not steer harder, it stops the
    duck. Badly off heading, the point turn is the honest move: vx to zero,
    wz to full, and no pretence of walking. Which also makes turning round at
    a wall free — "stroll the other way" IS the turn, closed loop, and the
    machine only has to name the new heading.
    """
    if not mem.get("head"):
        _set_head(sim, float(params.get("head", 0.0)))
        mem["head"] = True
    vx = float(params.get("vx", 0.30))
    try:
        _, _, yaw = _own_pose(sim)
    except _POSE_MISSING:
        # No odometry: walk the intent open-loop rather than steer on a guess.
        sim.policy.set_vel_cmd(vx, 0.0, 0.0)
        return
    err = _wrap(math.radians(float(params.get("heading_deg", 0.0))) - yaw)
    if abs(err) > math.radians(float(params.get("turn_in_deg", 25.0))):
        sim.policy.set_vel_cmd(0.0, 0.0, math.copysign(1.5, err))
        return
    wz_max = float(params.get("wz_max", 0.4))
    sim.policy.set_vel_cmd(
        vx, 0.0, _clip(float(params.get("gain", 1.0)) * err, -wz_max, wz_max))


def bhv_trick(sim, params, mem, digest):
    """Fire one episodic trick on entry, then hold — `celebrate` generalised.

    celebrate is this with the roulade already chosen; a pet needs the same
    one-shot for postures rather than stunts: `trick = "sit"` to settle down
    for a nap, `trick = "stand"` to get back up. The posture lands in the
    digest (`sitting`), so the machine can wait for it to actually arrive
    instead of trusting a timer — and a refused trick never satisfies that
    guard, which is exactly what the node's deadline transition is for.

    stage_ball defaults False here where celebrate leaves it True: staging
    teleports the ball to the foot, and a posture change has no business
    moving the world.
    """
    if not mem.get("done"):
        mem["done"] = True
        sim._handle_trick(str(params.get("trick", "stand")),
                          stage_ball=bool(params.get("stage_ball", False)))


def _drive_to(sim, dist, bearing):
    """Walk toward a trunk-frame target: full point turn when badly off,
    else forward with proportional steering (the session-tested gains)."""
    if abs(bearing) > math.radians(35):
        sim.policy.set_vel_cmd(0.0, 0.0, math.copysign(1.5, bearing))
    else:
        vx = 0.3 if dist > 0.35 else 0.2
        sim.policy.set_vel_cmd(vx, 0.0, _clip(2.5 * bearing, -1.2, 1.2))


def _aim_ready(sim, params, mem, digest, fwd, left, d, b, gb_deg):
    """The line-of-fire detour. True when the duck is behind the ball facing
    along it (attack: fall through to the normal pocket approach); False
    while still maneuvering (a velocity command has been issued).

    Geometry, all sensed/remembered: ball at (fwd, left); goal from the
    dead-reckoned estimate. The fire line runs ball->goal; the desired trunk
    heading is that direction minus kick_skew_deg (kick_right sends the ball
    ~30 deg right of the trunk, so the trunk wants to be ~30 deg left of the
    line). Walk to a standoff point behind the ball on that line — swerving
    to give the ball 0.28 m clearance so the walk-up cannot dribble it away
    — then attack.
    """
    if d > 1.1:
        # Far field: the ball position estimate wobbles ±0.25 m with the
        # gait out here — any alignment decision would be noise. Just close
        # in (the plain approach below); the maneuvering starts near-field.
        return True
    skew = math.radians(float(params.get("kick_skew_deg", -30.0)))
    tol = math.radians(float(params.get("aim_tol_deg", 14.0)))
    standoff = float(params.get("standoff_m", 0.45))
    gd = digest.get("goal_seen.est_distance_m")
    gd = gd if gd is not None else 2.0  # bearing-only memory: goal treated far
    gbr = math.radians(gb_deg)
    ux, uy = gd * math.cos(gbr) - fwd, gd * math.sin(gbr) - left
    if math.hypot(ux, uy) < 1e-6:
        return True
    # Desired arrival heading: along the fire line, rotated by the kick skew
    # (the trunk aims left of the goal so the rightward kick aims at it).
    h_des = _wrap(math.atan2(uy, ux) - skew)
    e = h_des  # current heading is 0 in the trunk frame
    ef = mem.get("aim_err_f")
    ef = e if ef is None else (0.9 * ef + 0.1 * e)  # ~0.2 s EMA vs gait noise
    mem["aim_err_f"] = ef
    mem["aim_n"] = mem.get("aim_n", 0) + 1
    if mem.get("attacking"):
        if abs(ef) > math.radians(60.0) and d > 0.20:
            # Aim badly wrong at ANY range (a wrong-side entanglement, the
            # geometry flipped as the ball got nudged): stop dribbling chaos,
            # go around again.
            mem["attacking"] = False
        elif abs(ef) > math.radians(35.0) and d > 0.50:
            mem["attacking"] = False  # drifted off the line; go around again
        elif d <= 0.20:
            return True  # fine range: the pocket trim below finishes the job
        else:
            # Attack run: FOLLOW THE LINE, don't chase the ball. Holding the
            # ball at a fixed bearing is a pursuit spiral — the heading
            # rotates all the way in and the aim washes out. Instead track a
            # carrot on the line parallel to the fire line, offset so the
            # duck's path passes pocket_left to the LEFT of the ball: it
            # arrives heading h_des with the ball already sitting in the
            # deep pocket, no terminal point turn needed.
            off = -float(params.get("pocket_left_m", -0.055))
            nx, ny = -math.sin(h_des), math.cos(h_des)  # left normal
            # Long lookahead and soft gain: a short twitchy carrot overshoots
            # laterally at close range and swings the ball out of the tight
            # head-down field of view right before the pocket.
            c = _clip(0.5 * d, 0.18, 0.35)
            cx = fwd + off * nx - c * math.cos(h_des)
            cy = left + off * ny - c * math.sin(h_des)
            bC = math.atan2(cy, cx)
            sim.policy.set_vel_cmd(0.3 if d > 0.4 else 0.25, 0.0,
                                   _clip(1.8 * bC, -0.9, 0.9))
            return False
    if abs(ef) < tol and abs(b) < math.radians(30.0) and mem["aim_n"] > 25:
        # aim_n > 25: the EMA needs ~0.5 s of samples before it means
        # anything — one lucky tick at node entry must not latch an attack.
        mem["attacking"] = True
        mem["standoff_w"] = None
        return True
    x, y, yaw = _own_pose(sim)
    cy_, sy_ = math.cos(yaw), math.sin(yaw)
    # Remember where the ball is in the world: the far-side detour below
    # deliberately walks it out of frame, and the arrival turn-in needs to
    # know where to look.
    mem["ball_w"] = (x + cy_ * fwd - sy_ * left, y + sy_ * fwd + cy_ * left)
    sx = fwd - standoff * math.cos(h_des)
    sy = left - standoff * math.sin(h_des)
    dS, bS = math.hypot(sx, sy), math.atan2(sy, sx)
    if dS < 0.12:
        # Behind the ball but not yet facing down the line: turn in place.
        sim.policy.set_vel_cmd(0.0, 0.0, math.copysign(1.5, ef))
        return False
    far_side = abs(_wrap(bS - b)) > math.radians(60)
    if far_side and d < 0.38:
        # Wrong side of the ball and right on top of it: back away first
        # (ball kept in view) — the detour needs room to swing around.
        sim.policy.set_vel_cmd(-0.25, 0.0, _clip(2.5 * b, -0.5, 0.5))
        return False
    if not far_side:
        if abs(b) > math.radians(30):
            # Eye on the ball: it is drifting toward the frame edge —
            # re-centre before maneuvering further.
            sim.policy.set_vel_cmd(0.0, 0.0, math.copysign(1.5, b))
            return False
        # Keep the ball in frame while walking: cap the walk direction to
        # within 35 deg of the ball; the path curves, re-planned every tick.
        bS = _clip(bS, b - math.radians(35), b + math.radians(35))
    # else: the standoff is on the FAR side — a deliberate blind detour on
    # dead-reckoning (the cached target + blink mode carry it, and the
    # arrival turn-in re-acquires the ball).
    # Swerve if the straight walk to the standoff would plow through the ball.
    phi = _wrap(b - bS)
    phi_min = math.asin(min(1.0, 0.30 / max(d, 0.05)))
    if math.cos(phi) > 0.0 and d < dS + 0.1 and abs(phi) < phi_min:
        sgn = 1.0 if phi >= 0 else -1.0
        bS = _wrap(b - sgn * phi_min)
    sxp, syp = dS * math.cos(bS), dS * math.sin(bS)
    mem["standoff_w"] = (x + cy_ * sxp - sy_ * syp, y + sy_ * sxp + cy_ * syp)
    _drive_to(sim, dS, bS)
    return False


def bhv_approach_ball(sim, params, mem, digest):
    """Close on the ball and settle into the kick pocket — the session-tested
    controller (continuous sticky intents; full-throttle point turns; bearing
    target ~-25 deg so the ball lines up with the right foot).

    Looks DOWN as it closes: with the head level, a floor ball drops below
    the camera frame at ~0.15 m ground distance — before the 0.10 m pocket —
    so near-range tracking is physically impossible without the head tilt.

    With `aim = true` (and a goal in the duck's memory) the approach comes in
    along the LINE OF FIRE: first walk to a standoff point behind the ball on
    the ball->goal line — detouring around the ball rather than dribbling it
    away by accident — then attack the pocket straight down that line. The
    heading is offset by `kick_skew_deg` (measured: a settled kick_right
    sends the ball ~30 deg RIGHT of the trunk heading), so the trunk aims
    left of the goal and the kick aims AT it.
    """
    aim = bool(params.get("aim", False))
    fwd = digest.get("ball_seen.est_forward_m")
    left = digest.get("ball_seen.est_left_m")
    if not digest.get("ball_seen.visible") or fwd is None or left is None:
        if aim and mem.get("attacking"):
            return  # hold course through a detector blink mid-attack — the
            # velocity intent is sticky, and the node's age guard catches a
            # ball that is genuinely gone.
        tgt = mem.get("standoff_w")
        if aim and tgt is not None and not mem.get("attacking"):
            # Mid-detour blink: the ball slid out of frame while we walk
            # around it. Keep walking to the cached standoff point (own
            # odometry); on arrival, turn toward the remembered ball to
            # re-acquire — and if it truly vanished, the node's guard exits.
            x, y, yaw = _own_pose(sim)
            dx, dy = tgt[0] - x, tgt[1] - y
            dT = math.hypot(dx, dy)
            bw = mem.get("ball_w")
            if dT < 0.15 and bw is not None:
                bb = _wrap(math.atan2(bw[1] - y, bw[0] - x) - yaw)
                sim.policy.set_vel_cmd(0.0, 0.0, math.copysign(1.5, bb))
            else:
                _drive_to(sim, dT, _wrap(math.atan2(dy, dx) - yaw))
        else:
            sim.policy.set_vel_cmd(0.0, 0.0, 0.0)  # lost it; a guard will exit
        return
    # Trunk-frame geometry — the pocket the kick was measured in. d/bearing
    # here are TRUNK-relative, not camera-relative: the camera sits ~8 cm
    # ahead of the trunk, which is a huge error at kick range.
    d = math.hypot(fwd, left)
    b = math.atan2(left, fwd)
    # 0.5 rad + the 20 deg mount pitch keeps a floor ball in frame from
    # ~0.45 m all the way into the pocket; hysteresis so the tilt won't flap.
    head_down = float(params.get("head_down", 0.5))
    if d < 0.45:
        mem["tilted"] = True
    elif d > 0.60:
        mem["tilted"] = False
    _set_head(sim, head_down if mem.get("tilted") else 0.0)
    gb = digest.get("goal_seen.est_bearing_deg")
    if aim and gb is not None and not _aim_ready(sim, params, mem, digest,
                                                fwd, left, d, b, gb):
        return  # walking the detour onto the line of fire
    # Kick sweep (measured): forward is the critical axis — the ball connects
    # hard at true forward <= 0.09 m, dies at 0.10-0.12. The camera reads
    # ~1.5 cm long at point-blank (beak occlusion biases the blob), so the
    # sensed stop target sits closer than the old 0.099.
    if d > 0.18:
        # Walk-in bearing target: 0 drives straight at the ball (soccer);
        # the striker holds the ball at ~-18 deg so it ARRIVES with the ball
        # already in the deep pocket — the fine-range tuck below is a trim,
        # not a point turn (a 1.5 rad/s spin overshoots ~17 deg per 5 Hz
        # detector tick and flings the ball out of frame).
        tgt_b = math.radians(float(params.get("pocket_bearing_deg", 0.0)))
        e_b = _wrap(b - tgt_b)
        if abs(e_b) > math.radians(35):
            vx, wz = 0.0, math.copysign(1.5, e_b)
        else:
            vx, wz = 0.3, _clip(2.5 * e_b, -1.0, 1.0)
    else:
        # Fine range: servo est_forward/est_left into the pocket guard's
        # window, so "controller satisfied" implies "guard fires". The creep
        # deliberately stops OUTSIDE the walking foot's reach (contact
        # forensics: the swinging right foot pokes any ball nearer than
        # ~0.09 true / ~0.11 sensed) — the kick guard fires while still
        # creeping, and the stop's own forward slide delivers the last
        # ~3 cm with feet planted. The stop closes the gap, not a step.
        vx = 0.2 if fwd > 0.110 else (-0.2 if fwd < 0.085 else 0.0)
        lmax = float(params.get("pocket_left_max", -0.020))
        lmin = float(params.get("pocket_left_min", -0.080))
        wz = 1.5 if left > lmax else (-1.5 if left < lmin else 0.0)
    sim.policy.set_vel_cmd(vx, 0.0, wz)


def bhv_kick(sim, params, mem, digest):
    """Stop, LOOK UP, settle, one honest kick, then look back down to verify.

    The head MUST be level for the swing: head_pose is part of the policy
    command vector, and the kick policy fed a bowed head does not swing at
    all (measured: 1.3-1.5 m with the head level, 0.00 m at 0.5 rad down,
    same ball, same stance). So the duck lines up watching the ball, looks
    up to kick like a striker, and glances down afterwards so the whiff
    guard has fresh eyes on where the ball ended up."""
    settle_s = float(params.get("settle_s", 0.8))
    if not mem.get("stopped"):
        sim.policy.set_vel_cmd(0.0, 0.0, 0.0)
        _set_head(sim, 0.0)
        mem["stopped"] = True
    if digest["elapsed_s"] >= settle_s and not mem.get("kicked"):
        mem["kicked"] = True
        foot = params.get("foot", "right")
        sim._handle_trick(f"kick_{foot}", stage_ball=bool(params.get("stage_ball", False)))
    # Swing done (trick auto-returns 3 s after trigger): look down again so
    # the whiff-check guard gets a fresh sighting of the pocket.
    if mem.get("kicked") and not mem.get("verified") \
            and digest["elapsed_s"] >= settle_s + 2.9:
        _set_head(sim, float(params.get("head_down", 0.5)))
        mem["verified"] = True


def bhv_celebrate(sim, params, mem, digest):
    if not mem.get("done"):
        mem["done"] = True
        sim._handle_trick(str(params.get("trick", "roulade")))


BEHAVIORS = {
    "idle": bhv_idle,
    "drive": bhv_drive,
    "stroll": bhv_stroll,
    "search_ball": bhv_search_ball,
    "approach_ball": bhv_approach_ball,
    "kick": bhv_kick,
    "trick": bhv_trick,
    "celebrate": bhv_celebrate,
}


# ---------- the machine ----------

class MachineError(ValueError):
    pass


class Machine:
    """A validated machine: nodes, compiled guards, and the 50 Hz executor."""

    def __init__(self, spec: dict, source_path: str = "<inline>", rng=None):
        self.source_path = source_path
        # The roll's source. Injectable so a test can pin a rate instead of
        # sampling one; a live machine gets the module's own generator.
        self.rng = rng if rng is not None else random.Random()
        m = spec.get("machine")
        if not isinstance(m, dict):
            raise MachineError("missing [machine] table")
        self.name = m.get("name", "unnamed")
        nodes = spec.get("node")
        if not nodes:
            raise MachineError("no [[node]] tables")
        self.nodes = {}
        for n in nodes:
            name = n.get("name")
            if not name:
                raise MachineError("a [[node]] has no name")
            if name in self.nodes:
                raise MachineError(f"duplicate node {name!r}")
            bname = n.get("behavior", "idle")
            if bname not in BEHAVIORS:
                raise MachineError(
                    f"node {name!r}: unknown behavior {bname!r} "
                    f"(have: {', '.join(sorted(BEHAVIORS))})")
            trans = []
            for t in n.get("transition", []):
                if "when" not in t or "to" not in t:
                    raise MachineError(f"node {name!r}: transition needs when+to")
                trans.append({"when": t["when"],
                              "guard": compile_guard(t["when"]),
                              "to": t["to"]})
            wake = n.get("wake")
            wake_hold = n.get("wake_hold")
            say = n.get("say")
            say_mood = n.get("say_mood")
            emote = n.get("emote")
            for key, val in (("wake", wake), ("wake_hold", wake_hold),
                             ("say", say), ("say_mood", say_mood),
                             ("emote", emote)):
                if val is not None and not isinstance(val, str):
                    raise MachineError(f"node {name!r}: {key} must be a string")
            if say is not None and len(say) > MAX_SAY_CHARS:
                raise MachineError(
                    f"node {name!r}: say is {len(say)} chars (max "
                    f"{MAX_SAY_CHARS}) — a behavior node is not a monologue")
            if say_mood is not None and say_mood not in MOOD_NAMES:
                raise MachineError(
                    f"node {name!r}: unknown say_mood {say_mood!r} "
                    f"(have: {', '.join(sorted(MOOD_NAMES))})")
            if say_mood is not None and say is None:
                raise MachineError(
                    f"node {name!r}: say_mood without say — a mood with "
                    f"nothing to say")
            if wake_hold is not None and wake is None:
                raise MachineError(
                    f"node {name!r}: wake_hold without wake — the hold is the "
                    f"declared default for a wake nobody answered")
            if wake is not None and wake_hold is None and not any(
                    "elapsed_s" in guard_paths(t["when"]) for t in trans):
                raise MachineError(
                    f"wake node {name!r} has no deadline: add a transition "
                    f"guarded on elapsed_s (the default that runs if the mind "
                    f"never answers) or an explicit wake_hold = \"why parking "
                    f"here forever is safe\"")
            self.nodes[name] = {"behavior": bname,
                                "params": n.get("params", {}),
                                "transitions": trans,
                                "wake": wake, "wake_hold": wake_hold,
                                "say": say, "say_mood": say_mood,
                                "emote": emote}
        # machine-level transitions (checked before the node's own — the
        # "fell over" escape hatch lives here)
        self.global_transitions = []
        for t in m.get("transition", []):
            if "when" not in t or "to" not in t:
                raise MachineError("[machine] transition needs when+to")
            self.global_transitions.append(
                {"when": t["when"], "guard": compile_guard(t["when"]),
                 "to": t["to"]})
        self.initial = m.get("initial")
        if self.initial not in self.nodes:
            raise MachineError(f"initial node {self.initial!r} does not exist")
        for src, n in self.nodes.items():
            for t in n["transitions"]:
                if t["to"] not in self.nodes:
                    raise MachineError(f"node {src!r}: transition to unknown "
                                       f"node {t['to']!r}")
        for t in self.global_transitions:
            if t["to"] not in self.nodes:
                raise MachineError(f"[machine] transition to unknown node {t['to']!r}")

        # live state
        self.armed = False
        self.current = self.initial
        self.entered_at = 0.0
        self.mem = {}
        self.roll = self.rng.random()

    @classmethod
    def load(cls, path: str, rng=None) -> "Machine":
        with open(path, "rb") as f:
            try:
                spec = tomllib.load(f)
            except tomllib.TOMLDecodeError as e:
                raise MachineError(f"{path}: TOML does not parse: {e}") from e
        return cls(spec, source_path=path, rng=rng)

    def enter(self, node: str, sim_time: float):
        self.current = node
        self.entered_at = sim_time
        self.mem = {}
        self.roll = self.rng.random()   # one draw per stay, not per tick

    def status(self) -> dict:
        return {"name": self.name, "source": self.source_path,
                "armed": self.armed, "node": self.current,
                "nodes": sorted(self.nodes),
                "wake_nodes": sorted(n for n, v in self.nodes.items()
                                     if v["wake"] is not None),
                "say_nodes": sorted(n for n, v in self.nodes.items()
                                    if v["say"] is not None),
                "say_mood_nodes": sorted(n for n, v in self.nodes.items()
                                         if v["say_mood"] is not None),
                "emote_nodes": sorted(n for n, v in self.nodes.items()
                                      if v["emote"] is not None)}

    def tick(self, sim, digest: dict):
        """One 50 Hz step: transitions first (global, then node, in order),
        then the current behavior. Returns the fired transition or None."""
        digest["elapsed_s"] = round(digest["sim_time_s"] - self.entered_at, 3)
        digest["roll"] = self.roll
        # Odometry, added here rather than by the server (GUARD_PATHS' base.*
        # note). In place, before the guards, so a wake pack's digest snapshot
        # carries the pose and the roll the transition actually fired on.
        digest.update(_pose_digest(sim))
        node = self.nodes[self.current]
        for t in self.global_transitions + node["transitions"]:
            if t["to"] != self.current and t["guard"](digest):
                fired = {"from": self.current, "to": t["to"], "when": t["when"]}
                self.enter(t["to"], digest["sim_time_s"])
                digest["elapsed_s"] = 0.0
                return fired
        BEHAVIORS[node["behavior"]](sim, node["params"], self.mem, digest)
        return None
