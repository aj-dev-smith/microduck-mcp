"""Socket-controlled Microduck MuJoCo simulator.

Runs the same CPU MuJoCo + ONNX policy stack as microduck_rl's
scripts/infer_policy.py (whose PolicyInference class is imported directly from
that repo), but replaces terminal-keyboard control with a Unix-socket JSON-lines
control plane — the same shape as the real robot's robotd socket: clients send
intents, never motor writes.

Every MuJoCo call happens on the sim thread. Socket connections enqueue
requests; the 50 Hz loop drains the queue between control ticks and replies.

Run headless (plain python) or with the interactive viewer (macOS: mjpython):

    duck-sim --rl-repo ../microduck_rl --policies ../microduck/policies
    uv run mjpython -m microduck_mcp.sim_server --viewer ...
"""

import argparse
import importlib.util
import json
import math
import os
import queue
import random
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque

import mujoco
import numpy as np

from .emote import HEAD_CHANNELS, EmoteError, EmoteLibrary
from .machine import Machine, MachineError

# Nm/A for the Dynamixel XL330 (BAM m6 model). Used for the firmware
# current-limit -> torque clamp when the `bam` package is not installed.
KT_XL330 = 0.3660
DEFAULT_CURRENT_LIMIT = 1.75  # A, XL330 firmware default

CONTROL_HZ = 50
DECIMATION = 4
TIMESTEP = 0.005
# How far the loop may fall behind its deadline before giving up on catching
# up (a stall: a slow render, the machine sleeping) and resyncing to now.
RESYNC_LAG_S = 0.5

SCENES = {
    "ball": "src/mjlab_microduck/robot/microduck/scene_ball.xml",
    "pitch": "src/mjlab_microduck/robot/microduck/scene_pitch.xml",
    "plain": "src/mjlab_microduck/robot/microduck/scene.xml",
}

# The goal in scene_pitch.xml: line at world x=+0.60 with the mouth opening
# back towards the origin (-x), posts at y=±0.20, crossbar underside at
# z=0.18. Only used when the loaded scene actually has a goal (see
# find_goal_geom); every other scene never sees these numbers.
GOAL_LINE_X = 0.60
GOAL_HALF_WIDTH_Y = 0.20
GOAL_HEIGHT_Z = 0.18
# Back of the net (the ground bar the back net slopes down to). The scoring
# volume is the NET INTERIOR — without this bound a ball that rolls wide,
# wraps around the outside of the net and stops behind the goal "scores"
# the moment its |y| drifts under the post line (it happened: a filmed
# match called a goal at x=1.11, half a metre behind the net).
GOAL_NET_BACK_X = 0.82

# Policy roles -> filenames as shipped in pollen-robotics/microduck's policies/
POLICY_FILES = {
    "walking": "alpha_walking.onnx",
    "standing": "alpha_stand.onnx",
    "sitstand": "alpha_sitstand.onnx",
    "ground_pick": "alpha_ground_pick.onnx",
    "kick_left": "ball_kick_left.onnx",
    "kick_right": "ball_kick_right.onnx",
    "roulade": "roulade.onnx",
}

CAMERA_VIEWS = ("follow", "front", "side", "top", "head")

# Head camera. The model ships a `head_camera` on the jaw_soft body, but as
# exported it sits inside the lens-housing mesh and looks backwards, so we
# re-pose it at load: nudged forward clear of the beak, aimed along the head's
# forward axis (-z of that body, +x is up), and pitched down so the floor in
# front of the feet is in frame — the real camera has to see the ball at its
# feet, and a level 70 deg lens 25 cm up does not reach it.
HEAD_CAM_FORWARD_M = 0.012
HEAD_CAM_PITCH_DEG = 20.0
HEAD_CAM_FOVY_DEG = 70.0

# "fake mediad": the ball detector that runs on head-camera pixels. Stands in
# for the real robot's mediad service, which owns the camera and publishes
# derived features rather than raw frames.
BALL_RADIUS_M = 0.035  # matches ball.xml's geom size
# ball.xml specifies rolling friction but leaves the geom at MuJoCo's default
# condim=3 (sliding only), so rolling resistance never enters any contact and
# a nudged ball rolls forever (26.9 m measured from one kick, never stopping).
# Enable the full friction cone and use a floorball-on-floor coefficient:
# a 1.5 m/s kick runs ~1.8 m and stops; an accidental toe-poke dies in ~9 cm.
BALL_ROLL_FRICTION = 0.002
DET_W, DET_H = 320, 240
DET_EVERY = 10  # control steps between detections -> 5 Hz at 50 Hz control
DET_MIN_PX = 6
DET_SPEED_WINDOW_S = 0.6   # baseline for the ball-speed estimate
DET_SPEED_MIN_DT_S = 0.25  # publish null speed until the track spans this

# Goal detector (fake mediad, part 2): the goal frame is white — but so are
# the painted pitch lines and the clouds. What tells them apart is WORLD-RAY
# ELEVATION from the robot's own kinematics: sky can only exist above the
# true horizon (the floor plane is infinite), painted lines on the ground sit
# below -5.5 deg anywhere on the pitch, and the crossbar — hung at almost
# exactly camera height — lives in the narrow band between. The ceiling is
# -0.5 (not 0) because past MuJoCo's 50 m far clip the ground is not drawn
# and a 1-px sliver of sky leaks below the geometric horizon. Grazing-angle
# line pixels that do reach the band arrive 1 px per image column; posts
# stack several, hence the dense-column filter.
GOAL_DET_MIN_PX = 8
GOAL_ELEV_LO_DEG = -5.5
GOAL_ELEV_HI_DEG = -0.5
GOAL_MOUTH_OUTER_W = 0.432  # outer post faces, m -> range from angular width

# Kickoff spot on the pitch scene: NOT the scene's penalty spot (0.30 m from
# the line, where the mouth subtends ±34 deg and scoring is luck-proof). A
# metre out the mouth is ±11 deg and the duck has to actually aim.
PITCH_BALL_SPAWN = (-0.40, 0.0)

# The mouth. The real robot has a mouth servo (one of the five neck/head/mouth
# XL330s) and a `robot.mouth` verb: opening 0 (closed) to 1 (open), sent as a
# continuous notification. The shipped MJCF has no mouth joint — the whole
# beak (`jaw` mesh, the yellow bill with the round pivot boss on its side) is
# fixed to the head body — so at load time the model is rebuilt with the
# beak's VISUAL geom (group 2) moved onto a mocap body, posed every tick from
# the head's own kinematics plus the commanded opening. Its collision twin
# (group 3) stays on the head, so physics is byte-identical to the stock
# model and no qpos/dof shifts under the walk policy. The pink soft-mouth
# interior stays on the head too — dropping the beak reveals it, BD-X style.
# Hinge placement was tuned against rendered frames: the beak pivots at the
# boss (rear-top, the visible hinge on the real robot) and swings down.
MOUTH_BODY = "jaw_soft"
MOUTH_MESH = "jaw"
MOUTH_HINGE_POS = (0.005, 0.0, -0.018)  # boss center, head body frame
MOUTH_HINGE_AXIS = (0.0, 1.0, 0.0)
MOUTH_MAX_RAD = 0.45

# The one voice-bank tag the server rations. Every other call is a reaction
# anybody may ask for; the wheee is the goal celebration, and a celebration
# you can play whenever you like is not a celebration.
WHEEE_TAG = "wheee"
WHEEE_REFUSAL = ("the wheee is for a goal the duck actually scored, and the "
                 "referee has none on the board this episode")

# Behaviors that own the head outright, and may not be interrupted to look
# expressive: approach_ball STEERS by what the head camera can see (a floor
# ball drops out of frame above ~0.15 m with the head level), and the kick
# policy fed a bowed head does not swing at all — measured, 1.3-1.5 m level
# against 0.00 m at 0.5 rad down. A gesture over either is not expression, it
# is a missed ball.
HEAD_BOUND_BEHAVIORS = ("approach_ball", "kick")

# How long a `mouth` intent keeps the beak: `duck say` streams openings at
# 40 Hz while it talks, so a gap wider than this means the stream ended rather
# than paused, and an emote may have the beak back.
MOUTH_HELD_S = 0.5


def load_infer_policy_module(rl_repo: str):
    """Import microduck_rl's scripts/infer_policy.py as a module."""
    path = os.path.join(rl_repo, "scripts", "infer_policy.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"infer_policy.py not found at {path} — pass --rl-repo pointing at a "
            "clone of https://github.com/pollen-robotics/microduck_rl"
        )
    spec = importlib.util.spec_from_file_location("infer_policy", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_model_with_mouth(xml_path: str):
    """Compile the scene with the mouth plate moved onto a mocap body.

    Returns (model, mouth_ok). Any failure — an MJCF without the head body or
    plate mesh, an older mujoco without MjSpec — falls back to the stock model
    with the mouth disabled, never a broken sim.
    """
    try:
        spec = mujoco.MjSpec.from_file(xml_path)
        plate = None
        for body in spec.bodies:
            if body.name == MOUTH_BODY:
                for geom in body.geoms:
                    # the beak has a visual/collision twin pair; take the
                    # visual (group 2) and leave the collision copy in place
                    if geom.meshname == MOUTH_MESH and geom.group == 2:
                        plate = geom
        if plate is None:
            raise ValueError(f"no visual {MOUTH_MESH} geom on body {MOUTH_BODY!r}")
        pos, quat, mat = plate.pos.copy(), plate.quat.copy(), plate.material
        spec.delete(plate)
        mouth = spec.worldbody.add_body(name="mouth_plate", mocap=True)
        mouth.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=MOUTH_MESH,
                       pos=pos, quat=quat, material=mat,
                       contype=0, conaffinity=0, group=2)
        return spec.compile(), True
    except Exception as e:
        print(f"note: mouth disabled ({e}) — loading stock model")
        return mujoco.MjModel.from_xml_path(xml_path), False


def mouth_pose(head_pos, head_quat, opening: float):
    """World pose for the mouth plate's mocap body.

    The plate's home pose coincides with the head body; opening it is a hinge
    rotation about MOUTH_HINGE_AXIS at MOUTH_HINGE_POS (head frame), so
    opening=0 reproduces the stock model exactly.
    """
    theta = float(np.clip(opening, 0.0, 1.0)) * MOUTH_MAX_RAD
    qh = np.zeros(4)
    mujoco.mju_axisAngle2Quat(qh, np.asarray(MOUTH_HINGE_AXIS, dtype=np.float64), theta)
    hinge = np.asarray(MOUTH_HINGE_POS, dtype=np.float64)
    swung = np.zeros(3)
    mujoco.mju_rotVecQuat(swung, hinge, qh)
    local = hinge - swung  # translation that keeps the hinge point fixed
    quat = np.zeros(4)
    mujoco.mju_mulQuat(quat, np.asarray(head_quat, dtype=np.float64), qh)
    offset = np.zeros(3)
    mujoco.mju_rotVecQuat(offset, local, np.asarray(head_quat, dtype=np.float64))
    return np.asarray(head_pos, dtype=np.float64) + offset, quat


def quat_to_rpy(q):
    """Quaternion [w, x, y, z] -> roll, pitch, yaw in radians."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


NOT_SEEN = {"visible": False, "distance_m": None, "ground_distance_m": None,
            "bearing_deg": None, "elevation_deg": None,
            "est_forward_m": None, "est_left_m": None, "speed_mps": None}

GOAL_NOT_SEEN = {"visible": False, "bearing_deg": None, "width_deg": None,
                 "distance_m": None}


def detect_ball_pixels(px, fovy_deg: float = HEAD_CAM_FOVY_DEG,
                       radius_m: float = BALL_RADIUS_M) -> dict:
    """Find the orange ball in an RGB frame -> visible/distance/bearing/elevation.

    Pure pixels + pinhole geometry, no scene access: this is everything the
    real mediad could know from one camera frame. Bearing and elevation come
    from the blob centroid's ray (bearing positive to the robot's left, to
    match the yaw-rate sign convention). Range comes from the solid angle the
    blob covers — a sphere of angular radius a covers 2*pi*(1-cos a), and
    sin a = r/d. Solid angle rather than a pixel radius because the projection
    of an off-axis sphere is a stretched ellipse: a flat pixel-radius model
    reads ~30% short by 40 deg off-centre.
    """
    h, w = px.shape[:2]
    f = px.astype(np.int16)
    r, g, b = f[:, :, 0], f[:, :, 1], f[:, :, 2]
    mask = (r > 100) & (r - b > 70) & (r - g > 40)
    if int(mask.sum()) < DET_MIN_PX:
        return dict(NOT_SEEN)
    ys, xs = np.nonzero(mask)
    fl = (h / 2) / math.tan(math.radians(fovy_deg) / 2)  # focal length, pixels
    u = (xs + 0.5 - w / 2) / fl  # +u right, +v up, in focal lengths
    v = (h / 2 - ys - 0.5) / fl
    omega = float(np.sum((1.0 + u * u + v * v) ** -1.5)) / (fl * fl)
    cos_a = min(1.0, max(-1.0, 1.0 - omega / (2 * math.pi)))
    sin_a = math.sqrt(max(1e-12, 1.0 - cos_a * cos_a))
    uc, vc = float(u.mean()), float(v.mean())
    return {
        "visible": True,
        "distance_m": round(radius_m / sin_a, 3),
        "bearing_deg": round(-math.degrees(math.atan(uc)), 3),
        "elevation_deg": round(math.degrees(math.atan(vc / math.sqrt(1 + uc * uc))), 3),
    }


def detect_goal_pixels(px, cam_rot, fovy_deg: float = HEAD_CAM_FOVY_DEG) -> dict:
    """Find the white goal frame in an RGB frame -> visible/bearing/width/range.

    Pixels + pinhole geometry + the camera's own pose (forward kinematics) —
    nothing reads the scene. White mask, then the elevation band that only the
    goal frame occupies (see GOAL_ELEV_* above), then the dense-column filter
    that rejects grazing-angle painted lines. Bearing is the horizontal WORLD
    azimuth of the blob centroid returned relative to the camera's yaw — i.e.
    robust to the head being pitched. Range comes from the angular width of
    the mouth when both edges are inside the frame; a partial view (goal
    half out of frame) publishes bearing but null distance.
    """
    h, w = px.shape[:2]
    f = px.astype(np.int16)
    r, g, b = f[:, :, 0], f[:, :, 1], f[:, :, 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    white = (mn > 150) & (mx - mn < 35)
    ys, xs = np.nonzero(white)
    if len(xs) < GOAL_DET_MIN_PX:
        return dict(GOAL_NOT_SEEN)
    fl = (h / 2) / math.tan(math.radians(fovy_deg) / 2)
    u = (xs + 0.5 - w / 2) / fl
    v = (h / 2 - ys - 0.5) / fl
    d = np.stack([u, v, -np.ones_like(u)], axis=1)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    wd = d @ np.asarray(cam_rot).reshape(3, 3).T  # rays in world frame
    elev = np.degrees(np.arcsin(np.clip(wd[:, 2], -1.0, 1.0)))
    band = (elev > GOAL_ELEV_LO_DEG) & (elev < GOAL_ELEV_HI_DEG)
    if int(band.sum()) < GOAL_DET_MIN_PX:
        return dict(GOAL_NOT_SEEN)
    xb = xs[band]
    cols = np.bincount(xb, minlength=w)
    dense = cols[xb] >= 2
    if int(dense.sum()) < GOAL_DET_MIN_PX:
        return dict(GOAL_NOT_SEEN)
    uk, xk, wk = u[band][dense], xb[dense], wd[band][dense]
    # centroid azimuth in the world's horizontal plane, then back to a
    # camera-yaw-relative bearing (positive left, like ball bearing)
    mean_dir = wk.mean(axis=0)
    az_w = math.atan2(mean_dir[1], mean_dir[0])
    fwd_w = np.asarray(cam_rot).reshape(3, 3) @ np.array([0.0, 0.0, -1.0])
    cam_az = math.atan2(fwd_w[1], fwd_w[0])
    bearing = math.degrees((az_w - cam_az + math.pi) % (2 * math.pi) - math.pi)
    span = math.degrees(math.atan(float(uk.max())) - math.atan(float(uk.min())))
    dist = None
    if xk.min() > 2 and xk.max() < w - 3 and span > 2.0:
        est = (GOAL_MOUTH_OUTER_W / 2) / math.tan(math.radians(span) / 2)
        if 0.2 < est < 4.5:  # the pitch is 3.2 m long; junk reads 15 m
            dist = round(est, 3)
    return {"visible": True, "bearing_deg": round(bearing, 3),
            "width_deg": round(span, 3), "distance_m": dist,
            "_azimuth_w_rad": az_w}


# ---------- the referee (goal scoring) ----------
# Deliberately reads the ball's ground-truth qpos: this is the referee /
# goal-line boundary sensor, world infrastructure any real pitch would have,
# not robot knowledge. The machine digest's honesty rule is about what the
# ROBOT senses (ball_seen.*, proprioception); a scoreboard is allowed to know
# where the ball is, and the robot only learns the score, never the position.

# Geom names probed to decide whether the loaded scene has a goal at all.
GOAL_GEOM_NAMES = ("goal_post_left", "goal_post_right", "goal_crossbar",
                   "goal_frame", "goal_net", "goal")


def find_goal_geom(model) -> str | None:
    """Name of a goal geom in the model, or None if the scene has no goal.

    Tolerant on purpose: the pitch scene is authored elsewhere, so try the
    likely names first and then accept any geom whose name starts with "goal".
    """
    for name in GOAL_GEOM_NAMES:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) >= 0:
            return name
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if name and name.startswith("goal"):
            return name
    return None


def ball_in_goal(x: float, y: float, z: float,
                 radius_m: float = BALL_RADIUS_M) -> bool:
    """Is the ball FULLY across the line and inside the NET?

    Fully across: the trailing edge of the ball has cleared the line, i.e. the
    centre is a whole radius past it. Inside: between the posts, under the
    crossbar, and in front of the back of the net — the enclosed volume that
    is physically reachable only through the mouth. A ball bouncing off the
    top of the frame, or rolling around the outside and stopping BEHIND the
    goal, is not a goal.
    """
    return (GOAL_LINE_X + radius_m < x < GOAL_NET_BACK_X
            and abs(y) < GOAL_HALF_WIDTH_Y
            and z < GOAL_HEIGHT_Z)


class GoalReferee:
    """Scoreboard: counts goals off a stream of ball positions.

    Latched while the ball stays in the goal (one score per shot, not one per
    tick), re-armed once the ball is back out in front of the line.
    """

    def __init__(self, radius_m: float = BALL_RADIUS_M):
        self.radius_m = radius_m
        self.count = 0
        self.scored = False  # latched: the ball is in the goal right now
        self.last_goal_sim_time_s = None

    def reset(self):
        self.count = 0
        self.scored = False
        self.last_goal_sim_time_s = None

    def update(self, x: float, y: float, z: float, sim_time: float) -> bool:
        """Feed one ball position. True on the rising edge of a goal."""
        if ball_in_goal(x, y, z, self.radius_m):
            if self.scored:
                return False  # still in there from the last tick
            self.scored = True
            self.count += 1
            self.last_goal_sim_time_s = round(float(sim_time), 2)
            return True
        if x < GOAL_LINE_X:
            # Back out in front of the line — someone fetched it. Re-arm.
            # Anything short of that (wedged against a post, sitting on the
            # crossbar) leaves the latch alone rather than re-scoring.
            self.scored = False
        return False

    def state(self) -> dict:
        return {"scored": self.scored, "count": self.count,
                "last_goal_sim_time_s": self.last_goal_sim_time_s}


class DuckSim:
    def __init__(self, rl_repo: str, policies_dir: str, scene: str, frames_dir: str):
        self.rl_repo = os.path.abspath(rl_repo)
        self.frames_dir = frames_dir
        os.makedirs(self.frames_dir, exist_ok=True)

        ip = load_infer_policy_module(self.rl_repo)

        xml_path = os.path.join(self.rl_repo, SCENES[scene])
        print(f"Loading MuJoCo model: {xml_path}")
        self.model, mouth_ok = load_model_with_mouth(xml_path)
        self.model.opt.timestep = TIMESTEP
        self.data = mujoco.MjData(self.model)
        self._apply_current_limit(DEFAULT_CURRENT_LIMIT)
        self._fix_ball_contact()
        self._head_cam_id = self._setup_head_camera()

        paths = {}
        for role, fname in POLICY_FILES.items():
            p = os.path.join(policies_dir, fname)
            if os.path.isfile(p):
                paths[role] = p
            else:
                print(f"note: no {role} policy at {p} — skipping")
        if scene == "plain":
            paths.pop("kick_left", None)
            paths.pop("kick_right", None)

        self.policy = ip.PolicyInference(
            self.model, self.data,
            walking_onnx_path=paths.get("walking"),
            standing_onnx_path=paths.get("standing"),
            sitstand_onnx_path=paths.get("sitstand"),
            ground_pick_onnx_path=paths.get("ground_pick"),
            kick_left_onnx_path=paths.get("kick_left"),
            kick_right_onnx_path=paths.get("kick_right"),
            roulade_onnx_path=paths.get("roulade"),
            new_cmd_obs=True,
            use_projected_gravity=True,
        )
        # Velocity command limits matching training ranges (walking robot).
        self.policy.vel_max_x, self.policy.vel_min_x = 0.3, -0.3
        self.policy.vel_max_y, self.policy.vel_min_y = 0.2, -0.2
        self.policy.vel_max_ang = 1.5
        # Zero command selects the standing policy before the first tick.
        self.policy.set_vel_cmd(0.0, 0.0, 0.0)

        # Initial stance (matches infer_policy main()).
        fj = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
        self.qpos_adr = int(self.model.jnt_qposadr[fj])
        self.qvel_adr = int(self.model.jnt_dofadr[fj])
        self.data.qpos[self.qpos_adr:self.qpos_adr + 3] = [0.0, 0.0, 0.125]
        self.data.qpos[self.qpos_adr + 3:self.qpos_adr + 7] = [1, 0, 0, 0]
        for i, qi in enumerate(self.policy.joint_qpos_indices):
            self.data.qpos[qi] = self.policy.default_pose[i]
        if scene == "pitch" and self.policy.ball_qpos_adr is not None:
            # Kickoff from a metre out, not the scene's point-blank penalty
            # spot — snapshot before _qpos0 so reset restarts the match here.
            ba = self.policy.ball_qpos_adr
            self.data.qpos[ba:ba + 2] = PITCH_BALL_SPAWN
        self.data.ctrl[:] = self.policy.default_pose
        mujoco.mj_forward(self.model, self.data)
        self._qpos0 = self.data.qpos.copy()

        self._requests: "queue.Queue[tuple[dict, queue.Queue]]" = queue.Queue()
        self._renderer = None
        self._frame_count = 0
        self.sim_time = 0.0
        self._det_renderer = None
        self._det_off = False
        self._det_step = 0
        self._ball_seen = dict(NOT_SEEN)
        self._ball_seen_t = 0.0
        self._ball_track = deque()  # (t, x, y) world estimates -> speed_mps
        self._goal_seen = dict(GOAL_NOT_SEEN)
        self._goal_seen_t = 0.0
        # Dead-reckoned goal memory: the goal is world-fixed, so a sighting
        # plus own odometry keeps an estimate alive while the head is down
        # tracking the ball. _goal_fix is a world (x, y) position (needs a
        # ranged sighting); _goal_azimuth_w a bearing-only memory.
        self._goal_fix = None
        self._goal_azimuth_w = None
        self.machine = None  # loaded behavior machine (machine.py), or None
        # Scoring only exists where a goal does: no goal geom (or no ball) and
        # the referee stays None, so state/digest look exactly as they did.
        self.referee = None
        goal_geom = find_goal_geom(self.model)
        if goal_geom is not None and self.policy.ball_qpos_adr is not None:
            self.referee = GoalReferee()
            print(f"Goal in scene (geom {goal_geom!r}) — scoring on: line "
                  f"x>{GOAL_LINE_X + BALL_RADIUS_M:.3f}, |y|<{GOAL_HALF_WIDTH_Y}, "
                  f"z<{GOAL_HEIGHT_Z}")
        # The mouth: a commanded opening in [0, 1] (robot.mouth semantics),
        # applied to the mocap plate every tick. Purely cosmetic — the plate
        # has no dynamics and the policy never sees it.
        self.mouth_opening = 0.0
        # Wall clock of the last `mouth` intent off the socket — how the sim
        # knows the beak is somebody else's right now (see _say_owns_mouth).
        self._mouth_intent_t = 0.0
        # A voice.SayPlayer if this host can talk (set by main); None means a
        # machine's `say` lines are annotations only — the honest default for
        # a robot whose speaker lives on somebody else's computer.
        self.voice = None
        # The emote directory and the gesture playing out of it right now
        # (both set by main / start_emote). Emotes are data the sim reads, so
        # a server started without a directory simply has no body language.
        self.emotes = None
        self.voice_bank = None
        self._emote = None
        self._mouth_mocap_id = -1
        self._mouth_head_id = -1
        if mouth_ok:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "mouth_plate")
            self._mouth_mocap_id = int(self.model.body_mocapid[bid])
            self._mouth_head_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, MOUTH_BODY)
            self.mouth_tick()  # park the plate on the head before first render
            mujoco.mj_forward(self.model, self.data)
        # Command-feed ring buffer for the AX debug page (webui.py).
        self.events = deque(maxlen=500)
        self._event_id = 0
        # The wake latch (ocarina's wake pack, robot-adapted): entering a
        # machine node that declares `wake = "..."` parks a pack here, and a
        # `machine wait` long-polls it from its own connection thread — never
        # the request queue, which the 50 Hz sim thread drains synchronously.
        # The robot cannot freeze like a paused game, so the wake node's own
        # behavior is the holding pattern and its deadline transition (or
        # declared wake_hold) is the no-answer default.
        self._wake_cond = threading.Condition()
        self._wakes = deque(maxlen=16)  # parked packs, oldest first
        self._wake_id = 0

    def _fix_ball_contact(self):
        """Give the ball real rolling resistance (see BALL_ROLL_FRICTION)."""
        gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
        if gid < 0:
            return
        self.model.geom_condim[gid] = 6
        self.model.geom_friction[gid, 2] = BALL_ROLL_FRICTION

    def _setup_head_camera(self) -> int:
        """Re-pose the model's head_camera (see HEAD_CAM_* above)."""
        cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
        if cid < 0:
            print("note: model has no head_camera — view='head' unavailable")
            return -1
        # Head body: -z forward, +x up. Rotate -90 deg about z to line the
        # camera frame up with that, then pitch down about its own x axis.
        a, b = math.radians(-90) / 2, math.radians(-HEAD_CAM_PITCH_DEG) / 2
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, np.array([math.cos(a), 0.0, 0.0, math.sin(a)]),
                           np.array([math.cos(b), math.sin(b), 0.0, 0.0]))
        self.model.cam_quat[cid] = q
        self.model.cam_pos[cid, 2] -= HEAD_CAM_FORWARD_M
        self.model.cam_fovy[cid] = HEAD_CAM_FOVY_DEG
        return int(cid)

    def mouth_tick(self):
        """Pose the mouth plate from the head's kinematics + the opening.

        Runs before each control step (and once at init/reset), so the next
        mj_step folds the mocap pose into xpos for the viewer and renderers.
        """
        if self._mouth_mocap_id < 0:
            return
        pos, quat = mouth_pose(self.data.xpos[self._mouth_head_id],
                               self.data.xquat[self._mouth_head_id],
                               self.mouth_opening)
        self.data.mocap_pos[self._mouth_mocap_id] = pos
        self.data.mocap_quat[self._mouth_mocap_id] = quat

    def _apply_current_limit(self, amps: float):
        try:
            from bam.model import load_model
            kt = load_model(motor_name="xl330", model="m6").kt.value
        except Exception:
            kt = KT_XL330
        limit = kt * amps
        self.model.actuator_forcerange[:, 0] = -limit
        self.model.actuator_forcerange[:, 1] = limit
        self.model.actuator_forcelimited[:] = 1
        print(f"Current limit {amps:.2f} A -> torque clamp ±{limit:.4f} Nm (kt={kt:.4f})")

    # ---------- sensing (sim thread only) ----------

    def sense(self):
        """Run the head-camera ball detector on its own, slower cadence."""
        self._det_step += 1
        if self._det_step % DET_EVERY == 0:
            self._detect_ball()

    def _detect_ball(self):
        # No ball in the scene -> nothing the detector could ever find, so skip
        # the render rather than burn 15 ms of the control period on it.
        if self._det_off or self._head_cam_id < 0 or self.policy.ball_qpos_adr is None:
            return
        try:
            if self._det_renderer is None:
                self._det_renderer = mujoco.Renderer(self.model, height=DET_H, width=DET_W)
            self._det_renderer.update_scene(self.data, camera=self._head_cam_id)
            px = self._det_renderer.render()
        except Exception as e:
            self._det_off = True
            print(f"note: head-camera detector disabled ({e})")
            return
        seen = detect_ball_pixels(px)
        if seen["visible"]:
            # Derived features a real robot could compute too: the camera pose
            # comes from its own forward kinematics, the ray and range from the
            # detection. Nothing here reads the ball's actual state.
            cam_pos = self.data.cam_xpos[self._head_cam_id]
            h = float(cam_pos[2]) - BALL_RADIUS_M
            d = seen["distance_m"]
            seen["ground_distance_m"] = round(math.sqrt(max(0.0, d * d - h * h)), 3)
            # Camera-frame ray from bearing/elevation (inverting the detector's
            # angles), through the camera pose, into the trunk's yaw frame —
            # so guards and behaviors get the same {forward,left} vocabulary
            # the kick pocket was measured in.
            uc = -math.tan(math.radians(seen["bearing_deg"]))
            vc = math.tan(math.radians(seen["elevation_deg"])) * math.sqrt(1 + uc * uc)
            ray = np.array([uc, vc, -1.0])
            ray /= np.linalg.norm(ray)
            R = self.data.cam_xmat[self._head_cam_id].reshape(3, 3)
            ball_w = np.asarray(cam_pos) + d * (R @ ray)
            adr = self.qpos_adr
            tp = self.data.qpos[adr:adr + 3]
            _, _, yaw = quat_to_rpy(self.data.qpos[adr + 3:adr + 7])
            dx, dy = float(ball_w[0] - tp[0]), float(ball_w[1] - tp[1])
            c, s = math.cos(yaw), math.sin(yaw)
            seen["est_forward_m"] = round(c * dx + s * dy, 3)
            seen["est_left_m"] = round(-s * dx + c * dy, 3)
            # Ball speed, honestly: difference the world-frame estimate across
            # detector ticks. The estimate already folds in the robot's own
            # kinematics, so the duck's motion cancels and a parked ball reads
            # ~0 even mid-stride. The camera bias wobbles with the gait, hence
            # a multi-tick baseline; a fresh sighting publishes null until the
            # track spans DET_SPEED_MIN_DT_S (guards treat null as "not < x").
            seen["speed_mps"] = None
            self._ball_track.append((self.sim_time, float(ball_w[0]), float(ball_w[1])))
            while self.sim_time - self._ball_track[0][0] > DET_SPEED_WINDOW_S:
                self._ball_track.popleft()
            t0, x0, y0 = self._ball_track[0]
            if self.sim_time - t0 >= DET_SPEED_MIN_DT_S:
                seen["speed_mps"] = round(
                    math.hypot(float(ball_w[0]) - x0, float(ball_w[1]) - y0)
                    / (self.sim_time - t0), 3)
        self._ball_seen = seen
        if seen["visible"]:
            # age_s = time since last SIGHTING (matches goal_seen.age_s): it
            # lets a guard say "unseen for 2 s" — one blind tick mid-maneuver
            # is a blink, two seconds is a lost ball.
            self._ball_seen_t = self.sim_time
        self._detect_goal(px)

    def _detect_goal(self, px):
        """Goal sighting from the SAME frame the ball detector rendered.

        Only on scenes with a goal (a mediad configured for the pitch knows
        the pitch has one). On a sighting, refresh the dead-reckoned memory:
        world azimuth always, world position when the sighting was ranged.
        """
        if self.referee is None:
            return
        cam_rot = self.data.cam_xmat[self._head_cam_id]
        gs = detect_goal_pixels(px, cam_rot)
        az_w = gs.pop("_azimuth_w_rad", None)
        self._goal_seen = gs
        if not gs["visible"]:
            return
        self._goal_seen_t = self.sim_time
        self._goal_azimuth_w = az_w
        if gs["distance_m"] is not None:
            cam = self.data.cam_xpos[self._head_cam_id]
            fx = float(cam[0]) + gs["distance_m"] * math.cos(az_w)
            fy = float(cam[1]) + gs["distance_m"] * math.sin(az_w)
            if self._goal_fix is None:
                self._goal_fix = (fx, fy)
            else:  # range is the noisy axis (±30%); smooth across sightings
                ox, oy = self._goal_fix
                self._goal_fix = (0.6 * ox + 0.4 * fx, 0.6 * oy + 0.4 * fy)

    def _goal_estimates(self):
        """(est_bearing_deg, est_distance_m) to the remembered goal, from own
        pose — live every tick, even with the goal out of frame. Nulls until
        the goal has been sighted at all."""
        if self._goal_azimuth_w is None:
            return None, None
        adr = self.qpos_adr
        tp = self.data.qpos[adr:adr + 3]
        _, _, yaw = quat_to_rpy(self.data.qpos[adr + 3:adr + 7])
        if self._goal_fix is not None:
            dx, dy = self._goal_fix[0] - float(tp[0]), self._goal_fix[1] - float(tp[1])
            az = math.atan2(dy, dx)
            dist = round(math.hypot(dx, dy), 3)
        else:  # bearing-only memory: goal treated as distant along last azimuth
            az, dist = self._goal_azimuth_w, None
        bear = math.degrees((az - yaw + math.pi) % (2 * math.pi) - math.pi)
        return round(bear, 3), dist

    # ---------- the referee (sim thread only) ----------

    def referee_tick(self):
        """Watch the ball across the goal line. No-op on goal-less scenes."""
        if self.referee is None:
            return
        adr = self.policy.ball_qpos_adr
        x, y, z = (float(v) for v in self.data.qpos[adr:adr + 3])
        if self.referee.update(x, y, z, self.sim_time):
            self._event_id += 1
            self.events.append({
                "id": self._event_id, "t": time.time(), "client": "referee",
                "cmd": "GOAL!",
                "args": {"count": self.referee.count,
                         "ball_position_m": [round(v, 3) for v in (x, y, z)]},
                "ok": True,
                "note": f"goal {self.referee.count} at "
                        f"t={self.referee.last_goal_sim_time_s}s",
            })
            print(f"GOAL! #{self.referee.count} at t="
                  f"{self.referee.last_goal_sim_time_s}s  "
                  f"ball=({x:.3f}, {y:.3f}, {z:.3f})")

    # ---------- emotes (sim thread only) ----------

    def _set_head_offset(self, vals):
        """Point the head through the policy's gaze command — the same path
        the `look` intent and the behaviors' _set_head use, so the balance
        policy compensates for the pose instead of being surprised by it.
        Order is (neck_pitch, head_pitch, head_yaw, head_roll), clamped."""
        p = self.policy
        p.head_offset[:] = np.clip(np.asarray(vals, dtype=np.float32),
                                   -p.head_max, p.head_max)
        p._update_command()

    def _say_owns_mouth(self) -> bool:
        """Is the beak busy being a voice? Two ways it can be: the machine's
        own SayPlayer is mid-line, or a `duck say` somewhere else is streaming
        `mouth` intents at us. Either way an emote's beak channel yields —
        words beat gestures on that one channel, because a beak fighting a
        sentence reads as a glitch, not as a mood."""
        if self.voice is not None and self.voice.busy:
            return True
        return time.time() - self._mouth_intent_t < MOUTH_HELD_S

    def _head_bound_node(self):
        """The node whose behavior currently needs the head, or None."""
        m = self.machine
        if m is None or not m.armed:
            return None
        if m.nodes[m.current]["behavior"] in HEAD_BOUND_BEHAVIORS:
            return m.current
        return None

    def start_emote(self, name: str, machine: bool = False) -> dict:
        """Begin a gesture — or say who has the head instead.

        Channel ownership lives here, in one place: say > emote for the beak,
        emote > behavior for the head, but only where the behavior can spare
        it. A MACHINE-triggered emote skips the ownership check, because the
        machine's author already made that call in source — that is the
        difference between an interruption and a decision.

        A gesture arriving mid-gesture is refused rather than queued or
        restarted, the same honesty as SayPlayer's dropped line: a duck that
        keeps restarting a nod looks broken, not busy.
        """
        if self.emotes is None:
            return {"ok": False, "error": "this server has no emote directory "
                    "(start duck-sim with --emotes DIR)"}
        try:
            emote = self.emotes.get(name)
        except EmoteError as e:
            return {"ok": False, "error": str(e)}
        if self._emote is not None:
            return {"ok": False, "error": f"mid-emote ({self._emote['name']}) "
                    f"— a duck restarting a gesture looks broken"}
        node = None if machine else self._head_bound_node()
        if node is not None:
            return {"ok": False, "error": f"the head belongs to {node!r} right "
                    f"now — that behavior steers by the head camera, so it "
                    f"keeps the head until it is done"}
        traj = emote.render(CONTROL_HZ)
        self._emote = {
            "name": emote.name, "t0": self.sim_time, "traj": traj,
            "n": len(traj["mouth"]),
            # Where the head was when the gesture borrowed it. Restored on
            # completion: an emote is a detour, not a new resting pose.
            "restore": np.array(self.policy.head_offset, dtype=np.float32),
        }
        note = self._play_bank_sound(emote.sound) if emote.sound else ""
        return {"ok": True, "emote": emote.name, "sound": emote.sound,
                "duration_s": round(emote.duration, 3), "note": note}

    def _play_bank_sound(self, tag: str) -> str:
        """Launch an emote's voice-bank call off the sim thread. Never raises.

        Same rule as every other noise the duck makes: audio is the host's job
        and best-effort, and the reasons it might not happen (no bank, no
        afplay, the duck already talking) are notes on the event, not errors.
        Returns that note, empty when the sound went out.
        """
        if self._say_owns_mouth():
            return f"{tag} skipped: the duck is already talking"
        from . import voice
        try:
            path = voice.bank_wav_path(self.voice_bank, tag)
        except voice.VoiceError as e:
            return f"{tag} skipped: {e}"
        player = shutil.which("afplay")
        if player is None:
            return f"{tag} skipped: no `afplay` on this host"
        threading.Thread(target=subprocess.run, args=([player, path],),
                         kwargs={"check": False}, daemon=True,
                         name="duck-emote-sound").start()
        return ""

    def emote_tick(self):
        """One control step of the gesture that is playing, if one is.

        Called AFTER machine_tick, and that order IS the arbitration for the
        head: whatever the current behavior just asked the head to do this
        tick, the emote overwrites. The beak goes the other way — a voice owns
        it while it has something to say.
        """
        e = self._emote
        if e is None:
            return
        i = int(round((self.sim_time - e["t0"]) * CONTROL_HZ))
        if i >= e["n"]:
            self._end_emote(e)
            return
        traj = e["traj"]
        self._set_head_offset([traj[c][i] for c in HEAD_CHANNELS])
        if not self._say_owns_mouth():
            self.mouth_opening = float(traj["mouth"][i])

    def _end_emote(self, e: dict):
        """Hand the head back to whoever had it, and shut the beak."""
        self._set_head_offset(e["restore"])
        if not self._say_owns_mouth():
            self.mouth_opening = 0.0
        self._emote = None

    def _handle_emote(self, req: dict) -> dict:
        if req.get("action") == "list":
            if self.emotes is None:
                return {"ok": False, "error": "this server has no emote "
                        "directory (start duck-sim with --emotes DIR)"}
            return {"ok": True, "dir": self.emotes.dir,
                    "emotes": self.emotes.listing(),
                    "playing": self._emote["name"] if self._emote else None}
        return self.start_emote(str(req.get("name", "")))

    # ---------- the behavior machine (sim thread only) ----------

    def _machine_digest(self) -> dict:
        """The guard vocabulary: sensed ball + proprioception + machine time.
        Deliberately excludes ground-truth ball position — a machine cannot
        act on knowledge the robot would not have. The referee's score is in
        here (a scoreboard is world infrastructure, and "someone scored" is
        not "where the ball is"); its keys are always present, False/0 on a
        scene without a goal, so the key set is stable across scenes."""
        proj_g = self.policy.get_projected_gravity()
        bs = self._ball_seen
        gs = self._goal_seen
        g_bear, g_dist = self._goal_estimates()
        r = self.referee
        return {
            "ball_seen.visible": bs["visible"],
            "ball_seen.distance_m": bs["distance_m"],
            "ball_seen.ground_distance_m": bs.get("ground_distance_m"),
            "ball_seen.bearing_deg": bs["bearing_deg"],
            "ball_seen.elevation_deg": bs["elevation_deg"],
            "ball_seen.est_forward_m": bs.get("est_forward_m"),
            "ball_seen.est_left_m": bs.get("est_left_m"),
            "ball_seen.speed_mps": bs.get("speed_mps"),
            "ball_seen.age_s": round(self.sim_time - self._ball_seen_t, 3),
            "goal_seen.visible": gs["visible"],
            "goal_seen.bearing_deg": gs["bearing_deg"],
            "goal_seen.width_deg": gs["width_deg"],
            "goal_seen.distance_m": gs["distance_m"],
            "goal_seen.age_s": round(self.sim_time - self._goal_seen_t, 3),
            "goal_seen.est_bearing_deg": g_bear,
            "goal_seen.est_distance_m": g_dist,
            "goal.scored": bool(r.scored) if r is not None else False,
            "goal.count": int(r.count) if r is not None else 0,
            "upright": bool(proj_g[2] < -0.7),
            "sitting": bool(self.policy.sit_mode),
            "active_policy": self.policy.current_policy,
            "behavior": self.policy.behavior_mode,
            "sim_time_s": self.sim_time,
            "node": self.machine.current if self.machine else None,
        }

    def machine_tick(self):
        if self.machine is None or not self.machine.armed:
            return
        digest = self._machine_digest()
        fired = self.machine.tick(self, digest)
        if fired:
            self._event_id += 1
            self.events.append({
                "id": self._event_id, "t": time.time(), "client": "machine",
                "cmd": f"-> {fired['to']}",
                "args": {"from": fired["from"], "when": fired["when"]},
                "ok": True, "note": "",
            })
            print(f"machine: {fired['from']} -> {fired['to']}  [{fired['when']}]")
            if self.machine.nodes[fired["from"]]["wake"] is not None:
                self._resolve_wakes(fired["from"], fired)
            say = self.machine.nodes[fired["to"]].get("say")
            if say:
                self._say_line(fired["to"], say)
            emote = self.machine.nodes[fired["to"]].get("emote")
            if emote:
                self._emote_node(fired["to"], emote)
            wake = self.machine.nodes[fired["to"]]["wake"]
            if wake is not None:
                self._latch_wake(fired["to"], wake,
                                 {"from": fired["from"], "when": fired["when"]},
                                 digest)

    def _say_line(self, node: str, text: str):
        """A speaking node's line: onto the control surface, then into the air.

        The annotation goes through the same `say` verb `duck say` uses, so a
        line the machine decided to say and a line a person asked for look
        identical on the event feed — and the film's control-surface feed
        picks it up for free.

        Speaking is best-effort and belongs to the host: the robot has a mouth
        servo, not a speaker. Without a voice this session the line is still an
        event, which is the point of it being an annotation.
        """
        resp = self.handle({"cmd": "say", "text": text})
        self._log_event("machine", {"cmd": "say", "node": node, "text": text},
                        resp)
        if self.voice is not None:
            try:
                self.voice.speak(text, self)
            except Exception as e:      # this runs in the 50 Hz control loop:
                print(f"note: voice failed ({e})", file=sys.stderr)  # nothing
                self.voice = None       # about talking may stall the walking

    def _emote_node(self, node: str, name: str):
        """An emoting node's gesture: onto the control surface, then the head.

        `say`'s twin, deliberately — the mouth says the line, the body plays
        the gesture, and both land on the event feed as the same kind of act.
        The machine's trigger outranks the head-ownership refusal a client
        would meet (start_emote's `machine` flag): the author wrote this
        gesture on this node, so declining it would be second-guessing source.

        A gesture that cannot play — an emote nobody has, a directory that
        isn't there, another gesture still running — is a note on the feed and
        nothing more. The machine is mid-run; there is no one to raise at.
        """
        resp = self.start_emote(name, machine=True)
        self._log_event("machine",
                        {"cmd": "emote", "node": node, "name": name}, resp)
        if not resp.get("ok"):
            print(f"note: {node} could not emote {name!r} ({resp['error']})",
                  file=sys.stderr)

    def _emote_warnings(self, m: Machine) -> list:
        """Which of a machine's gestures this server cannot actually play.

        A lint, run at load and reload: the machine is valid either way (the
        grammar knows nothing about emotes), but a duck that will silently not
        startle is worth a line in the response before it doesn't.
        """
        out = []
        for node in sorted(m.nodes):
            name = m.nodes[node]["emote"]
            if name is None:
                continue
            if self.emotes is None:
                out.append(f"node {node!r} emotes {name!r}, but this server "
                           f"has no emote directory")
                continue
            try:
                self.emotes.get(name)
            except EmoteError as e:
                out.append(f"node {node!r}: {e}")
        return out

    def _latch_wake(self, node: str, reason: str, via: dict, digest: dict):
        """Sim thread: park a wake pack and release any blocked machine_wait.
        The pack carries what ocarina's does — reason, digest snapshot, event
        tail, the transition that fired — so the mind wakes into context, not
        into a contextless prompt."""
        self._event_id += 1
        self.events.append({
            "id": self._event_id, "t": time.time(), "client": "machine",
            "cmd": "wake", "args": {"node": node, **via}, "ok": True,
            "note": reason,
        })
        pack = {
            "reason": reason, "node": node, "via": via,
            "sim_time_s": round(self.sim_time, 3),
            "digest": digest,
            "events": list(self.events)[-8:],
            "resolved": None,
        }
        with self._wake_cond:
            self._wake_id += 1
            pack["id"] = self._wake_id
            self._wakes.append(pack)
            self._wake_cond.notify_all()
        print(f"machine: WAKE [{node}] {reason}")

    def _resolve_wakes(self, node: str, fired: dict):
        """Sim thread: the machine left a wake node on its own (the deadline
        default, or any other guard firing first) — mark still-parked packs
        for that node, so a late listener learns the body already answered
        itself, and how."""
        with self._wake_cond:
            for pack in self._wakes:
                if pack["node"] == node and pack["resolved"] is None:
                    pack["resolved"] = {"to": fired["to"],
                                        "when": fired["when"],
                                        "sim_time_s": round(self.sim_time, 3)}

    def _maybe_entry_wake(self, via: dict):
        """Sim thread: arm/force landed the machine directly on a wake node."""
        m = self.machine
        if m is not None and m.armed and m.nodes[m.current]["wake"] is not None:
            self._latch_wake(m.current, m.nodes[m.current]["wake"], via,
                             self._machine_digest())

    def machine_wait(self, block_s: float = 55.0) -> dict:
        """CLIENT-connection thread: long-poll the wake latch. Returns the
        oldest parked pack, or an honest no_wake once block_s expires — the
        caller re-arms with another wait (ocarina's max_block_s pattern, for
        clients whose request timeouts are shorter than a quiet afternoon)."""
        block_s = max(0.0, min(float(block_s), 600.0))
        t0 = time.monotonic()
        deadline = t0 + block_s
        with self._wake_cond:
            while True:
                if self._wakes:
                    return {"ok": True, "wake": self._wakes.popleft()}
                m = self.machine
                if m is None:
                    return {"ok": False,
                            "error": "no machine loaded (action=load first)"}
                if not m.armed:
                    return {"ok": False, "error": "machine is disarmed — "
                            "nothing will ever wake (arm it first)"}
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return {"ok": True, "no_wake": True,
                            "waited_s": round(time.monotonic() - t0, 1)}
                # Short slices so a disarm/unload during the wait is noticed.
                self._wake_cond.wait(timeout=min(remaining, 0.5))

    def _handle_machine(self, req: dict) -> dict:
        action = req.get("action", "status")
        m = self.machine
        if action == "load":
            path = req.get("path")
            if not path:
                return {"ok": False, "error": "load needs a path"}
            try:
                self.machine = Machine.load(path)
            except (MachineError, OSError) as e:
                return {"ok": False, "error": f"machine rejected: {e}"}
            return {"ok": True, **self.machine.status(),
                    "note": "loaded, disarmed — arm to run",
                    "warnings": self._emote_warnings(self.machine)}
        if m is None:
            return {"ok": False, "error": "no machine loaded (action=load first)"}
        if action == "reload":
            try:
                fresh = Machine.load(m.source_path)
            except (MachineError, OSError) as e:
                return {"ok": False, "error": f"reload rejected, old machine "
                        f"kept: {e}"}
            fresh.armed = m.armed
            fresh.enter(fresh.initial if m.current not in fresh.nodes
                        else m.current, self.sim_time)
            self.machine = fresh
            return {"ok": True, **fresh.status(), "note": "hot-swapped",
                    "warnings": self._emote_warnings(fresh)}
        if action == "arm":
            m.enter(m.initial, self.sim_time)
            m.armed = True
            self._maybe_entry_wake({"action": "arm"})
            return {"ok": True, **m.status()}
        if action == "disarm":
            m.armed = False
            self.policy.set_vel_cmd(0.0, 0.0, 0.0)
            return {"ok": True, **m.status()}
        if action == "force":
            node = req.get("node")
            if node not in m.nodes:
                return {"ok": False, "error": f"unknown node {node!r} "
                        f"(have: {', '.join(sorted(m.nodes))})"}
            m.enter(node, self.sim_time)
            self._maybe_entry_wake({"action": "force"})
            return {"ok": True, **m.status()}
        if action == "status":
            return {"ok": True, **m.status()}
        return {"ok": False, "error": f"unknown machine action {action!r}"}

    # ---------- state ----------

    def get_state(self) -> dict:
        adr, vadr = self.qpos_adr, self.qvel_adr
        pos = self.data.qpos[adr:adr + 3]
        quat = self.data.qpos[adr + 3:adr + 7].astype(np.float32)
        roll, pitch, yaw = quat_to_rpy(quat)
        v_world = np.array(self.data.qvel[vadr:vadr + 3], dtype=np.float32)
        v_body = self.policy.quat_rotate_inverse(quat, v_world)
        proj_g = self.policy.get_projected_gravity()
        p = self.policy
        state = {
            "ok": True,
            "sim_time_s": round(self.sim_time, 2),
            "position_m": [round(float(v), 3) for v in pos],
            "rpy_deg": [round(math.degrees(a), 1) for a in (roll, pitch, yaw)],
            "vel_body_mps": {"forward": round(float(v_body[0]), 3),
                             "lateral": round(float(v_body[1]), 3)},
            "yaw_rate_rps": round(float(self.data.qvel[vadr + 5]), 3),
            "trunk_height_mm": round(float(pos[2]) * 1000, 1),
            "upright": bool(proj_g[2] < -0.7),
            "active_policy": p.current_policy,
            "vel_cmd": [round(float(v), 2) for v in p.vel_cmd],
            "sitting": bool(p.sit_mode),
            "behavior": p.behavior_mode,
            "ground_pick": bool(p.ground_pick_mode),
            # Camera-derived, unlike ball_position_m below: what the head
            # camera actually saw, as the real robot's mediad would report it.
            "mouth": (round(self.mouth_opening, 3)
                      if self._mouth_mocap_id >= 0 else None),
            "ball_seen": {**self._ball_seen,
                          "age_s": round(self.sim_time - self._ball_seen_t, 3)},
            "machine": ({"name": self.machine.name, "armed": self.machine.armed,
                         "node": self.machine.current}
                        if self.machine is not None else None),
        }
        if p.ball_qpos_adr is not None:
            b = self.data.qpos[p.ball_qpos_adr:p.ball_qpos_adr + 3]
            state["ball_position_m"] = [round(float(v), 3) for v in b]
            # Ball offset in the robot's yaw frame — the frame kick training
            # uses (x forward, y left). Kick staging places the ball at
            # x=0.09, y=±0.042; an unstaged kick needs it near there.
            dx, dy = float(b[0] - pos[0]), float(b[1] - pos[1])
            c, s = math.cos(yaw), math.sin(yaw)
            state["ball_offset_m"] = {"forward": round(c * dx + s * dy, 3),
                                      "left": round(-s * dx + c * dy, 3)}
        # Only on a scene with a goal — omitted entirely elsewhere.
        if self.referee is not None:
            state["goal"] = self.referee.state()
            g_bear, g_dist = self._goal_estimates()
            state["goal_seen"] = {
                **self._goal_seen,
                "age_s": round(self.sim_time - self._goal_seen_t, 3),
                "est_bearing_deg": g_bear, "est_distance_m": g_dist}
        return state

    # ---------- request handlers (sim thread only) ----------

    def handle(self, req: dict) -> dict:
        cmd = req.get("cmd")
        p = self.policy
        if cmd == "ping":
            return {"ok": True, "server": "microduck-sim", "sim_time_s": round(self.sim_time, 2)}
        if cmd == "state":
            return self.get_state()
        if cmd == "set_velocity":
            vx = float(np.clip(req.get("vx", 0.0), p.vel_min_x, p.vel_max_x))
            vy = float(np.clip(req.get("vy", 0.0), p.vel_min_y, p.vel_max_y))
            wz = float(np.clip(req.get("wz", 0.0), -p.vel_max_ang, p.vel_max_ang))
            p.set_vel_cmd(vx, vy, wz)
            return self.get_state()
        if cmd == "trick":
            return self._handle_trick(req.get("name", ""),
                                      stage_ball=bool(req.get("stage_ball", True)))
        if cmd == "look":
            self._set_head_offset([req.get(k, 0.0) for k in HEAD_CHANNELS])
            return self.get_state()
        if cmd == "push":
            mag = float(np.clip(req.get("magnitude", 1.0), 0.0, 2.0))
            angle = math.radians(req["angle_deg"]) if "angle_deg" in req else random.uniform(0, 2 * math.pi)
            self.data.qvel[self.qvel_adr + 0] = mag * math.cos(angle)
            self.data.qvel[self.qvel_adr + 1] = mag * math.sin(angle)
            return {"ok": True, "pushed": {"magnitude": mag, "angle_deg": round(math.degrees(angle), 1)}}
        if cmd == "mouth":
            # robot.mouth semantics: a continuous opening intent, clamped.
            # Ambient (streamed at ~40 Hz by `duck say`), so _log_event skips
            # it — the say annotation below is the loggable act. The timestamp
            # is how an emote knows the beak is spoken for.
            self.mouth_opening = float(np.clip(req.get("opening", 0.0), 0.0, 1.0))
            self._mouth_intent_t = time.time()
            return {"ok": True, "mouth": round(self.mouth_opening, 3)}
        if cmd == "say":
            # Annotation only: speech is rendered and played host-side (the
            # sim has no speaker); this puts the act on the control surface —
            # the event feed and the film's feed — like any other intent.
            text = str(req.get("text", ""))[:200]
            return {"ok": True, "text": text,
                    "duration_s": req.get("duration_s")}
        if cmd == "chirp":
            return self._handle_chirp(str(req.get("tag", "")))
        if cmd == "emote":
            return self._handle_emote(req)
        if cmd == "camera":
            return self._handle_camera(req)
        if cmd == "camera_web":
            return self._handle_camera(req, live=True)
        if cmd == "reset":
            return self._handle_reset()
        if cmd == "machine":
            return self._handle_machine(req)
        return {"ok": False, "error": f"unknown cmd {cmd!r}"}

    def _handle_chirp(self, tag: str) -> dict:
        """A voice-bank call, annotated — and the one call the server can veto.

        Like `say`, the sound itself is played host-side and this is only the
        act landing on the control surface. Unlike `say`, one tag is not the
        caller's to spend: `wheee` belongs to a goal the duck actually scored,
        so the server checks the referee's board rather than trusting whoever
        is holding the socket. The film has had this rule since it had sound
        (soundtrack.py: one goal, one wheee); here it stops being the film's
        discipline and becomes the duck's.

        No referee means no goal line means no earned goals — a scene without
        a goal in it cannot produce one, so the answer there is no.
        """
        if tag == WHEEE_TAG:
            if self.referee is None or self.referee.count == 0:
                return {"ok": False, "error": WHEEE_REFUSAL}
        return {"ok": True, "tag": tag,
                "goals": self.referee.count if self.referee else 0}

    def _handle_trick(self, name: str, stage_ball: bool = True) -> dict:
        p = self.policy
        if name == "sit":
            if not p.sit_mode:
                p.toggle_sit()
            started = p.sit_mode
        elif name == "stand":
            if p.sit_mode:
                p.toggle_sit()
            else:
                p.set_vel_cmd(0.0, 0.0, 0.0)
            started = True
        elif name == "ground_pick":
            p.trigger_ground_pick()
            started = p.ground_pick_mode
        elif name in ("kick_left", "kick_right", "roulade"):
            if not stage_ball and p.ball_qpos_adr is not None:
                # Honest mode: trigger_behavior teleports the ball to the foot
                # (matching the training reset); snapshot and restore so the
                # kick runs against wherever the ball actually is.
                adr, vadr = p.ball_qpos_adr, p.ball_qvel_adr
                ball_qpos = self.data.qpos[adr:adr + 7].copy()
                ball_qvel = self.data.qvel[vadr:vadr + 6].copy()
                p.trigger_behavior(name)
                self.data.qpos[adr:adr + 7] = ball_qpos
                self.data.qvel[vadr:vadr + 6] = ball_qvel
            else:
                p.trigger_behavior(name)
                if p.ball_qpos_adr is not None:
                    # Staging teleported the ball; don't let the detector's
                    # speed track read the jump as a 2 m/s ball.
                    self._ball_track.clear()
            started = p.behavior_mode == name
        else:
            return {"ok": False, "error": f"unknown trick {name!r} "
                    "(sit, stand, ground_pick, kick_left, kick_right, roulade)"}
        out = self.get_state()
        out["trick"] = name
        out["started"] = started
        if not started:
            out["ok"] = False
            out["error"] = f"{name} refused (busy: policy={p.current_policy}, sitting={p.sit_mode})"
        return out

    def _handle_reset(self) -> dict:
        p = self.policy
        self.data.qpos[:] = self._qpos0
        self.data.qvel[:] = 0.0
        p.last_action = np.zeros(p.n_joints, dtype=np.float32)
        p.vel_cmd = np.zeros(3, dtype=np.float32)
        p.head_offset[:] = 0.0
        p.body_cmd[:] = 0.0
        p.sit_mode = False
        p.ground_pick_mode = False
        p.behavior_mode = None
        p.set_vel_cmd(0.0, 0.0, 0.0)  # selects the standing policy
        self.data.ctrl[:] = p.default_pose
        self.mouth_opening = 0.0
        # A gesture is about the episode it was made in; the head it would
        # restore belongs to a world that no longer exists.
        self._emote = None
        self.mouth_tick()
        mujoco.mj_forward(self.model, self.data)
        self.sim_time = 0.0
        self._det_step = 0
        self._ball_seen = dict(NOT_SEEN)
        self._ball_seen_t = 0.0
        self._ball_track.clear()  # sim clock rewound; stale points are future-dated
        self._goal_seen = dict(GOAL_NOT_SEEN)
        self._goal_seen_t = 0.0
        self._goal_fix = None       # the goal has not moved, but the clock
        self._goal_azimuth_w = None  # rewound: start the episode's memory clean
        if self.referee is not None:
            self.referee.reset()  # new episode, new scoreboard
        if self.machine is not None:
            # The machine rides into the new episode, but not its old head:
            # re-enter the current node so per-node memory and the entry clock
            # start clean. Otherwise the aiming approach's cached world-frame
            # detour targets outlive the world they were measured in (live-
            # observed: post-reset ghost detours herding the ball into the
            # same far corner), and elapsed_s goes negative against the
            # rewound sim clock, freezing every timed guard. A deliberate
            # non-wake: the reset was mind- or human-initiated, so re-entering
            # a wake node here does not latch a pack.
            self.machine.enter(self.machine.current, self.sim_time)
        self._detect_ball()  # so state right after a reset is not stale
        return self.get_state()

    def _handle_camera(self, req: dict, live: bool = False) -> dict:
        view = req.get("view", "follow")
        if view not in CAMERA_VIEWS:
            return {"ok": False, "error": f"unknown view {view!r} (choose from {CAMERA_VIEWS})"}
        try:
            if self._renderer is None:
                w = min(640, int(self.model.vis.global_.offwidth))
                h = min(480, int(self.model.vis.global_.offheight))
                self._renderer = mujoco.Renderer(self.model, height=h, width=w)
            if view == "head":
                if self._head_cam_id < 0:
                    return {"ok": False, "error": "model has no head_camera"}
                cam = self._head_cam_id
            else:
                cam = mujoco.MjvCamera()
                trunk = self.data.qpos[self.qpos_adr:self.qpos_adr + 3]
                quat = self.data.qpos[self.qpos_adr + 3:self.qpos_adr + 7]
                yaw_deg = math.degrees(quat_to_rpy(quat)[2])
                cam.lookat[:] = [trunk[0], trunk[1], max(0.08, float(trunk[2]))]
                cam.distance = float(req.get("distance", 0.7))
                if view == "follow":
                    cam.azimuth, cam.elevation = yaw_deg, -20
                elif view == "front":
                    cam.azimuth, cam.elevation = yaw_deg + 180, -15
                elif view == "side":
                    cam.azimuth, cam.elevation = yaw_deg + 90, -10
                elif view == "top":
                    cam.azimuth, cam.elevation = yaw_deg, -89
            self._renderer.update_scene(self.data, camera=cam)
            pixels = self._renderer.render()
            from PIL import Image
            if live:
                # Ambient page refresh: one file per view, overwritten in place.
                path = os.path.join(self.frames_dir, f"live_{view}.png")
            else:
                self._frame_count += 1
                path = os.path.join(self.frames_dir, f"frame_{self._frame_count:05d}_{view}.png")
            Image.fromarray(pixels).save(path)
            out = self.get_state()
            out["frame"] = path
            out["view"] = view
            return out
        except Exception as e:
            return {"ok": False, "error": f"render failed: {e} "
                    "(offscreen rendering may be unavailable under mjpython; use headless mode)"}

    # ---------- socket plumbing ----------

    def submit(self, req: dict, timeout: float = 10.0) -> dict:
        respq: "queue.Queue[dict]" = queue.Queue()
        self._requests.put((req, respq))
        return respq.get(timeout=timeout)

    def drain_requests(self):
        while True:
            try:
                req, respq = self._requests.get_nowait()
            except queue.Empty:
                return
            client = req.pop("client", "cli")
            try:
                resp = self.handle(req)
            except Exception as e:
                resp = {"ok": False, "error": repr(e)}
            respq.put(resp)
            self._log_event(client, req, resp)

    def _log_event(self, client: str, req: dict, resp: dict):
        cmd = req.get("cmd", "?")
        if cmd in ("camera_web", "ping", "mouth"):  # ambient traffic, not agent intent
            return
        if not resp.get("ok"):
            note = resp.get("error", "error")
        elif cmd == "trick":
            note = "started" if resp.get("started") else "refused"
        elif cmd in ("set_velocity", "reset", "state", "look", "push"):
            note = f"{resp.get('active_policy', '')}{'' if resp.get('upright', True) else ' DOWN'}"
        else:
            note = ""
        self._event_id += 1
        self.events.append({
            "id": self._event_id, "t": time.time(), "client": client, "cmd": cmd,
            "args": {k: v for k, v in req.items() if k != "cmd"},
            "ok": bool(resp.get("ok")), "note": note.strip(),
        })


def serve_socket(sim: DuckSim, sock_path: str, stop: threading.Event):
    if os.path.exists(sock_path):
        os.unlink(sock_path)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(sock_path)
    server.listen(8)
    server.settimeout(0.5)
    print(f"Control socket: {sock_path}")

    def dispatch(req: dict) -> dict:
        # `machine wait` (and arm/force with a block_s) long-polls the wake
        # latch on THIS connection thread — it must never ride the request
        # queue, which the sim thread drains synchronously between ticks.
        if req.get("cmd") == "machine":
            block_s = req.pop("block_s", None)
            if req.get("action") == "wait":
                return sim.machine_wait(55.0 if block_s is None else block_s)
            resp = sim.submit(req)
            if (block_s is not None and resp.get("ok")
                    and req.get("action") in ("arm", "force")):
                wake = sim.machine_wait(block_s)
                resp = {**resp,
                        **{k: v for k, v in wake.items() if k != "ok"},
                        "ok": bool(wake.get("ok"))}
            return resp
        return sim.submit(req)

    def client_thread(conn):
        with conn, conn.makefile("rwb") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    req = json.loads(line)
                    resp = dispatch(req)
                except Exception as e:
                    resp = {"ok": False, "error": repr(e)}
                f.write((json.dumps(resp) + "\n").encode())
                f.flush()

    def accept_loop():
        while not stop.is_set():
            try:
                conn, _ = server.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            threading.Thread(target=client_thread, args=(conn,), daemon=True).start()

    threading.Thread(target=accept_loop, daemon=True).start()
    return server


def run_loop(sim: DuckSim, viewer=None, realtime: bool = True, stop: threading.Event = None):
    control_dt = DECIMATION * TIMESTEP
    prev = time.perf_counter()
    deadline = prev + control_dt
    print(f"Sim loop running at {CONTROL_HZ} Hz "
          f"({'realtime' if realtime else 'fast'}, {'viewer' if viewer else 'headless'})")
    while not (stop and stop.is_set()):
        t0 = time.perf_counter()
        if viewer is not None and not viewer.is_running():
            break
        sim.drain_requests()
        dt = (t0 - prev) if realtime else control_dt
        prev = t0
        sim.policy.update_ground_pick_phase(dt)
        sim.policy.update_behavior(dt)
        action = sim.policy.infer()
        sim.policy.apply_action(action)
        sim.mouth_tick()
        for _ in range(DECIMATION):
            mujoco.mj_step(sim.model, sim.data)
        sim.sim_time += control_dt
        sim.sense()
        sim.referee_tick()
        sim.machine_tick()
        sim.emote_tick()  # after the machine: a gesture outranks the behavior
                          # for the head, for as long as it lasts
        if viewer is not None:
            viewer.sync()
        if realtime:
            # Absolute deadlines, not per-tick relative sleeps: sleep(x) on
            # macOS routinely overshoots by a millisecond or two, and relative
            # pacing bakes every overshoot into the schedule (~0.85x realtime).
            now = time.perf_counter()
            if now - deadline > RESYNC_LAG_S:
                deadline = now  # stalled — resync rather than sprint to catch up
            else:
                time.sleep(max(0.0, deadline - now))
            deadline += control_dt


def main():
    parser = argparse.ArgumentParser(description="Socket-controlled Microduck simulator")
    parser.add_argument("--rl-repo", default=os.environ.get("MICRODUCK_RL_REPO", "../microduck_rl"),
                        help="Path to a microduck_rl clone (scenes + PolicyInference)")
    parser.add_argument("--policies", default=os.environ.get("MICRODUCK_POLICIES", "../microduck/policies"),
                        help="Directory of ONNX policies (microduck repo's policies/)")
    parser.add_argument("--socket", default=os.environ.get("DUCK_SIM_SOCKET",
                        os.path.join(tempfile.gettempdir(), "microduck-sim.sock")))
    parser.add_argument("--scene", choices=sorted(SCENES), default="ball")
    parser.add_argument("--frames-dir", default=os.path.join(tempfile.gettempdir(), "microduck-frames"))
    parser.add_argument("--viewer", action="store_true",
                        help="Open the MuJoCo viewer (on macOS run under mjpython)")
    parser.add_argument("--fast", action="store_true", help="Run faster than realtime (headless only)")
    parser.add_argument("--web", type=int, default=8400, metavar="PORT",
                        help="Port for the AX debug page (0 disables; default 8400)")
    parser.add_argument("--no-voice", action="store_true",
                        help="Never speak a machine's `say` lines aloud "
                             "(they stay on the control surface)")
    parser.add_argument("--voice-bank", default=os.environ.get("DUCK_VOICE_BANK"),
                        metavar="DIR",
                        help="Voice-bank wavs for the duck's chirps (see "
                             "`duck say --voice-bank`)")
    parser.add_argument("--emotes", metavar="DIR",
                        default=os.environ.get("DUCK_EMOTES", "emotes"),
                        help="Directory of emote TOML files (default: emotes/ "
                             "under the working directory). Edits are picked "
                             "up live, like machine source")
    args = parser.parse_args()

    sim = DuckSim(args.rl_repo, args.policies, args.scene, args.frames_dir)
    sim.voice_bank = args.voice_bank
    sim.emotes = EmoteLibrary(args.emotes)
    if not os.path.isdir(sim.emotes.dir):
        print(f"note: no emote directory at {sim.emotes.dir} — the duck has "
              f"no body language this session (--emotes DIR)")
    if not args.no_voice:
        from . import voice
        sim.voice = voice.SayPlayer.available(bank_dir=args.voice_bank)
    stop = threading.Event()
    server = serve_socket(sim, args.socket, stop)
    web_server = None
    if args.web:
        from .webui import start_web
        web_server = start_web(sim, args.web)
    try:
        if args.viewer:
            import mujoco.viewer
            with mujoco.viewer.launch_passive(sim.model, sim.data,
                                              show_left_ui=False, show_right_ui=False) as viewer:
                run_loop(sim, viewer=viewer, realtime=True, stop=stop)
        else:
            run_loop(sim, viewer=None, realtime=not args.fast, stop=stop)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        if web_server is not None:
            web_server.shutdown()
        server.close()
        if os.path.exists(args.socket):
            os.unlink(args.socket)
        print("Sim stopped.")


if __name__ == "__main__":
    main()
