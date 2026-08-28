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
import socket
import sys
import tempfile
import threading
import time
from collections import deque

import mujoco
import numpy as np

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
DET_W, DET_H = 320, 240
DET_EVERY = 10  # control steps between detections -> 5 Hz at 50 Hz control
DET_MIN_PX = 6


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


def quat_to_rpy(q):
    """Quaternion [w, x, y, z] -> roll, pitch, yaw in radians."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


NOT_SEEN = {"visible": False, "distance_m": None, "ground_distance_m": None,
            "bearing_deg": None, "elevation_deg": None,
            "est_forward_m": None, "est_left_m": None}


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
    """Is the ball FULLY across the line and inside the mouth?

    Fully across: the trailing edge of the ball has cleared the line, i.e. the
    centre is a whole radius past it. Inside: between the posts, under the
    crossbar (a ball bouncing off the top of the frame is not a goal).
    """
    return (x > GOAL_LINE_X + radius_m
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
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.model.opt.timestep = TIMESTEP
        self.data = mujoco.MjData(self.model)
        self._apply_current_limit(DEFAULT_CURRENT_LIMIT)
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
        # Command-feed ring buffer for the AX debug page (webui.py).
        self.events = deque(maxlen=500)
        self._event_id = 0

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
        self._ball_seen = seen
        self._ball_seen_t = self.sim_time

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
        r = self.referee
        return {
            "ball_seen.visible": bs["visible"],
            "ball_seen.distance_m": bs["distance_m"],
            "ball_seen.ground_distance_m": bs.get("ground_distance_m"),
            "ball_seen.bearing_deg": bs["bearing_deg"],
            "ball_seen.elevation_deg": bs["elevation_deg"],
            "ball_seen.est_forward_m": bs.get("est_forward_m"),
            "ball_seen.est_left_m": bs.get("est_left_m"),
            "ball_seen.age_s": round(self.sim_time - self._ball_seen_t, 3),
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
        fired = self.machine.tick(self, self._machine_digest())
        if fired:
            self._event_id += 1
            self.events.append({
                "id": self._event_id, "t": time.time(), "client": "machine",
                "cmd": f"-> {fired['to']}",
                "args": {"from": fired["from"], "when": fired["when"]},
                "ok": True, "note": "",
            })
            print(f"machine: {fired['from']} -> {fired['to']}  [{fired['when']}]")

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
                    "note": "loaded, disarmed — arm to run"}
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
            return {"ok": True, **fresh.status(), "note": "hot-swapped"}
        if action == "arm":
            m.enter(m.initial, self.sim_time)
            m.armed = True
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
            vals = [req.get(k, 0.0) for k in ("neck_pitch", "head_pitch", "head_yaw", "head_roll")]
            p.head_offset[:] = np.clip(np.array(vals, dtype=np.float32), -p.head_max, p.head_max)
            p._update_command()
            return self.get_state()
        if cmd == "push":
            mag = float(np.clip(req.get("magnitude", 1.0), 0.0, 2.0))
            angle = math.radians(req["angle_deg"]) if "angle_deg" in req else random.uniform(0, 2 * math.pi)
            self.data.qvel[self.qvel_adr + 0] = mag * math.cos(angle)
            self.data.qvel[self.qvel_adr + 1] = mag * math.sin(angle)
            return {"ok": True, "pushed": {"magnitude": mag, "angle_deg": round(math.degrees(angle), 1)}}
        if cmd == "camera":
            return self._handle_camera(req)
        if cmd == "camera_web":
            return self._handle_camera(req, live=True)
        if cmd == "reset":
            return self._handle_reset()
        if cmd == "machine":
            return self._handle_machine(req)
        return {"ok": False, "error": f"unknown cmd {cmd!r}"}

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
        mujoco.mj_forward(self.model, self.data)
        self.sim_time = 0.0
        self._det_step = 0
        self._ball_seen = dict(NOT_SEEN)
        self._ball_seen_t = 0.0
        if self.referee is not None:
            self.referee.reset()  # new episode, new scoreboard
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
        if cmd in ("camera_web", "ping"):  # ambient page traffic, not agent intent
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

    def client_thread(conn):
        with conn, conn.makefile("rwb") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    req = json.loads(line)
                    resp = sim.submit(req)
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
        for _ in range(DECIMATION):
            mujoco.mj_step(sim.model, sim.data)
        sim.sim_time += control_dt
        sim.sense()
        sim.referee_tick()
        sim.machine_tick()
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
    args = parser.parse_args()

    sim = DuckSim(args.rl_repo, args.policies, args.scene, args.frames_dir)
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
