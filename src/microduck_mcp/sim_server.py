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
import io
import json
import math
import os
import queue
import random
import secrets
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

# Scenes that ship with THIS repo rather than microduck_rl. They are templates,
# not loadable MJCF: MuJoCo resolves <include> relative to the including file
# but `meshdir` relative to the MAIN file, so a scene living outside
# microduck_rl can reach the robot MJCF by no fixed relative path at all. The
# daemon is the only thing that knows where the clone is (--rl-repo), so it
# substitutes both absolute paths at load time — see local_scene_xml().
LOCAL_SCENES = {"desktop": "scene_desktop.xml"}
ROBOT_XML_REL = "src/mjlab_microduck/robot/microduck/robot_allcollisions.xml"

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

# ---------- the desktop pet ----------
# The duck on the Dock (notes/desktop-pet.md). The overlay app is a *viewer*:
# it owns no animation data, only a window that travels. Everything below is
# the sim half — the cutout frames, the pose the window is placed from, and
# the world the app configures from the screen it is drawn on.
#
# The mapping, in one line: sim world x -> screen x at `px_per_meter`, sim
# world z=0 -> the Dock's top edge. It holds exactly because the pet camera is
# ORTHOGRAPHIC (metres-per-pixel is fovy/height, at every depth, everywhere in
# frame), which is also why a platform box can stand in for a real window's
# ledge without drifting off it towards the edges of the screen.

# The geoms scene_desktop.xml pre-allocates for the pet to move around. Named,
# never indexed: the mouth mocap plate shifts geom ids at load (G20).
PET_WALL_GEOMS = ("pet_wall_left", "pet_wall_right")      # screen edges, ±x
PET_RAIL_GEOMS = ("pet_rail_near", "pet_rail_far")        # depth corridor, ±y
PET_PLATFORM_GEOMS = tuple(f"pet_platform_{i:02d}" for i in range(12))
# Where an unused platform box waits: past the rails, out of every contact
# pair, and (because scene_desktop.xml pins <statistic>) unable to inflate the
# model extent that the near/far clipping planes are derived from.
PET_PARK_POS = (0.0, 10.0, 0.5)
PET_PARK_SIZE = (0.05, 0.15, 0.02)
# Walls and rails: 0.60 m of box, two centimetres of it underground so the
# ground line has no seam. Must match scene_desktop.xml's wall/rail sizes.
PET_SOLID_HALF_H = 0.30
PET_SOLID_SINK = 0.02
PET_PLATFORM_DEPTH_M = 0.30   # default y extent of a ledge: wide enough to
                              # walk along, narrow enough to fall off

# Duck geometry, measured off the model (mesh vertices, STAND keyframe,
# excluding the floor and the unposed mouth plate): 0.2744 m floor to crown.
# A ~180 pt pet on a 1728 pt screen therefore wants 656 px/m, and that is the
# whole reason for every default below.
DUCK_HEIGHT_M = 0.2744
PET_PX_PER_METER = 656.0
PET_FRAME_PX = 300            # the overlay window's edge, in points
PET_SUPERSAMPLE = 2           # render 2x, box-downsample: real antialiasing on
                              # the alpha (the segmentation mask is hard-edged)
                              # and a Retina-native frame at backingScale 2.0
PET_SCREEN_WIDTH_M = 1728.0 / PET_PX_PER_METER  # one built-in display, points
PET_CORRIDOR_M = 0.35         # half-width of the depth corridor the rails make
PET_FLOOR_PAD_PX = 26         # output px between the sim floor line and the
                              # bottom of the frame — the duck's feet want a
                              # little room before the window ends
PET_CAM_DISTANCE_M = 1.0      # orthographic: sets depth only, never scale
PET_CAM_AZIMUTH_DEG = 90.0    # camera at -y looking +y, so +world x is screen
                              # right and the duck walks the way it drives
PET_CAM_ELEVATION_DEG = 0.0
PET_FRAME_MIN_PX, PET_FRAME_MAX_PX = 32, 512
PET_OFFSCREEN_MAX_PX = 1024   # scene_desktop.xml's <global offwidth/offheight>
# How many live `mujoco.Renderer`s to keep. A Renderer owns a GL context and an
# MjrContext: cheap to hold, ~50 ms to build. Keyed on render size, so two
# consumers asking for different sizes (the overlay at 512, a `duck pet frame
# --size-px 300` diagnostic beside it) each keep their own instead of tearing
# the other's down and rebuilding it on the 50 Hz sim thread every frame —
# measured, that alternation cost 93-117 ms a frame against 42-45 ms steady.
PET_RENDERER_CACHE = 3
# The pet world is one screen's worth of floor. `<statistic extent="0.6">` is
# pinned in scene_desktop.xml and every clipping plane is derived from it, so a
# ledge is metres, never kilometres and never inf — a non-finite half-extent
# lands in geom_rbound and swallows the duck whole (measured).
PET_WORLD_MAX_M = 20.0
# The chroma fallback key, when segmentation rendering is unavailable: the
# backdrop's exact magenta, which nothing on the duck comes near.
PET_CHROMA_RGB = (255, 0, 255)
PET_CHROMA_TOL = 60
PET_PUSH_MAX = 2.0            # same clamp the `push` intent already applies
                              # horizontally, now also on the vertical axis
                              # a drag gesture can reach (webui._pet_push)
# How long an MCP intent keeps the duck "inhabited" on the pet's status line.
# Long enough to survive Claude thinking between two tool calls, short enough
# that a finished residency stops claiming the wheel.
PET_INHABITED_S = 8.0
# Polls, not intents: these never count as somebody taking the wheel, and they
# never reach the AX event feed (a 20 fps frame stream would flush it in
# fifteen seconds — G18). `pet_sense` is here for the same arithmetic at a
# fifth of the rate: five cursor samples a second flush the 500-entry feed in
# under two minutes, and where the mouse pointer happens to be is not an act.
PET_AMBIENT_CMDS = ("pet_frame", "pet_state", "pet_sense")

# ---------- the hand on the other side of the glass ----------
# The overlay can see something the sim cannot: the human's pointer. It is
# reported in the same metres as everything else (the screen mapping is
# exact because the pet camera is orthographic), which is what lets a machine
# guard on it without the grammar learning a second coordinate system.
#
# How long a cursor sample stays believable. The app heartbeats at 1 Hz even
# when the pointer has not moved (pet_map.SENSE_HEARTBEAT_S), so anything
# older than two seconds means the app stopped talking — a locked screen, a
# quit overlay, a pointer that left this display — and the duck should stop
# acting as though somebody were standing next to it.
PET_CURSOR_STALE_S = 2.0
# How high above the walk line a pointer still counts as "down here with me".
# 0.35 m is about 230 pt: the Dock's own band plus a little. A cursor up in a
# text editor is not visiting the duck.
PET_CURSOR_FLOOR_M = 0.35
# Smoothing on the pointer's speed. A mouse moves in jerks (one sample can be
# a whole screen), and an unfiltered |dx|/dt is a number no guard could ever
# sensibly threshold. 0.4 keeps two thirds of a burst after three samples.
PET_CURSOR_SPEED_EMA = 0.4
# Where a `touch.petted` window ends. Three seconds is one stroke's worth of
# "that just happened", long enough for the machine to notice at 50 Hz and
# short enough that a pet does not keep a duck standing still.
PET_TOUCH_RECENT_S = 3.0
# ...and how often the duck answers one out loud. The nuzzle carries a coo,
# and a coo is the only sound in the pet's whole vocabulary a human can
# trigger on purpose — so a long stroke is one coo, not twenty.
PET_TOUCH_ACK_COOLDOWN_S = 2.5
# The gesture the duck understands. A roster of one, and a roster rather than
# a boolean because "scratched", "tickled" and "picked up by the beak" are
# all the same shape of request and this is where they would go.
PET_TOUCH_KINDS = ("pet",)
# What a pet is answered with: a lean into the hand, and the voice-bank tag
# that rides on it (emotes/nuzzle.toml).
PET_TOUCH_EMOTE = "nuzzle"

# ---------- the toy ----------
# scene_desktop.xml carries one free body that is neither duck nor invisible
# furniture: the same 70 mm floorball the pitch scenes use. It is the first
# thing in the pet's world the segmentation mask has to be able to tell APART
# from the duck rather than merely exclude — the frame's `bbox` is what the
# overlay hit-tests a click against, and a ball folded into it would be a duck
# you could grab from half a window away. Hence two masks, not one.
PET_BALL_GEOM = "ball_geom"
PET_BALL_JOINT = "ball_free"


def kick_policies_apply(scene_key: str, has_ball: bool) -> bool:
    """Should this daemon load the two kick policies?

    Two terms, and the second one used to be a name standing in for a fact.
    There has to be a ball to kick, asked of the COMPILED MODEL rather than of
    the scene's name — which is both more correct generally and the thing that
    lets the desktop pet boot a toy: `scene_desktop.xml` grew a ball, and the
    old `scene_key in LOCAL_SCENES` gate would have gone on refusing it the
    policy that kicks one, leaving `machines/pet.toml`'s `boot` a silent 4.4 s
    no-op that nothing would ever have complained about.

    And the `plain` scene has no ball however the question is asked, so it
    keeps its explicit "don't bother" — it is the scene people start for a
    walk test and the load time is the whole reason it exists.

    Its own function because a gate nothing can call is a gate nothing can
    test: a regression here is invisible both to a test suite that never runs
    `__init__` and to a casual look at a running pet, since a missing kick
    policy is SAFE (infer_policy.trigger_behavior prints and returns, the
    machine's pocket guard is never satisfied, the node's deadline exits).
    Safe is not the same as working.
    """
    return scene_key != "plain" and bool(has_ball)
# Breathing room between the toy and an invisible wall when the toy has to be
# put back inside one (`_pet_ball_into_play`). On top of the wall's own half
# thickness and the ball's radius, both of which are read off the model — this
# is only the gap that stops a nudged-back ball resting in permanent contact.
PET_BALL_WALL_PAD_M = 0.02
# Where a `push` may be aimed. The default is "duck" so every caller written
# before the toy existed (pet_mock, `duck pet`, every test, mcp_server's
# duck_push) keeps working unchanged.
PET_PUSH_TARGETS = ("duck", "ball")

# ---------- being picked up ----------
# The one gesture that is not a force. A poke and a flick are impulses the
# controller has to survive; a pick-up is a CONSTRAINT — scene_desktop.xml
# carries an invisible mocap "hand" and a weld between it and the trunk, and
# the whole feature is toggling that weld's `eq_active` and dragging the hand
# around. Nothing about the duck is animated while it hangs there: a standing
# policy with no ground under it flails, and that flail is the honest answer
# a real robot gives when you lift it off the floor.
PET_CARRY_EQ = "pet_carry"        # <equality><weld name=...> in the scene
PET_HAND_BODY = "pet_hand"        # the mocap body the weld hangs off
# Pet furniture that is neither wall, rail nor ledge: registered in
# `_pet_geoms` purely so its body joins `pet_bodies` and is excluded from the
# duck's segmentation mask. Group 4 already keeps it out of the picture, but
# the mask is built from body ids and the two must agree.
PET_PROP_GEOMS = ("pet_hand",)
# The deadman. An overlay that crashed, was force-quit or lost its connection
# mid-lift must not leave the duck hanging in the air for the rest of the
# session, so silence for this long is a release. The app re-states the hand
# at PET_CARRY_HZ (pet_map.CARRY_HZ, 20 Hz) even when the mouse is not moving,
# which is what makes silence mean something.
PET_CARRY_TIMEOUT_S = 1.5
# How far inside the invisible walls a carried duck may be taken. The walls
# stop a duck that walks into them; nothing stops a hand, so the hand is
# clamped instead — and a little inside, because the duck hangs BELOW the
# hand and swings.
PET_CARRY_WALL_PAD_M = 0.05
# The floor of the carry, and the ceiling. Below the first the hand is
# putting the duck down (which is what `end` is for); above the second it has
# left the screen it lives on. The app overrides `carry_max_z_m` from the
# actual display height (pet_map.ScreenMap.config_payload).
PET_CARRY_MIN_Z_M = 0.05
PET_CARRY_MAX_Z_M = 0.55
# How fast the hand chases the pointer, metres per second. A cursor can jump
# a whole screen between two samples and a welded body would go with it —
# through a wall, through the floor, at a speed the solver cannot answer for.
# 1.5 m/s is faster than anyone lifts a duck and slow enough to be physics.
PET_CARRY_HAND_SPEED_MPS = 1.5
# The release velocity is the hand's own, measured over the last fraction of
# a second of its track. Six samples at 20 Hz is 0.3 s of history, which is
# more than the window needs and enough that a jerk at the end does not have
# to be the whole answer.
PET_CARRY_TRACK = 6
PET_CARRY_VEL_WINDOW_S = 0.1
# Trunk z above which the pet frame's floor line starts to rise with the duck
# (see `_pet_camera`). A standing trunk is ~0.116 m and the worst shove
# stagger measured stays well under 0.20, so ordinary walking, stumbles and
# face-plants never move the camera — the Dock's edge stays pinned, which is
# the entire illusion. Only a duck that has genuinely left the floor lifts
# the picture with it.
PET_LIFT_TRIGGER_M = 0.20
# The verbs a carry understands. `start` mints a token, `move` restates where
# the hand is, `end` lets go — and every `move`/`end` has to carry the token
# it was given, so a stale gesture cannot release a grab that came after it.
PET_CARRY_ACTIONS = ("start", "move", "end")


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


def local_scenes_dir() -> str:
    """Where this repo's `scenes/` lives.

    The same ladder `--emotes` walks, two rungs longer: an explicit env
    override, then the copy inside the installed package (pyproject
    force-includes `scenes/` as `microduck_mcp/scenes`, so a wheel is not a
    daemon that cannot open `--scene desktop`), then the checkout the package
    is running out of (so a dev daemon started from anywhere still finds the
    editable one), then the working directory.
    """
    env = os.environ.get("DUCK_SCENES")
    if env:
        return os.path.abspath(os.path.expanduser(env))
    here = os.path.dirname(os.path.abspath(__file__))  # src/microduck_mcp
    packaged = os.path.join(here, "scenes")
    if os.path.isdir(packaged):
        return packaged
    repo = os.path.abspath(os.path.join(here, os.pardir, os.pardir, "scenes"))
    if os.path.isdir(repo):
        return repo
    return os.path.abspath("scenes")


def local_scene_xml(name: str, rl_repo: str) -> tuple[str, str]:
    """Read one of LOCAL_SCENES and bind it to this microduck_rl clone.

    Returns (xml_text, template_path). The template names the robot MJCF and
    its mesh directory as @ROBOT_XML@ / @MESHDIR@ because neither can be
    reached relatively from outside microduck_rl: MuJoCo resolves an <include>
    against the including file but `meshdir` against the MAIN file, so a scene
    over here would need two different relative roots and gets neither.
    """
    path = os.path.join(local_scenes_dir(), LOCAL_SCENES[name])
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"scene template {LOCAL_SCENES[name]!r} not found at {path} — set "
            "DUCK_SCENES to microduck-mcp's scenes/ directory")
    robot = os.path.join(os.path.abspath(os.path.expanduser(rl_repo)),
                         ROBOT_XML_REL)
    if not os.path.isfile(robot):
        raise FileNotFoundError(
            f"robot MJCF not found at {robot} — pass --rl-repo pointing at a "
            "clone of https://github.com/pollen-robotics/microduck_rl")
    with open(path) as f:
        text = f.read()
    text = text.replace("@ROBOT_XML@", robot)
    text = text.replace("@MESHDIR@", os.path.join(os.path.dirname(robot), "assets"))
    return text, path


def load_model_with_mouth(xml_path: str, xml_text: str | None = None):
    """Compile the scene with the mouth plate moved onto a mocap body.

    Returns (model, mouth_ok). Any failure — an MJCF without the head body or
    plate mesh, an older mujoco without MjSpec — falls back to the stock model
    with the mouth disabled, never a broken sim.

    `xml_text` compiles an in-memory scene (a bound LOCAL_SCENES template)
    instead of a file; `xml_path` is then only what gets printed and blamed.
    """
    try:
        spec = (mujoco.MjSpec.from_string(xml_text) if xml_text is not None
                else mujoco.MjSpec.from_file(xml_path))
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
        if xml_text is not None:
            return mujoco.MjModel.from_xml_string(xml_text), False
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

        # Three kinds of scene, one normalized key. `scene_key` is what the
        # rest of __init__ branches on, so a scene that arrives as a path or
        # as one of this repo's own templates still lands in the right
        # "has a ball / has a goal / has neither" branch.
        self.scene = scene
        xml_text = None
        if scene in LOCAL_SCENES:
            self.scene_key = scene
            xml_text, xml_path = local_scene_xml(scene, self.rl_repo)
        elif scene.endswith(".xml"):
            xml_path = os.path.abspath(os.path.expanduser(scene))
            self.scene_key = os.path.splitext(os.path.basename(xml_path))[0]
        elif scene in SCENES:
            self.scene_key = scene
            xml_path = os.path.join(self.rl_repo, SCENES[scene])
        else:
            raise ValueError(
                f"unknown scene {scene!r} — choose from "
                f"{', '.join(sorted(SCENES) + sorted(LOCAL_SCENES))}, or pass "
                "a path to an .xml")
        print(f"Loading MuJoCo model: {xml_path}")
        self.model, mouth_ok = load_model_with_mouth(xml_path, xml_text)
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
        has_ball = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT,
                                     PET_BALL_JOINT) >= 0
        if not kick_policies_apply(self.scene_key, has_ball):
            paths.pop("kick_left", None)
            paths.pop("kick_right", None)
        # Which file each role is running RIGHT NOW — kept true across live
        # swaps, so `policy list` never has to guess from symlinks on disk.
        self.policy_paths = {r: os.path.realpath(p) for r, p in paths.items()}

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
        if self.scene_key == "pitch" and self.policy.ball_qpos_adr is not None:
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
        # The desktop pet's own renderer, world and screen mapping. All of it
        # is inert on a scene without the pet geoms — `pet_*` commands then
        # refuse with the scene's name rather than half-working.
        self.pet = self._pet_default_config()
        # Every movable pet part is a mocap body wearing a same-named geom:
        # the geom id carries the size, the mocap id carries the pose.
        self._pet_geoms, self._pet_mocap = {}, {}
        for name in (PET_WALL_GEOMS + PET_RAIL_GEOMS + PET_PLATFORM_GEOMS
                     + PET_PROP_GEOMS):
            self._pet_geoms[name] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            self._pet_mocap[name] = (int(self.model.body_mocapid[bid])
                                     if bid >= 0 else -1)
        self.pet_scene = all(self._pet_geoms[n] >= 0 and self._pet_mocap[n] >= 0
                             for n in PET_WALL_GEOMS)
        self._pet_resolve_masks()
        # The pick-up: one weld, one invisible hand, one body to weld them to.
        # Resolved by NAME once, here, for the reason every other lookup in
        # this file is — the mouth plate is injected at load and shifts ids.
        # All three are -1 on a scene without them, which is what
        # `_pet_can_carry` reads.
        self._pet_carry_eq = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_EQUALITY, PET_CARRY_EQ)
        self._pet_hand_mocap = self._pet_mocap.get(PET_HAND_BODY, -1)
        self._pet_trunk_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        # render_px -> Renderer, most-recently-used last (see
        # PET_RENDERER_CACHE). `_pet_renderer` is whichever one the frame in
        # hand is being drawn with — `_pet_alpha` renders through it.
        self._pet_renderers = {}
        self._pet_renderer = None
        self._pet_renderer_px = 0
        self._pet_option = None
        # Did a caller pin `wall_margin_m`? Until one does, the margin tracks
        # the window size; once one has, a later config that does not mention
        # it must LEAVE IT ALONE. `duck pet config` with no flags is documented
        # as a read (client.py), and silently walking the walls back to half a
        # window would put them inside the arc pet.toml turns around in.
        self._pet_wall_pinned = False
        # Segmentation gives an exact duck mask; if this host cannot do a
        # second pass we fall back to keying the backdrop's magenta, and say
        # so in pet_state rather than silently shipping a worse cutout.
        self._pet_seg_ok = True
        self._pet_cache = None  # (key, png bytes, w, h) for one sim tick
        # The human, as far as the duck can tell. Both are written on the sim
        # thread (by the pet_sense/pet_touch handlers, which run inside
        # drain_requests) and read on the sim thread (_machine_digest,
        # pet_state), so there is no lock here and none is wanted.
        self._pet_cursor = None      # the last pointer sample, or None
        self._pet_touch = {"t": None, "count": 0, "ack_t": None}
        # ...and the hand that is actually holding it: None, or the one live
        # grab. Same thread discipline as the two above — written by
        # `_handle_pet_carry` inside drain_requests, advanced by
        # `pet_carry_tick` in the same loop, read by `_machine_digest` and
        # `pet_state`. All sim thread, so no lock, and none is wanted.
        self._pet_carry = None
        if self.pet_scene:
            self._pet_apply_world_geometry()
        # Who has the wheel. Set by drain_requests when an MCP client sends an
        # intent (not a poll) — the pet's "Claude is driving" light, and the
        # only place the sim cares which client is talking.
        self._mcp_intent_t = 0.0
        self._mcp_intent_cmd = None

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
        # The desktop pet's own senses. They live here rather than in
        # _pose_digest (the executor's half) because unlike `base.*` they are
        # not derivable from the robot's own state — only the server has been
        # told where the human's hand is. Always present, null/False/999.0 on
        # a scene with no overlay attached, so the key set stays stable and a
        # machine that guards on a cursor loads on any daemon new enough.
        cur = self._pet_cursor_state()
        touch = self._pet_touch_state()
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
            "cursor.present": cur["present"],
            "cursor.x_m": cur["x_m"],
            "cursor.z_m": cur["z_m"],
            "cursor.dx_m": cur["dx_m"],
            "cursor.dist_m": cur["dist_m"],
            "cursor.age_s": cur["age_s"],
            "cursor.near_floor": cur["near_floor"],
            "cursor.speed_mps": cur["speed_mps"],
            "touch.petted": touch["petted"],
            "touch.age_s": touch["age_s"],
            "touch.count": touch["count"],
            # The pick-up. `machines/pet.toml` carves `not carried` out of its
            # reflexes for one measured reason: a dangling duck reads `not
            # upright` within a second of being lifted, and without the
            # carve-out the machine storms `fallen`, latches a wake, droops
            # and says "I have gone over" every single time anybody picks it
            # up. False and 0.0 on a scene with no weld in it, which is what
            # lets the same file load on a daemon that cannot be picked up.
            "carried": self._pet_carry is not None,
            "carried_s": (round(self.sim_time - self._pet_carry["t0"], 3)
                          if self._pet_carry is not None else 0.0),
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
                self._say_line(fired["to"], say,
                               self.machine.nodes[fired["to"]].get("say_mood")
                               or "neutral")
            emote = self.machine.nodes[fired["to"]].get("emote")
            if emote:
                self._emote_node(fired["to"], emote)
            wake = self.machine.nodes[fired["to"]]["wake"]
            if wake is not None:
                self._latch_wake(fired["to"], wake,
                                 {"from": fired["from"], "when": fired["when"]},
                                 digest)

    def _say_line(self, node: str, text: str, mood: str = "neutral"):
        """A speaking node's line: onto the control surface, then into the air.

        The annotation goes through the same `say` verb `duck say` uses, so a
        line the machine decided to say and a line a person asked for look
        identical on the event feed — and the film's control-surface feed
        picks it up for free.

        The mood rides along on the event only when there IS one: a neutral
        line is the ordinary case, and an event feed that annotates the
        ordinary case stops being readable.

        Speaking is best-effort and belongs to the host: the robot has a mouth
        servo, not a speaker. Without a voice this session the line is still an
        event, which is the point of it being an annotation.
        """
        args = {"cmd": "say", "node": node, "text": text}
        if mood != "neutral":
            args["mood"] = mood
        resp = self.handle({"cmd": "say", "text": text})
        self._log_event("machine", args, resp)
        if self.voice is not None:
            try:
                self.voice.speak(text, self, mood)
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
            # What is being shoved. Default "duck" so every caller written
            # before the toy existed is unchanged, and a scene with no ball
            # says so rather than silently shoving the duck instead — a poke
            # that moved the wrong thing is worse than one that refused.
            target = str(req.get("target", "duck"))
            if target not in PET_PUSH_TARGETS:
                return {"ok": False, "error":
                        f"unknown push target {target!r} — this scene "
                        f"understands {', '.join(PET_PUSH_TARGETS)}"}
            if target == "ball":
                if getattr(self.policy, "ball_qvel_adr", None) is None:
                    return {"ok": False,
                            "error": "no ball in this scene to shove"}
                adr = self.policy.ball_qvel_adr
            else:
                adr = self.qvel_adr
            mag = float(np.clip(req.get("magnitude", 1.0), 0.0, 2.0))
            angle = math.radians(req["angle_deg"]) if "angle_deg" in req else random.uniform(0, 2 * math.pi)
            self.data.qvel[adr + 0] = mag * math.cos(angle)
            self.data.qvel[adr + 1] = mag * math.sin(angle)
            pushed = {"magnitude": mag, "angle_deg": round(math.degrees(angle), 1),
                      "target": target}
            # Optional vertical component. The desktop pet's drag gesture
            # happens in a side view, where "up the screen" is world +z and a
            # shove that cannot lift the duck is only half a shove. Absent, the
            # vertical velocity is left exactly as the physics had it — the
            # horizontal push has behaved this way since it was written.
            if req.get("vz") is not None:
                vz = float(np.clip(req["vz"], -PET_PUSH_MAX, PET_PUSH_MAX))
                self.data.qvel[adr + 2] = vz
                pushed["vz"] = round(vz, 3)
            return {"ok": True, "target": target, "pushed": pushed}
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
        if cmd == "policy":
            return self._handle_policy(req)
        if cmd == "pet_frame":
            return self._handle_pet_frame(req)
        if cmd == "pet_state":
            return self._handle_pet_state(req)
        if cmd == "pet_config":
            return self._handle_pet_config(req)
        if cmd == "pet_world":
            return self._handle_pet_world(req)
        if cmd == "pet_sense":
            return self._handle_pet_sense(req)
        if cmd == "pet_touch":
            return self._handle_pet_touch(req)
        if cmd == "pet_carry":
            return self._handle_pet_carry(req)
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

    # Where PolicyInference keeps each role's ONNX session. The sitstand role
    # lands in sit_session (with is_sitstand=True) — infer_policy's legacy
    # naming, papered over here so the wire protocol speaks roles only.
    POLICY_SESSION_ATTRS = {
        "walking": "walking_session",
        "standing": "standing_session",
        "sitstand": "sit_session",
        "ground_pick": "ground_pick_session",
    }
    BEHAVIOR_ROLES = ("kick_left", "kick_right", "roulade")

    def _policy_session(self, role):
        p = self.policy
        if role in self.POLICY_SESSION_ATTRS:
            return getattr(p, self.POLICY_SESSION_ATTRS[role])
        if role in self.BEHAVIOR_ROLES:
            return p.behavior_sessions.get(role)
        return None

    def _handle_policy(self, req: dict) -> dict:
        """Inspect or hot-swap the ONNX brains, on a live sim.

        The train→export→swap loop ends here: a freshly exported checkpoint
        goes into a running body without a restart, and a bad one is undone
        with one more swap (the reply carries the path it displaced). The
        contract that makes that safe: the new session is BUILT AND VALIDATED
        off to the side first — if the file is missing, malformed, or speaks a
        different obs width than the incumbent, the refusal arrives before
        anything is touched and the old brain never stops flying. Runs on the
        sim thread like every handler, so the rebind lands between physics
        steps, never mid-inference.
        """
        p = self.policy
        action = req.get("action", "list")
        roles = list(self.POLICY_SESSION_ATTRS) + list(self.BEHAVIOR_ROLES)
        if action == "list":
            slots = []
            for role in roles:
                sess = self._policy_session(role)
                slots.append({
                    "role": role,
                    "loaded": sess is not None,
                    "file": self.policy_paths.get(role),
                    "obs_dim": int(sess.get_inputs()[0].shape[-1]) if sess else None,
                })
            return {"ok": True, "active_policy": p.current_policy, "slots": slots}
        if action != "swap":
            return {"ok": False, "error": f"unknown policy action {action!r} (list, swap)"}

        role = req.get("role", "")
        if role not in roles:
            return {"ok": False, "error": f"unknown role {role!r} ({', '.join(roles)})"}
        path = os.path.realpath(os.path.expanduser(str(req.get("path", ""))))
        if not os.path.isfile(path):
            return {"ok": False, "error": f"no ONNX file at {path}"}
        import onnxruntime as ort
        try:
            sess = ort.InferenceSession(path)
        except Exception as e:
            return {"ok": False, "error": f"ONNX load failed, incumbent untouched: {e}"}
        obs_dim = int(sess.get_inputs()[0].shape[-1])
        incumbent = self._policy_session(role)
        if incumbent is not None:
            want = int(incumbent.get_inputs()[0].shape[-1])
            if obs_dim != want:
                return {"ok": False, "error":
                        f"obs contract mismatch: {role} runs {want}D, "
                        f"{os.path.basename(path)} wants {obs_dim}D — refused, "
                        "incumbent untouched"}
        if role in self.POLICY_SESSION_ATTRS:
            setattr(p, self.POLICY_SESSION_ATTRS[role], sess)
            if role == "sitstand":
                p.is_sitstand = True
        else:
            p.behavior_sessions[role] = sess
        previous = self.policy_paths.get(role)
        self.policy_paths[role] = path
        print(f"policy swap: {role} ← {path}" +
              (f" (was {previous})" if previous else ""))
        return {"ok": True, "role": role, "file": path, "obs_dim": obs_dim,
                "previous": previous,
                "active_policy": p.current_policy,
                "note": "live next control tick; swap back by passing `previous`"}

    def _handle_reset(self) -> dict:
        p = self.policy
        # Let go FIRST. `eq_active` is sim state and the rewind below does not
        # touch it, so a reset made mid-carry would rewind qpos while the weld
        # cheerfully dragged the duck straight back to wherever the hand was
        # left — a reset that looks like it failed. The release velocity it
        # writes is wiped by the qvel zeroing two lines down, which is the
        # right answer: a new episode starts at rest.
        self._pet_carry_release()
        self.data.qpos[:] = self._qpos0
        self.data.qvel[:] = 0.0
        # ...and `_qpos0` carries the scene's baked-in toy spawn, which is a
        # number about a display nobody has measured yet. The walls are still
        # wherever this screen put them (the pet's WORLD is deliberately not
        # reset — it belongs to the screen, not the episode), so a spawn that
        # lands inside one has to be trimmed here too or every reset re-ejects
        # the ball off the far side of the world.
        self._pet_ball_into_play()
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
        # The pet's one-tick frame cache is keyed on sim_time, and the clock
        # just rewound onto a tick it may already have a picture of. (The
        # pet's WORLD — walls, ledges — is deliberately not reset: it belongs
        # to the screen, not the episode.)
        self._pet_cache = None
        # The human, though, IS episode state, and its clock is the one that
        # just rewound: a cursor sample stamped at t=41 against a sim clock
        # back at 0 is future-dated, reads as age -41 s, and every "the hand
        # has gone" guard in machines/pet.toml stops being able to fire. Drop
        # it and let the overlay's next sample (≤1 s away, it heartbeats)
        # re-establish where the pointer is. `count` survives on purpose —
        # it is a session tally of how often this duck has been petted, not a
        # fact about the episode that just ended.
        self._pet_cursor = None
        self._pet_touch["t"] = None
        self._pet_touch["ack_t"] = None
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

    # ---------- the desktop pet (sim thread only) ----------
    # Four verbs, and between them the whole sim half of the overlay:
    #   pet_config  what screen the duck is living on -> where the walls go
    #   pet_world   what is on that screen -> where the platform ledges go
    #   pet_frame   a cutout of the duck, alpha and all, plus the pose
    #   pet_state   the pose alone, for when a frame would be wasteful
    # `pet_frame` deliberately carries the full pet state: the window has to
    # be moved to match the frame it is showing, and one submit costs a 20 ms
    # tick, so asking for the pose separately would halve the frame rate to
    # learn something the render already knew.
    # Two more (`pet_sense`, `pet_touch`) live further down, under "the
    # human": they are about the person rather than the picture.

    def _pet_resolve_masks(self):
        """Decide, once at load, which geoms are duck and which are toy.

        By NAME, not by index: the mouth plate is injected at load and shifts
        every geom id after it (G20). Everything hanging off the worldbody is
        scenery, everything on a pet mocap body is invisible furniture, and
        what is left used to be, by elimination, robot.

        That elimination stopped being safe the moment the scene grew a ball.
        A free body is not the worldbody and is not a mocap body, so the old
        one-mask rule would have quietly made the toy part of the duck — in
        the alpha, which is cosmetic, and in `bbox`, which is not: `bbox` is
        what the overlay hit-tests a click against, so a ball on the far side
        of the window would have been a duck you could grab from there. Hence
        two masks that partition the drawn world instead of one that assumes
        it. Both are ngeom-long boolean arrays, indexed straight by the
        segmentation pass's object ids.
        """
        pet_bodies = {int(self.model.geom_bodyid[self._pet_geoms[n]])
                      for n in self._pet_geoms if self._pet_geoms[n] >= 0}
        ball_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                                     PET_BALL_GEOM)
        ball_body = (int(self.model.geom_bodyid[ball_gid])
                     if ball_gid >= 0 else None)
        self._pet_ball_geom = np.array(
            [ball_body is not None
             and int(self.model.geom_bodyid[g]) == ball_body
             for g in range(self.model.ngeom)], dtype=bool)
        self._pet_duck_geom = np.array(
            [int(self.model.geom_bodyid[g]) != 0
             and int(self.model.geom_bodyid[g]) not in pet_bodies
             and not self._pet_ball_geom[g]
             for g in range(self.model.ngeom)], dtype=bool)

    def _pet_default_config(self) -> dict:
        return {
            "px_per_meter": PET_PX_PER_METER,
            "frame_px": PET_FRAME_PX,
            "supersample": PET_SUPERSAMPLE,
            "screen_width_m": PET_SCREEN_WIDTH_M,
            # Half a window: the duck stops with its whole window on screen
            # rather than with its nose against the bezel.
            "wall_margin_m": PET_FRAME_PX / (2 * PET_PX_PER_METER),
            "corridor_m": PET_CORRIDOR_M,
            "floor_pad_px": PET_FLOOR_PAD_PX,
            # How high a pointer can be and still be visiting (see
            # PET_CURSOR_FLOOR_M). Config rather than a constant because it is
            # a claim about a SCREEN — a taller Dock, a bigger duck or a
            # display at another scale all move it.
            "cursor_floor_m": PET_CURSOR_FLOOR_M,
            # How big the toy is, in the same metres as everything else, so
            # the overlay can draw a grab rectangle around a ball that has
            # rolled off the frame and out of `ball_bbox`. Reported rather
            # than negotiated — BALL_RADIUS_M is baked into the head camera's
            # range solve, so this is the scene telling the app, never the
            # other way round. It is still in the config bounds table (and so
            # writable) for one reason: a future scene with a bigger toy would
            # otherwise need a protocol change to say so. Writing it moves the
            # overlay's grab box and NOTHING else — not the geom, not the
            # detector — so it is a lie unless the scene changed with it.
            "ball_radius_m": BALL_RADIUS_M,
            # The pick-up's three, and all three are claims about a SCREEN
            # rather than about physics, which is why they are config and not
            # constants. The app knows how tall the display is and overrides
            # `carry_max_z_m` from it on the first `pet_config` — a duck
            # carried above the menu bar has left the world it lives in, and
            # only the app can say where that is.
            "carry_max_z_m": PET_CARRY_MAX_Z_M,
            "carry_hand_speed_mps": PET_CARRY_HAND_SPEED_MPS,
            "lift_trigger_m": PET_LIFT_TRIGGER_M,
            "camera_distance_m": PET_CAM_DISTANCE_M,
            "azimuth_deg": PET_CAM_AZIMUTH_DEG,
            "elevation_deg": PET_CAM_ELEVATION_DEG,
        }

    def _pet_wall_x(self) -> float:
        """Where the invisible screen edges stand, in metres from the origin."""
        return max(0.05, self.pet["screen_width_m"] / 2
                   - self.pet["wall_margin_m"])

    def _pet_set_geom(self, name: str, pos, size=None):
        """Move (and optionally resize) one piece of the pet's world, live.

        The pose goes through mjData.mocap_pos, NOT mjModel.geom_pos: a geom
        hanging off the worldbody keeps the compile-time bounding volume that
        the broadphase culls against, so moving one moves its picture and
        silently retires its physics (measured — see scene_desktop.xml).
        Every movable pet part is therefore its own mocap body.

        The size does go through the model, and takes geom_rbound and
        geom_aabb with it: MuJoCo derives both from the size at compile time
        and never revisits them, so a box you grew would stop generating
        contacts past its old bounding sphere.
        """
        gid = self._pet_geoms.get(name, -1)
        mid = self._pet_mocap.get(name, -1)
        if gid < 0 or mid < 0:
            return False
        self.data.mocap_pos[mid] = pos
        if size is not None:
            self.model.geom_size[gid] = size
            self.model.geom_rbound[gid] = float(np.linalg.norm(size))
            self.model.geom_aabb[gid] = [0.0, 0.0, 0.0, *size]
        return True

    def _pet_apply_world_geometry(self):
        """Put the walls at the screen edges and the rails around the walkway.

        Sunk PET_SOLID_SINK below the floor so there is no seam at the
        ground line for a toe to catch (the walls are 0.60 m of box; two
        centimetres of it can live underground).

        The toy comes with them. Where the walls stand is a fact about the
        SCREEN and it changes under the scene's feet; where the ball spawned
        is a number baked into the scene. `_pet_ball_into_play` is what keeps
        the second inside the first.
        """
        half = self._pet_wall_x()
        z = PET_SOLID_HALF_H - PET_SOLID_SINK
        for name, sign in zip(PET_WALL_GEOMS, (-1.0, 1.0)):
            self._pet_set_geom(name, [sign * half, 0.0, z])
        for name, sign in zip(PET_RAIL_GEOMS, (-1.0, 1.0)):
            self._pet_set_geom(name, [0.0, sign * self.pet["corridor_m"], z])
        self._pet_ball_into_play()

    def _pet_ball_into_play(self) -> bool:
        """Put the toy back between the walls, wherever the walls have moved.

        The spawn is baked into `scenes/scene_desktop.xml` — x = 0.75, and the
        same number again in the STAND keyframe — but the walls are placed at
        RUNTIME from the app's screen (`_pet_wall_x`: half the usable band,
        less one duck-depth). The two only agree on a display wide enough for
        2 * (0.75 + 0.1227) = 1.745 m of sim floor. They do not agree on, say,
        a 1512 pt MacBook Pro 14" asked for a 240 pt duck: 875 px/m puts the
        band at 1.729 m and the walls at ±0.7417, and the right wall's box
        (0.02 half-thick) then spans 0.7217..0.7617 — with the ball's centre
        at 0.75, INSIDE it. Measured: the solver ejects it at 1.06 m/s and it
        settles at x = 3.22 m, out of play for the rest of the session. It
        cannot even be fetched back: the head camera does not see the wall
        (group 4 is culled from the pet's MjvOption) so `ball_seen` keeps
        reporting a toy `chase_ball` walks into a wall trying to reach, and
        `reset` restores `_qpos0`, which is the scene's number again.

        So the spawn is a wish and this is where it is trimmed — on every
        config that moves the walls, and after every reset. Velocity goes with
        it: a ball that had to be teleported was not rolling anywhere.

        Returns True if the toy actually had to be moved.
        """
        adr = getattr(self.policy, "ball_qpos_adr", None)
        if not self.pet_scene or adr is None:
            return False
        gid = self._pet_geoms.get(PET_WALL_GEOMS[1], -1)
        # Off the model rather than from a constant: the wall's half thickness
        # and the ball's radius are both the scene's to state, and a scene
        # with a fatter wall or a bigger toy must not need this edited too.
        wall_half = float(self.model.geom_size[gid][0]) if gid >= 0 else 0.0
        bgid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                                 PET_BALL_GEOM)
        radius = (float(self.model.geom_size[bgid][0]) if bgid >= 0
                  else BALL_RADIUS_M)
        limit = max(0.0, self._pet_wall_x() - wall_half - radius
                    - PET_BALL_WALL_PAD_M)
        x = float(self.data.qpos[adr])
        if abs(x) <= limit:
            return False
        self.data.qpos[adr] = math.copysign(limit, x)
        vadr = getattr(self.policy, "ball_qvel_adr", None)
        if vadr is not None:
            self.data.qvel[vadr:vadr + 6] = 0.0
        return True

    def _pet_not_a_pet_scene(self) -> dict:
        return {"ok": False, "error":
                f"scene {self.scene!r} has no pet geoms — start duck-sim with "
                "--scene desktop (scenes/scene_desktop.xml)"}

    def _pet_config_state(self) -> dict:
        p = self.pet
        half = self._pet_wall_x()
        frame_px = p["frame_px"]
        return {
            **p,
            "render_px": frame_px * p["supersample"],
            "view_height_m": frame_px / p["px_per_meter"],
            "walls_m": [-half, half],
            "duck_height_m": DUCK_HEIGHT_M,
            "duck_height_px": DUCK_HEIGHT_M * p["px_per_meter"],
            "platform_capacity": len(PET_PLATFORM_GEOMS),
            "segmentation": self._pet_seg_ok,
        }

    def _handle_pet_config(self, req: dict) -> dict:
        """Tell the sim what screen it is being drawn on.

        The app measures the screen (NSScreen.visibleFrame) and picks how big
        the duck should look; everything else — where the walls stand, how
        many metres of floor the screen is worth, what the camera's view
        height has to be — falls out of that. Sending it again is how a Dock
        resize or a display change is handled: no restart, no reset, the duck
        keeps walking while the world it is walking in changes shape.

        A call that names no key at all is a READ: it answers with the config
        and touches nothing. `duck pet config` with no flags is exactly that
        call (client.py sends only the keys it was given), and a "read" that
        moved the walls and ran an mj_forward would be a trap.
        """
        if not self.pet_scene:
            return self._pet_not_a_pet_scene()
        wanted = dict(self.pet)
        bounds = {
            "px_per_meter": (50.0, 8000.0),
            "frame_px": (PET_FRAME_MIN_PX, PET_FRAME_MAX_PX),
            "supersample": (1, 3),
            "screen_width_m": (0.2, 40.0),
            "wall_margin_m": (0.0, 5.0),
            "corridor_m": (0.05, 5.0),
            "floor_pad_px": (0, PET_FRAME_MAX_PX // 2),
            "cursor_floor_m": (0.05, 3.0),
            "ball_radius_m": (0.005, 0.5),
            "carry_max_z_m": (0.10, 3.0),
            "carry_hand_speed_mps": (0.1, 10.0),
            "lift_trigger_m": (0.12, 2.0),
            "camera_distance_m": (0.2, 20.0),
            "azimuth_deg": (-360.0, 360.0),
            "elevation_deg": (-89.0, 89.0),
        }
        if not (set(bounds) | {"screen_width_px"}) & set(req):
            return {"ok": True, "config": self._pet_config_state(),
                    "note": f"walls at ±{self._pet_wall_x():.3f} m (read only)"}
        for key, (lo, hi) in bounds.items():
            if key not in req:
                continue
            try:
                val = float(req[key])
            except (TypeError, ValueError):
                return {"ok": False, "error": f"{key} must be a number, "
                        f"got {req[key]!r}"}
            if not lo <= val <= hi:
                return {"ok": False, "error":
                        f"{key}={val} out of range [{lo}, {hi}]"}
            wanted[key] = (int(val) if key in ("frame_px", "supersample",
                                               "floor_pad_px") else val)
        # A screen width in points is what the app actually has; converting it
        # here keeps the one division in one place.
        if "screen_width_px" in req:
            try:
                px = float(req["screen_width_px"])
            except (TypeError, ValueError):
                return {"ok": False, "error": "screen_width_px must be a number"}
            if not 100.0 <= px <= 40000.0:
                return {"ok": False, "error": f"screen_width_px={px} out of range"}
            wanted["screen_width_m"] = px / wanted["px_per_meter"]
        if "wall_margin_m" in req:
            # Pinned. It stays pinned: the overlay sends one duck-depth on the
            # first config and then only ever re-sends the screen, and a later
            # partial config must not walk the walls back inside the arc the
            # machine turns around in (notes/desktop-pet.md's last bullet).
            self._pet_wall_pinned = True
        elif not self._pet_wall_pinned:
            # Nobody has an opinion yet, so track the window size.
            wanted["wall_margin_m"] = wanted["frame_px"] / (2 * wanted["px_per_meter"])
        if wanted["frame_px"] * wanted["supersample"] > PET_OFFSCREEN_MAX_PX:
            return {"ok": False, "error":
                    f"frame_px*supersample = "
                    f"{wanted['frame_px'] * wanted['supersample']} exceeds the "
                    f"{PET_OFFSCREEN_MAX_PX} px offscreen framebuffer"}
        # No renderer invalidation here: they are cached by render size
        # (PET_RENDERER_CACHE), so a resize picks up the right one — or builds
        # it — and the old one stays warm for whoever is still asking for it.
        self.pet = wanted
        self._pet_apply_world_geometry()
        self._pet_cache = None
        mujoco.mj_forward(self.model, self.data)  # walls move now, not next tick
        return {"ok": True, "config": self._pet_config_state(),
                "note": f"walls at ±{self._pet_wall_x():.3f} m"}

    def _handle_pet_world(self, req: dict) -> dict:
        """Reposition the pre-allocated platform boxes from screen rectangles.

        This is the window-ledge lane (notes/desktop-pet.md, "parked for v2"):
        the app mirrors real window frames in, and the duck stands on, walks
        along and falls off them with real contacts. Rectangles are in the
        SAME screen-space metres as everything else — x rightwards from the
        origin, y upwards from the Dock's top edge — with (x, y) the bottom
        left corner, because that is the shape a window rect arrives in.

        The list is the whole world, not a patch: every box the call does not
        name goes back to the parking lot. That way a window closing on the
        real desktop cannot leave a ledge hanging in the sim.
        """
        if not self.pet_scene:
            return self._pet_not_a_pet_scene()
        rects = req.get("rects", [])
        if not isinstance(rects, list):
            return {"ok": False, "error": "rects must be a list of "
                    "{x, y, w, h} rectangles in metres"}
        if len(rects) > len(PET_PLATFORM_GEOMS):
            return {"ok": False, "error":
                    f"{len(rects)} rectangles but only "
                    f"{len(PET_PLATFORM_GEOMS)} platform boxes are "
                    "pre-allocated (scene_desktop.xml) — send the nearest ones"}
        placed = []
        for i, r in enumerate(rects):
            if not isinstance(r, dict):
                return {"ok": False, "error": f"rect {i} is not an object"}
            try:
                x, y = float(r["x"]), float(r["y"])
                w, h = float(r["w"]), float(r["h"])
                depth = float(r.get("depth_m", PET_PLATFORM_DEPTH_M))
            except (KeyError, TypeError, ValueError):
                return {"ok": False, "error":
                        f"rect {i} needs numeric x, y, w, h (metres)"}
            # Range check every number, the way _handle_pet_config does, and
            # for a harder reason: this path writes model.geom_size and with
            # it geom_rbound and geom_aabb. `json.loads` accepts `Infinity`
            # and `1e400`, and inf > 0 is True — an infinite half-extent is a
            # box that contains the whole world, and the duck sinks into it
            # (measured: base z 0.116 -> 0.051 over 500 steps) with no way
            # back short of another POST, because `reset` deliberately leaves
            # the pet world alone.
            if not all(math.isfinite(v) for v in (x, y, w, h, depth)):
                return {"ok": False, "error":
                        f"rect {i} has a non-finite value: x={x} y={y} w={w} "
                        f"h={h} depth={depth}"}
            if not (w > 0 and h > 0 and depth > 0):
                return {"ok": False, "error":
                        f"rect {i} has a non-positive size: w={w} h={h} "
                        f"depth={depth}"}
            if max(abs(x), abs(y), w, h, depth) > PET_WORLD_MAX_M:
                return {"ok": False, "error":
                        f"rect {i} is outside the pet world: every value must "
                        f"be within ±{PET_WORLD_MAX_M} m (these are metres of "
                        "screen floor, not pixels)"}
            size = [w / 2, depth / 2, h / 2]
            pos = [x + w / 2, 0.0, y + h / 2]
            self._pet_set_geom(PET_PLATFORM_GEOMS[i], pos, size)
            placed.append({"geom": PET_PLATFORM_GEOMS[i], "center_m": pos,
                           "half_size_m": size})
        for name in PET_PLATFORM_GEOMS[len(rects):]:
            self._pet_set_geom(name, list(PET_PARK_POS), list(PET_PARK_SIZE))
        self._pet_cache = None
        mujoco.mj_forward(self.model, self.data)
        return {"ok": True, "placed": placed,
                "parked": len(PET_PLATFORM_GEOMS) - len(placed),
                "capacity": len(PET_PLATFORM_GEOMS)}

    def _pet_lift_m(self) -> float:
        """How far the frame's floor line has risen off world z=0, in metres.

        Zero almost always, and that is the point. The pet frame is 0.39 m of
        world with a 0.274 m duck in it — 9.6 cm of headroom — so a duck being
        carried leaves its own picture almost immediately, and the window has
        to travel up the screen with it. But a camera that followed every
        wobble would slide the Dock's edge up and down the frame and turn a
        stumble into a camera move, which is the illusion this whole overlay
        is built on. So the follow has a floor under it: nothing happens until
        the trunk passes `lift_trigger_m` (0.20 m against a standing 0.116 and
        a worst measured shove-stagger well under it), and past that the frame
        rises 1:1 with the duck.

        Reported to the app as `pet_state.screen.frame_floor_z_m`, which is
        what `pet_map.window_origin` hangs the window off. Emergent and worth
        knowing: a flick UPWARDS now also lifts the frame, so the shove
        gesture stops throwing the duck out of its own picture.
        """
        z = float(self.data.qpos[self.qpos_adr + 2])
        return max(0.0, z - self.pet["lift_trigger_m"])

    def _pet_frame_z_span(self) -> tuple:
        """World z of the bottom and top edges of the pet frame, in metres.

        Falls straight out of `_pet_camera`'s framing: the floor line sits
        `floor_pad_px` output pixels above the bottom edge, so the bottom edge
        is that far BELOW the floor (negative z — the inside of the Dock) and
        the top edge is the rest of the frame above it. Used to answer
        `ball.in_frame`, which is the app's licence to hit-test a click
        against a toy it can actually see.

        Both edges carry `_pet_lift_m`, because the frame's "floor" is not
        world z=0 once the duck is off the ground — it is wherever the camera
        put it. Without that a ball at the duck's feet would read as in frame
        while the duck was dangling half a metre above it.
        """
        p = self.pet
        lift = self._pet_lift_m()
        return (lift - p["floor_pad_px"] / p["px_per_meter"],
                lift + (p["frame_px"] - p["floor_pad_px"]) / p["px_per_meter"])

    def _pet_ball_state(self) -> dict:
        """Where the toy is, or None on a scene that has none.

        Ground truth, and deliberately so: this is not a sense, it is the
        renderer's half of the contract. The overlay has to know where the
        ball's pixels are to decide whether a click was aimed at it, and the
        camera is orthographic, so world x and z ARE screen coordinates once
        multiplied out. The DUCK's knowledge of the ball is a different thing
        entirely and stays where it was — `ball_seen.*`, out of the head
        camera, with nothing in it the real robot could not compute.

        `in_frame` is the useful half: the pet camera renders 0.39 m of world
        around the duck, so a ball further away than that is neither drawn nor
        clickable — which is the design, not a defect. Past the edge of the
        window the duck has to go and fetch it, which is exactly what
        `machines/pet.toml`'s chase does.
        """
        p = self.policy
        if getattr(p, "ball_qpos_adr", None) is None:
            return None
        adr, vadr = p.ball_qpos_adr, p.ball_qvel_adr
        b = self.data.qpos[adr:adr + 3]
        v = self.data.qvel[vadr:vadr + 3]
        r = self.pet["ball_radius_m"]
        dx = float(b[0]) - float(self.data.qpos[self.qpos_adr])
        half_x = 0.5 * self.pet["frame_px"] / self.pet["px_per_meter"]
        z_lo, z_hi = self._pet_frame_z_span()
        return {
            "present": True,
            "x_m": float(b[0]), "y_m": float(b[1]), "z_m": float(b[2]),
            # Signed and horizontal, exactly like `cursor.dx_m`: the frame is
            # centred on the duck, so this is also where in the window to draw
            # the grab box.
            "dx_m": dx,
            "radius_m": r,
            # The disc overlapping the frame, not just its centre — half a
            # ball sticking into the picture is still a ball you can click.
            "in_frame": bool(abs(dx) <= half_x + r
                             and z_lo - r <= float(b[2]) <= z_hi + r),
            "vel_mps": [float(x) for x in v],
        }

    # ---------- the human: a pointer, and a hand on the duck ----------
    # Two more verbs, and between them everything the duck can know about the
    # person it lives with:
    #   pet_sense   where the mouse pointer is, in the duck's own metres
    #   pet_touch   a hand was laid on the duck, gently, and let go
    # Both are the overlay's to send: the sim has no window server and cannot
    # see a cursor, exactly the way it cannot see the Dock. The app measures,
    # the sim decides what it means — which is the same division of labour
    # `pet_config` already runs on.

    def _pet_cursor_state(self) -> dict:
        """What the duck makes of the last pointer sample it was given.

        Everything derived is derived HERE rather than in the app, for the
        reason `pet_state`'s screen block exists: two implementations of the
        same arithmetic drift, and the one that matters (`cursor.dist_m`, what
        a machine walks towards) has to agree with the duck's own odometry to
        the millimetre or the approach node oscillates around its own target.

        A stale sample keeps its `age_s` and loses everything else. That split
        is what lets a machine say both "the hand is gone" (`age_s > 2.5`) and
        "the hand is close" (`dist_m < 0.22`) without either question
        answering about a pointer that stopped existing two minutes ago — a
        null compares False in the grammar, which is the honest answer to
        "how far away is a cursor that is not there".
        """
        blank = {"present": False, "x_m": None, "z_m": None, "dx_m": None,
                 "dist_m": None, "age_s": 999.0, "near_floor": False,
                 "speed_mps": None}
        c = self._pet_cursor
        if c is None:
            return blank
        age = round(self.sim_time - c["t"], 3)
        if age > PET_CURSOR_STALE_S:
            return {**blank, "age_s": age}
        base_x = float(self.data.qpos[self.qpos_adr])
        dx = c["x_m"] - base_x
        return {
            "present": True,
            "x_m": c["x_m"],
            "z_m": c["z_m"],
            # Signed, and the sign is the whole point: it is which way to
            # walk. Positive is world +x, which the pet camera puts on the
            # right of the screen.
            "dx_m": dx,
            # Horizontal only. A cursor held directly over the duck is 0 m
            # away, because the duck cannot walk upwards to reach it.
            "dist_m": abs(dx),
            "age_s": age,
            "near_floor": bool(c["z_m"] <= self.pet["cursor_floor_m"]),
            "speed_mps": c["speed_mps"],
        }

    def _pet_touch_state(self) -> dict:
        """When the duck was last petted, and how often it ever has been."""
        t = self._pet_touch["t"]
        age = 999.0 if t is None else round(self.sim_time - t, 3)
        return {"petted": bool(t is not None and age < PET_TOUCH_RECENT_S),
                "age_s": age, "count": int(self._pet_touch["count"])}

    def _handle_pet_sense(self, req: dict) -> dict:
        """A pointer position, in the duck's metres. Cheap on purpose.

        No render, no mj_forward, no emote — this arrives five times a second
        for as long as the overlay is awake, and the only thing it may cost
        the sim thread is the queued tick it already spent getting here. It
        is also deliberately not a `push`: knowing where a hand is and being
        shoved by one are different events, and only one of them is physics.
        """
        if not self.pet_scene:
            return self._pet_not_a_pet_scene()
        if not req.get("present", True):
            # The pointer left this screen, or the app is going idle. Drop the
            # sample rather than let it age out: "gone" is information, and
            # two seconds of the duck still walking towards a cursor on
            # another display is two seconds of it looking broken.
            self._pet_cursor = None
            return {"ok": True, "cursor": self._pet_cursor_state()}
        try:
            x_m = float(req["x_m"])
            z_m = float(req["z_m"])
        except (KeyError, TypeError, ValueError):
            return {"ok": False, "error": "x_m and z_m must be numbers"}
        if not (math.isfinite(x_m) and math.isfinite(z_m)):
            return {"ok": False, "error": "x_m and z_m must be numbers"}
        # `t_s` is the SENDER's clock (the overlay stamps each sample as it
        # measures it), and it is what speed is computed from. Arrival time
        # is a lie here: sense posts queue on the sim thread behind ~40 ms
        # renders, so two samples sent 400 ms apart routinely land in the
        # same tick — measured live, that read 0.375 m/s of real hand as
        # 2.23 m/s once and as a dt of zero the next time, and the zero
        # reset the EMA to None, which is how `cursor.speed_mps > 0.05`
        # spent the first hands-on test never once being true.
        t_s = req.get("t_s")
        if t_s is not None:
            try:
                t_s = float(t_s)
            except (TypeError, ValueError):
                t_s = None
            else:
                if not math.isfinite(t_s):
                    t_s = None
        prev = self._pet_cursor
        speed = None
        if prev is not None:
            if t_s is not None and prev.get("t_s") is not None:
                dt = t_s - prev["t_s"]
            else:
                dt = self.sim_time - prev["t"]
            if dt > 0.001:
                inst = abs(x_m - prev["x_m"]) / dt
                # An EMA rather than the raw difference: a mouse moves in
                # jerks, and a guard cannot low-pass a number for itself.
                speed = (inst if prev["speed_mps"] is None else
                         PET_CURSOR_SPEED_EMA * inst
                         + (1.0 - PET_CURSOR_SPEED_EMA) * prev["speed_mps"])
            else:
                # Two samples from the same instant carry no new information
                # about the hand — keep what was known rather than forget it.
                speed = prev["speed_mps"]
        # The SIM clock, like ball_seen.age_s: a --fast daemon and a laptop
        # that slept both break wall-clock ages, and this one is read by
        # guards that decide whether anybody is still there. `t_s` rides
        # along for the next speed only; age never reads it.
        self._pet_cursor = {"x_m": x_m, "z_m": z_m, "t": self.sim_time,
                            "t_s": t_s, "speed_mps": speed}
        return {"ok": True, "cursor": self._pet_cursor_state()}

    def _handle_pet_touch(self, req: dict) -> dict:
        """A hand was laid on the duck — the gesture that is not a shove.

        The app decides what a gesture WAS (pet_map.classify_release: a click
        is a poke, a fast finish is a shove, a slow short stroke that ends on
        the animal is this); the sim decides what it MEANS. Nothing here
        reaches qvel, and that is the whole distinction the feature exists to
        draw: a poke moves the duck, a pet does not.

        The answer is one gesture and one sound, rate-limited, and it is
        honest about not happening. `start_emote(machine=False)` means a pet
        arriving while the head is busy steering (mid-approach, mid-kick) is
        refused in the emote engine's own words and those words come back in
        `note` — the duck keeps its eye on the ball, and the reply does not
        claim a nuzzle that never played.
        """
        if not self.pet_scene:
            return self._pet_not_a_pet_scene()
        kind = str(req.get("kind", "pet"))
        if kind not in PET_TOUCH_KINDS:
            return {"ok": False, "error":
                    f"unknown touch {kind!r} — this duck understands "
                    f"{', '.join(PET_TOUCH_KINDS)}"}
        for key in ("x_m", "z_m", "duration_s", "travel_m"):
            if req.get(key) is None:
                continue
            try:
                float(req[key])
            except (TypeError, ValueError):
                return {"ok": False, "error": f"{key} must be a number"}
        # The tally and the clock move whether or not the duck answers: being
        # petted twice in a second is two pets, and a machine watching
        # `touch.age_s` should see the second one even though it heard no coo.
        self._pet_touch["t"] = self.sim_time
        self._pet_touch["count"] += 1
        out = {"ok": True, "kind": kind, "acknowledged": False,
               "emote": None, "sound": None, "note": "",
               "count": self._pet_touch["count"],
               "cooldown_s": PET_TOUCH_ACK_COOLDOWN_S}
        ack_t = self._pet_touch["ack_t"]
        if ack_t is not None and self.sim_time - ack_t < PET_TOUCH_ACK_COOLDOWN_S:
            out["note"] = "still enjoying the last one"
            return out
        played = self.start_emote(PET_TOUCH_EMOTE, machine=False)
        if not played.get("ok"):
            # Mid-gesture, head bound, or no emote directory at all. Say so
            # in the daemon's own words rather than inventing a refusal.
            out["note"] = played.get("error", "")
            return out
        self._pet_touch["ack_t"] = self.sim_time
        out.update(acknowledged=True, emote=played.get("emote"),
                   sound=played.get("sound"), note=played.get("note", ""))
        return out

    # ---------- the hand that closes: picking the duck up ----------
    # The third thing a mouse can say, and the only one that is not a force.
    # A poke and a flick are impulses the controller has to survive; this is a
    # CONSTRAINT — scene_desktop.xml's `pet_carry` weld, activated between an
    # invisible mocap hand and the trunk, with the hand then dragged around by
    # the pointer. Nothing is animated while the duck dangles: `machines/
    # pet.toml`'s `carried` node commands zero velocity, which selects the
    # standing policy, and a standing policy with no ground under it flails.
    # That flail is not a bug to hide behind a keyframe — it is exactly what a
    # real Microduck does when you lift it off the floor, and it is the whole
    # reason this feature is honest.

    def _pet_can_carry(self) -> bool:
        """Does this compiled scene actually have a hand and a weld in it?

        Separate from `pet_scene`: a daemon can have walls, rails and ledges
        (an older `scene_desktop.xml`, or a hand-written one) and no pick-up,
        and the refusal should say which of the two is missing.
        """
        return (self._pet_carry_eq >= 0 and self._pet_hand_mocap >= 0
                and self._pet_trunk_body >= 0)

    def _pet_carry_limits(self) -> dict:
        """The box the hand may move in, and how fast. Sent with every reply.

        The app draws no boundary of its own from these — it just posts where
        the pointer is and lets the clamp below have the last word — but a
        gesture that is being silently trimmed should be able to say so, and
        `duck pet state` should be able to show why a duck stopped rising.
        """
        wall = max(0.05, self._pet_wall_x() - PET_CARRY_WALL_PAD_M)
        return {"x_m": [-wall, wall],
                "z_m": [PET_CARRY_MIN_Z_M, self.pet["carry_max_z_m"]],
                "hand_speed_mps": self.pet["carry_hand_speed_mps"],
                "timeout_s": PET_CARRY_TIMEOUT_S}

    def _pet_carry_clamp(self, x_m: float, z_m: float, y_m: float) -> list:
        """Where the hand is ALLOWED to be, given where it was asked to go.

        This is the whole safety story of the feature. The walls stop a duck
        that walks into them; nothing stops a hand, and a welded body follows
        a hand anywhere — through a wall, under the floor, off the display. So
        the target is clamped before it is ever chased, and the chase itself
        is rate-limited (`pet_carry_tick`), because a cursor can jump a whole
        screen between two samples and a teleporting constraint is a solver
        explosion rather than a pick-up.

        `y` is pinned to whatever depth the duck was grabbed at and then held
        inside the rails: the pet camera looks along y, so depth is invisible
        on screen and a hand that wandered in it would be moving the duck
        somewhere the human cannot see.
        """
        wall = max(0.05, self._pet_wall_x() - PET_CARRY_WALL_PAD_M)
        corridor = max(0.02, self.pet["corridor_m"] - 0.05)
        return [float(np.clip(x_m, -wall, wall)),
                float(np.clip(y_m, -corridor, corridor)),
                float(np.clip(z_m, PET_CARRY_MIN_Z_M,
                              self.pet["carry_max_z_m"]))]

    def _pet_carry_grab(self, token: str, target, grab=(0.0, 0.0)) -> dict:
        """Close the weld on the duck exactly where it is standing.

        Two things happen here and the second one is the one that matters.

        The hand is teleported onto the trunk's own origin, because a weld
        pulls the two bodies into their declared relative pose and a hand that
        started anywhere else would yank the duck across the screen.

        `grab` is the third thing, and it is bookkeeping rather than physics:
        the (x, z) offset from where the human pressed to where the weld
        actually holds on. Every later `move` adds it back, so the point that
        was grabbed is the point that stays under the pointer.

        And `eq_data[3:10]` is rewritten from the LIVE relative pose. MuJoCo
        bakes a weld's relpose from qpos0 at compile time, and on this scene
        that is the hand parked at z=3 against a duck at z=0.12 — measured,
        [0,0,0, 0,0,-2.88, 1,0,0,0, 1]. Activating it unedited drives the duck
        2.88 m straight down on the first step. With the hand moved onto the
        trunk, the honest relative pose is "no offset at all, wearing the
        trunk's current orientation", which is what goes in.
        """
        tb, mid, eid = self._pet_trunk_body, self._pet_hand_mocap, self._pet_carry_eq
        hand = np.array(self.data.xpos[tb], dtype=float)
        self.data.mocap_pos[mid] = hand
        self.data.mocap_quat[mid] = [1.0, 0.0, 0.0, 0.0]
        self.model.eq_data[eid, 0:3] = 0.0          # anchor, in body2's frame
        self.model.eq_data[eid, 3:6] = 0.0          # relpos: the hand IS the trunk
        self.model.eq_data[eid, 6:10] = self.data.xquat[tb]
        self.model.eq_data[eid, 10] = 1.0           # torquescale
        self.data.eq_active[eid] = 1
        track = deque(maxlen=PET_CARRY_TRACK)
        track.append((self.sim_time, hand.copy()))
        self._pet_carry = {"token": token, "t0": self.sim_time,
                           "last_move_t": self.sim_time,
                           "target": target, "track": track,
                           "grab": (float(grab[0]), float(grab[1]))}
        return self._pet_carry

    def _pet_carry_hand_vel(self, carry: dict):
        """How fast the hand was moving when it let go, per axis.

        Over the last PET_CARRY_VEL_WINDOW_S of the hand's own track rather
        than over the last tick: a 20 Hz channel against a 50 Hz loop means
        two of every five ticks moved the hand nowhere at all, and
        differencing those would throw the duck away in whichever direction
        the last sample happened to land.
        """
        track = carry["track"]
        if len(track) < 2:
            return np.zeros(3)
        t1, p1 = track[-1]
        t0, p0 = track[0]
        for t, p in track:
            if t1 - t <= PET_CARRY_VEL_WINDOW_S:
                t0, p0 = t, p
                break
        dt = t1 - t0
        if dt <= 0.0:
            return np.zeros(3)
        return (np.asarray(p1) - np.asarray(p0)) / dt

    def _pet_carry_release(self):
        """Open the weld and hand the duck whatever the hand was doing.

        The release velocity goes into the trunk's linear qvel, clamped to the
        same PET_PUSH_MAX every other gesture is: a duck let go mid-swing
        should fly, and a duck set down gently should not. Angular velocity is
        deliberately left alone — the controller is already fighting for its
        balance and a spin nobody asked for is not information, it is noise.

        Returns the velocity that landed, or None if nothing was being held.
        """
        carry, self._pet_carry = self._pet_carry, None
        if carry is None:
            return None
        if self._pet_carry_eq >= 0:
            self.data.eq_active[self._pet_carry_eq] = 0
        v = np.clip(self._pet_carry_hand_vel(carry), -PET_PUSH_MAX, PET_PUSH_MAX)
        self.data.qvel[self.qvel_adr:self.qvel_adr + 3] = v
        return [float(x) for x in v]

    def pet_carry_tick(self):
        """Move the hand one control step towards where the pointer said.

        Called from `run_loop` beside `mouth_tick`, BEFORE the physics steps,
        so the pose the solver sees this tick is the pose the human asked for
        this tick — a mocap body written after the steps is a frame behind,
        and one frame behind at 1.5 m/s is three centimetres of lag in a
        constraint that is meant to feel like fingers.

        The deadman lives here too, and it is not paranoia: `eq_active` is
        sim state, so an overlay that crashed, was force-quit or lost its
        connection mid-lift would otherwise leave the duck hanging in the air
        for the rest of the session with nothing able to put it down.
        """
        carry = self._pet_carry
        if carry is None:
            return
        if self.sim_time - carry["last_move_t"] > PET_CARRY_TIMEOUT_S:
            self._pet_carry_release()
            return
        mid = self._pet_hand_mocap
        here = np.array(self.data.mocap_pos[mid], dtype=float)
        want = np.array(carry["target"], dtype=float)
        step = self.pet["carry_hand_speed_mps"] * DECIMATION * TIMESTEP
        delta = want - here
        dist = float(np.linalg.norm(delta))
        if dist > step > 0.0:
            want = here + delta * (step / dist)
        self.data.mocap_pos[mid] = want
        carry["track"].append((self.sim_time, want.copy()))

    def _pet_carry_state(self) -> dict:
        """What the overlay and the machine are told about the grip."""
        carry = self._pet_carry
        if carry is None:
            return {"carried": False, "token": None, "held_s": 0.0,
                    "hand_m": None}
        return {"carried": True, "token": carry["token"],
                "held_s": round(self.sim_time - carry["t0"], 3),
                "hand_m": [float(v)
                           for v in self.data.mocap_pos[self._pet_hand_mocap]]}

    def _handle_pet_carry(self, req: dict) -> dict:
        """`start` / `move` / `end` — one grip, and the daemon owns the token.

        THE DAEMON MINTS IT, never the app. A grab is a piece of sim state
        that outlives the gesture that made it (the deadman can end one, a
        `reset` can, a second overlay could), so "is this still the hand I am
        holding" has to be answerable by the side that knows — and a `move` or
        an `end` carrying a token that is no longer current changes nothing
        and says 409. That is what stops a stale `end`, from a gesture the
        window server abandoned when a Space switched, releasing a grab the
        human started afterwards.

        A seated duck is stood up first. Welding a sitter and putting it down
        leaves `sit_mode` set with the duck in the air, and nothing in the
        machine or the policy recovers from that state — the sit is a
        transition that has already been consumed (docs/mcp-design-notes.md
        says the same thing about swapping a policy into a seated body).
        """
        if not self.pet_scene:
            return self._pet_not_a_pet_scene()
        if not self._pet_can_carry():
            return {"ok": False, "error":
                    f"scene {self.scene!r} has no {PET_CARRY_EQ!r} weld — this "
                    "duck cannot be picked up (scenes/scene_desktop.xml grew "
                    "one; an older copy of it did not)"}
        action = str(req.get("action", ""))
        if action not in PET_CARRY_ACTIONS:
            return {"ok": False, "error":
                    f"unknown carry action {action!r} — this hand understands "
                    f"{', '.join(PET_CARRY_ACTIONS)}"}
        coords = None
        if action in ("start", "move"):
            if req.get("x_m") is not None or req.get("z_m") is not None:
                try:
                    coords = (float(req["x_m"]), float(req["z_m"]))
                except (KeyError, TypeError, ValueError):
                    return {"ok": False,
                            "error": "x_m and z_m must be numbers"}
                if not all(math.isfinite(v) for v in coords):
                    return {"ok": False,
                            "error": "x_m and z_m must be numbers"}
        carry = self._pet_carry
        note = ""

        if action == "start":
            if carry is not None:
                if req.get("token") == carry["token"]:
                    # A restatement of the grab we already answered — the app
                    # never does this, but a retried POST is not a conflict.
                    return self._pet_carry_reply(action, note="already yours")
                return {"ok": False, "conflict": True, "error":
                        "the duck is already in somebody's hand (that grab is "
                        "still live — let go of it, or wait for the deadman)"}
            if self.policy.sit_mode:
                self._handle_trick("stand")
                note = "stood it up first — a sitter cannot be put back down"
            token = secrets.token_hex(4)
            pos = self.data.qpos[self.qpos_adr:self.qpos_adr + 3]
            # THE HAND CLOSES ON THE TRUNK, NOT ON THE CURSOR. `_pet_carry_grab`
            # teleports the hand onto the trunk's origin and writes a zero
            # relpos, so the weld holds the duck by that one point — and the
            # press landed wherever on the animal the human aimed, because
            # `pet_app.hit_rect_pt`'s box is the whole silhouette. Driving the
            # hand at the raw cursor would therefore SNAP the trunk under the
            # pointer: measured on this scene, a press on the head (z 0.25
            # against a trunk at 0.12) yanks the duck 0.13 m upwards, and a
            # press on the feet (z 0.02, clamped to PET_CARRY_MIN_Z_M) presses
            # it 6.6 cm below standing until the feet penetrate the floor
            # against ~55 N of constraint force. Roughly half of all presses
            # are below trunk height, so both are the common case.
            #
            # So the grab remembers the offset instead, `move` adds it back,
            # and the point the human took hold of is the point that follows
            # the pointer. A `start` with no coordinates at all (the CLI, the
            # tests) has no grab point to speak of and gets a zero offset,
            # which is the old behaviour exactly.
            grab = ((float(pos[0]) - coords[0], float(pos[2]) - coords[1])
                    if coords is not None else (0.0, 0.0))
            target = self._pet_carry_clamp(float(pos[0]), float(pos[2]),
                                           float(pos[1]))
            self._pet_carry_grab(token, target, grab)
            return self._pet_carry_reply(action, note=note)

        if carry is None or req.get("token") != carry["token"]:
            return {"ok": False, "conflict": True, "error":
                    "not the current carry (token expired — the hand was "
                    "released)"}
        if action == "move":
            if coords is not None:
                # Clamped AFTER the offset, not before: the bounds are about
                # where the weld's hand may be, and the hand is the trunk.
                gx, gz = carry["grab"]
                carry["target"] = self._pet_carry_clamp(
                    coords[0] + gx, coords[1] + gz, carry["target"][1])
            # Even a move that named no coordinates is a heartbeat: the hand
            # is still there, and the deadman is the only thing listening.
            carry["last_move_t"] = self.sim_time
            return self._pet_carry_reply(action)
        released = self._pet_carry_release()
        return self._pet_carry_reply(action, released=released,
                                     note="put down")

    def _pet_carry_reply(self, action: str, released=None,
                         note: str = "") -> dict:
        state = self._pet_carry_state()
        carry = self._pet_carry
        return {"ok": True, "action": action, "note": note,
                "token": state["token"], "carried": state["carried"],
                "held_s": state["held_s"], "hand_m": state["hand_m"],
                "target_m": ([carry["target"][0], carry["target"][2]]
                             if carry is not None else None),
                "limits": self._pet_carry_limits(),
                "released_vel_mps": released}

    def _pet_inhabited(self):
        """Is an MCP client driving right now, and how long since it spoke?"""
        if not self._mcp_intent_t:
            return False, None, None
        age = time.time() - self._mcp_intent_t
        return age < PET_INHABITED_S, round(age, 2), self._mcp_intent_cmd

    def _handle_pet_state(self, req: dict = None) -> dict:
        """Everything the overlay window needs to place and label itself.

        Unrounded, unlike get_state(): position_m is rounded to a millimetre
        for humans, and a millimetre is two thirds of a pixel of window jitter
        at 656 px/m. Nothing else reads these fields, so they can be honest.
        """
        adr, vadr = self.qpos_adr, self.qvel_adr
        pos = self.data.qpos[adr:adr + 3]
        _, _, yaw = quat_to_rpy(self.data.qpos[adr + 3:adr + 7])
        proj_g = self.policy.get_projected_gravity()
        upright = bool(proj_g[2] < -0.7)
        inhabited, age_s, cmd = self._pet_inhabited()
        p = self.pet
        return {
            "ok": True,
            "sim_time_s": self.sim_time,
            # The pose is real on any scene; the walls and ledges are not.
            # pet_frame/config/world refuse elsewhere, but a pose is a pose,
            # so this one answers and says which world it is answering about.
            "pet_scene": self.pet_scene,
            "base_x_m": float(pos[0]),
            "base_y_m": float(pos[1]),
            "base_z_m": float(pos[2]),
            "heading_deg": math.degrees(yaw),
            "vel_world_mps": [float(v) for v in self.data.qvel[vadr:vadr + 3]],
            "upright": upright,
            "fallen": not upright,
            "sitting": bool(self.policy.sit_mode),
            "active_policy": self.policy.current_policy,
            "behavior": self.policy.behavior_mode,
            "vel_cmd": [float(v) for v in self.policy.vel_cmd],
            "machine": ({"name": self.machine.name, "armed": self.machine.armed,
                         "node": self.machine.current}
                        if self.machine is not None else None),
            # "Claude has the wheel" — an MCP intent inside PET_INHABITED_S.
            "inhabited": inhabited,
            "inhabited_age_s": age_s,
            "inhabited_cmd": cmd,
            # The human, from the duck's side of the glass. Unrounded like
            # everything else here — `dist_m` is what a machine walks towards,
            # and a millimetre is two thirds of a pixel at 656 px/m.
            "cursor": self._pet_cursor_state(),
            "touch": self._pet_touch_state(),
            # ...and whether the hand actually closed. The token is in here on
            # purpose: it is what a second overlay, or `duck pet state` in a
            # terminal, reads to find out that somebody else is holding the
            # duck right now.
            "carry": self._pet_carry_state(),
            # The toy, in the same unrounded metres — `null` on a scene
            # without one, so the app can ask "is there a ball" with the same
            # question it asks "where is it".
            "ball": self._pet_ball_state(),
            # Precomputed screen mapping, so the app does no arithmetic it
            # could get subtly out of step with the renderer's.
            "screen": {
                "center_offset_px": float(pos[0]) * p["px_per_meter"],
                "floor_px_from_bottom": p["floor_pad_px"],
                "px_per_meter": p["px_per_meter"],
                # What world z the frame's floor row is showing. 0 for a duck
                # on the Dock, which is every duck that is not being carried;
                # above the lift trigger it is how far the WINDOW has to climb
                # the screen (pet_map.ScreenMap.window_origin).
                "frame_floor_z_m": self._pet_lift_m(),
                "frame_px": p["frame_px"],
            },
            "config": self._pet_config_state(),
        }

    def _pet_camera(self, size_px: int = None) -> "mujoco.MjvCamera":
        """The tracking camera: centred on the duck, level with the world.

        lookat.x follows the duck so it is always in the middle of its window
        (the WINDOW is what travels); lookat.z follows only ABOVE
        `lift_trigger_m`, or the Dock's edge would slide up and down the frame
        and a stumble would look like a camera move instead of a fall. Below
        that trigger the floor line is pinned exactly as it always was — see
        `_pet_lift_m` for the measurement. lookat.y follows only to keep the
        duck inside the depth clip — under orthographic it changes nothing on
        screen, which is exactly why the side view can be trusted.

        `size_px` is the frame actually being rendered, which is not always
        the configured one: `/pet/frame?size_px=` can ask for another. Framing
        off the config instead would leave the floor line somewhere other than
        `floor_pad_px` up from the bottom, and `pet_state.screen` — which is
        what the app hangs the window off — would be quietly describing a
        different picture.
        """
        p = self.pet
        cam = mujoco.MjvCamera()
        pos = self.data.qpos[self.qpos_adr:self.qpos_adr + 3]
        # World z of the frame's centre: put the floor `floor_pad_px` above
        # the bottom edge and everything else follows.
        floor_from_center_px = (size_px or p["frame_px"]) / 2 - p["floor_pad_px"]
        # ...plus the lift, which is 0 for every duck that is on the floor —
        # see `_pet_lift_m` for why the follow has a dead band under it.
        cam.lookat[:] = [float(pos[0]), float(pos[1]),
                         self._pet_lift_m()
                         + floor_from_center_px / p["px_per_meter"]]
        cam.distance = p["camera_distance_m"]
        cam.azimuth = p["azimuth_deg"]
        cam.elevation = p["elevation_deg"]
        return cam

    def _pet_alpha(self, rgb):
        """What is drawn, split into the two things it can be.

        Returns `(alpha, duck_mask, ball_mask)`. The alpha is everything the
        window should show — duck and toy together, because they are one
        picture — and the two masks are how the frame answers WHICH of them a
        given pixel belongs to. They are separate because `bbox` is a hit
        rectangle, not decoration: the overlay swallows a click inside it, and
        one box drawn round a duck and a ball a window apart is a duck you
        could grab from anywhere between them.

        Segmentation pass: render() returns (H, W, 2) int32 of (object id,
        object type), so looking each id up in the tables built at load is the
        exact cutout — every duck part, nothing of the scene, no colour
        heuristic to fool. The mouth plate is a body of its own and comes
        along for free; walls and ledges are excluded by name even though the
        group cull already keeps them out of the render.

        Chroma fallback for a host that cannot do the second pass: key the
        backdrop's magenta. Worse (it eats any magenta highlight on the duck)
        but it is one render instead of two and it never returns a blank duck.
        It also cannot tell a duck from a ball — the whole method is "not the
        backdrop" — so it says so by returning no masks at all, and the frame
        reports `ball_bbox: null` rather than guessing.
        """
        if self._pet_seg_ok:
            try:
                self._pet_renderer.enable_segmentation_rendering()
                try:
                    seg = self._pet_renderer.render()
                finally:
                    self._pet_renderer.disable_segmentation_rendering()
                ids = np.clip(seg[:, :, 0], 0, self.model.ngeom - 1)
                is_geom = seg[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
                duck = is_geom & self._pet_duck_geom[ids]
                ball = is_geom & self._pet_ball_geom[ids]
                return ((duck | ball).astype(np.uint8) * 255, duck, ball)
            except Exception as e:
                self._pet_seg_ok = False
                print(f"note: pet segmentation pass disabled ({e}) — falling "
                      "back to the chroma key")
        d = np.abs(rgb.astype(np.int16) - np.asarray(PET_CHROMA_RGB, dtype=np.int16))
        return ((~(d.max(axis=2) <= PET_CHROMA_TOL)).astype(np.uint8) * 255,
                None, None)

    @staticmethod
    def _pet_compose(rgb, alpha, ss: int):
        """Supersampled RGBA -> the frame the window shows.

        Premultiply BEFORE the box filter, or the background bleeds into every
        edge pixel: the segmentation mask is perfectly binary (measured:
        np.unique == [0, 255]) while the RGB pass is antialiased against the
        backdrop, so a naive downsample would fringe the duck magenta. With
        premultiplied averaging the colour of an edge pixel is the average of
        the duck pixels that landed in it and nothing else — which is what
        antialiasing means.
        """
        if ss <= 1:
            return np.dstack([rgb, alpha])
        h, w = alpha.shape
        h2, w2 = h // ss, w // ss
        a = alpha[:h2 * ss, :w2 * ss].astype(np.float32) / 255.0
        pm = rgb[:h2 * ss, :w2 * ss].astype(np.float32) * a[:, :, None]
        a = a.reshape(h2, ss, w2, ss).mean(axis=(1, 3))
        pm = pm.reshape(h2, ss, w2, ss, 3).mean(axis=(1, 3))
        out = np.clip(pm / np.maximum(a, 1e-6)[:, :, None], 0, 255).astype(np.uint8)
        return np.dstack([out, np.round(a * 255).astype(np.uint8)])

    def _pet_renderer_for(self, render_px: int) -> "mujoco.Renderer":
        """The renderer for this size, kept warm — see PET_RENDERER_CACHE.

        A `mujoco.Renderer` is a GL context and an MjrContext; building one
        costs tens of milliseconds *on the sim thread*, and the previous
        one-renderer version rebuilt it on every frame as soon as two
        consumers wanted two sizes. Holding a handful is far cheaper than
        holding the physics up. The scene option is built once and shared —
        it does not depend on the size.
        """
        r = self._pet_renderers.pop(render_px, None)
        if r is None:
            self.model.vis.global_.offwidth = max(
                int(self.model.vis.global_.offwidth), render_px)
            self.model.vis.global_.offheight = max(
                int(self.model.vis.global_.offheight), render_px)
            r = mujoco.Renderer(self.model, height=render_px, width=render_px)
        self._pet_renderers[render_px] = r      # move to the MRU end
        while len(self._pet_renderers) > PET_RENDERER_CACHE:
            dead = self._pet_renderers.pop(next(iter(self._pet_renderers)))
            try:
                dead.close()
            except Exception:
                pass
        if self._pet_option is None:
            # Group 4 is the invisible world: floor, walls, rails, ledges.
            # Culling them here rather than making them transparent is the
            # difference between "not drawn" and "drawn as a hole" — a
            # transparent wall still occludes the duck in the SEGMENTATION
            # pass, and the duck would come back with bites taken out of it.
            self._pet_option = mujoco.MjvOption()
            self._pet_option.geomgroup[:] = 0
            for g in (0, 1, 2):
                self._pet_option.geomgroup[g] = 1
        return r

    def _handle_pet_frame(self, req: dict) -> dict:
        """One RGBA cutout of the duck and its toy, plus the pose it was taken at.

        Its own renderer, never `self._renderer`: that one is clamped to
        640x480 and shared with the AX debug page, and toggling segmentation
        on it would corrupt the page's frames mid-stream.

        Two boxes come back beside the picture, not one. `bbox` is the DUCK
        and only the duck; `ball_bbox` is the toy, `null` when it is off frame
        or when the chroma fallback is running (which cannot tell them apart
        and says so). The overlay hit-tests both, and which one a press landed
        on is what decides whether the drag that follows shoves a duck or
        rolls a ball.

        Budget: rendering runs on the 50 Hz sim thread, and ~40 fps of it
        stops physics dead (measured: sim/wall 0.012). The app should stay at
        or under 25 fps. The one-tick cache below is the cheap insurance — two
        pollers on one daemon get one render between them instead of two.
        """
        if not self.pet_scene:
            return self._pet_not_a_pet_scene()
        p = self.pet
        size = int(req.get("size_px") or p["frame_px"])
        ss = int(req.get("supersample") or p["supersample"])
        size = max(PET_FRAME_MIN_PX, min(PET_FRAME_MAX_PX, size))
        ss = max(1, min(3, ss))
        render_px = size * ss
        if render_px > PET_OFFSCREEN_MAX_PX:
            ss = max(1, PET_OFFSCREEN_MAX_PX // size)
            render_px = size * ss
        state = self._handle_pet_state()
        key = (render_px, ss, self.sim_time, p["px_per_meter"],
               p["floor_pad_px"], p["camera_distance_m"], p["azimuth_deg"],
               p["elevation_deg"])
        if self._pet_cache is not None and self._pet_cache[0] == key:
            png, w, h, bbox, ball_bbox = self._pet_cache[1:]
            # `cached` says where the PNG came from; `alpha` says what the
            # boxes MEAN, and the app now branches on it (`pet_app.hit_rect_pt`
            # cannot trust a chroma `bbox` to be duck-only). Reporting "cache"
            # there answered the wrong question and hid the one that matters.
            return {**state, "png": png, "width": w, "height": h,
                    "bbox": bbox, "ball_bbox": ball_bbox, "cached": True,
                    "alpha": "segmentation" if self._pet_seg_ok else "chroma"}
        model = self.model
        was_ortho = int(model.vis.global_.orthographic)
        was_fovy = float(model.vis.global_.fovy)
        try:
            self._pet_renderer = self._pet_renderer_for(render_px)
            self._pet_renderer_px = render_px
            # fovy is degrees under perspective and METRES under orthographic;
            # both live on the shared model, so borrow and give back. Every
            # render is on the sim thread, so nothing can look in between.
            model.vis.global_.orthographic = 1
            model.vis.global_.fovy = size / p["px_per_meter"]
            self._pet_renderer.update_scene(self.data,
                                            camera=self._pet_camera(size),
                                            scene_option=self._pet_option)
            rgb = self._pet_renderer.render().copy()
            alpha, duck_mask, ball_mask = self._pet_alpha(rgb)
        except Exception as e:
            return {"ok": False, "error": f"pet render failed: {e} (offscreen "
                    "rendering is unavailable under mjpython — run the pet "
                    "daemon headless)"}
        finally:
            model.vis.global_.orthographic = was_ortho
            model.vis.global_.fovy = was_fovy
        rgba = self._pet_compose(rgb, alpha, ss)
        buf = io.BytesIO()
        from PIL import Image
        # compress_level 1: this is a loopback stream at 20+ fps, so the CPU
        # spent squeezing the last 20% out of a 30 KB frame is CPU the physics
        # does not get.
        Image.fromarray(rgba, "RGBA").save(buf, format="PNG", compress_level=1)
        png = buf.getvalue()
        h, w = rgba.shape[:2]
        # From the MASKS, not from the composed alpha, because the composed
        # alpha is both of them at once. `ss` is passed so the boxes land in
        # OUTPUT pixels — the same pixels the PNG is measured in, which is the
        # only frame of reference the app has.
        if duck_mask is None:
            # Chroma: one keyed silhouette, no way to say which half is which.
            # `bbox` stays the honest answer to "where are the drawn pixels";
            # `ball_bbox` is null rather than a guess.
            bbox, ball_bbox = self._pet_bbox(rgba[:, :, 3]), None
        else:
            bbox = self._pet_bbox(duck_mask, ss)
            ball_bbox = self._pet_bbox(ball_mask, ss)
        self._pet_cache = (key, png, w, h, bbox, ball_bbox)
        return {**state, "png": png, "width": w, "height": h, "bbox": bbox,
                "ball_bbox": ball_bbox, "cached": False,
                "alpha": "segmentation" if self._pet_seg_ok else "chroma"}

    @staticmethod
    def _pet_bbox(mask, ss: int = 1):
        """Where that mask's pixels actually are, in OUTPUT frame pixels,
        top-left origin, as [x0, y0, x1, y1] half-open.

        The overlay window is a transparent square sitting over the Dock, so
        it has to decide per cursor position whether to swallow a click or let
        it fall through to a Dock icon. It can only do that honestly if it
        knows the silhouette, and the silhouette is already in our hands: the
        segmentation mask cost nothing extra and two `nonzero` reductions on a
        binary channel are cheaper than the app scanning the PNG it just
        decoded. None when the thing is off frame — the app then falls back to
        arithmetic (`pet_app.hit_rect_pt` / `ball_rect_pt`).

        `ss` folds the supersample down the way `_pet_compose` does, and it
        has to agree with it exactly: a box filter over a binary mask leaves
        an output pixel non-zero iff ANY of its ss×ss inputs was set, so
        `.any()` over the same blocks is the same picture — measured against
        the composed alpha, not assumed. Doing it here rather than reading the
        composed channel back is what lets the two boxes be separate at all.
        """
        if mask is None:
            return None
        if ss > 1:
            h, w = mask.shape
            h2, w2 = h // ss, w // ss
            mask = mask[:h2 * ss, :w2 * ss].reshape(h2, ss, w2, ss).any(axis=(1, 3))
        cols = np.flatnonzero(mask.any(axis=0))
        rows = np.flatnonzero(mask.any(axis=1))
        if not cols.size or not rows.size:
            return None
        return [int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1]

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
            if client == "mcp" and req.get("cmd") not in ("state", "ping"):
                # A mind took the wheel. Polls do not count: watching the duck
                # is not driving it, and the pet's "inhabited" light saying so
                # would make it useless. (mcp_server.py tags every request.)
                self._mcp_intent_t = time.time()
                self._mcp_intent_cmd = req.get("cmd")
            try:
                resp = self.handle(req)
            except Exception as e:
                resp = {"ok": False, "error": repr(e)}
            respq.put(resp)
            self._log_event(client, req, resp)

    def _log_event(self, client: str, req: dict, resp: dict):
        cmd = req.get("cmd", "?")
        # Ambient traffic, not agent intent. The pet polls belong here or a
        # 20 fps overlay flushes the whole 300-entry feed in fifteen seconds.
        if cmd in ("camera_web", "ping", "mouth") or cmd in PET_AMBIENT_CMDS:
            return
        # A carry is three events, not sixty: `start` and `end` are acts and
        # belong on the feed, and the `move`s in between are the same channel
        # `pet_sense` is — a position, restated twenty times a second, which
        # would flush the 500-entry feed in twenty-five seconds (G18 again).
        if cmd == "pet_carry" and req.get("action") == "move":
            return
        if not resp.get("ok"):
            note = resp.get("error", "error")
        elif cmd == "trick":
            note = "started" if resp.get("started") else "refused"
        elif cmd in ("set_velocity", "reset", "state", "look", "push"):
            note = f"{resp.get('active_policy', '')}{'' if resp.get('upright', True) else ' DOWN'}"
        else:
            # Whatever the handler wanted said about a success: which emote
            # played and why its sound stayed home, what a machine load did.
            note = str(resp.get("note") or "")
        self._event_id += 1
        self.events.append({
            "id": self._event_id, "t": time.time(), "client": client, "cmd": cmd,
            "args": {k: v for k, v in req.items() if k != "cmd"},
            "ok": bool(resp.get("ok")), "note": note.strip(),
        })


def _wire_safe(obj):
    """Last resort for the JSON-lines control plane: describe, never crash."""
    if isinstance(obj, (bytes, bytearray)):
        return f"<{len(obj)} bytes — in-process only, use the /pet HTTP routes>"
    return repr(obj)


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
                # `default=` so one handler returning something unencodable
                # (pet_frame hands the webui raw PNG bytes, which are for
                # in-process consumers) cannot kill a client's connection.
                f.write((json.dumps(resp, default=_wire_safe) + "\n").encode())
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
        # The human's hand, moved before the steps for the same reason the
        # beak is: a mocap body written after them is a frame behind, and one
        # frame behind at 1.5 m/s is three centimetres of lag in a constraint
        # that is meant to feel like fingers.
        sim.pet_carry_tick()
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
    # Not a closed enum any more: `desktop` is this repo's own scene (see
    # scenes/scene_desktop.xml and LOCAL_SCENES), and a path lets a scene be
    # tried without editing the server. Unknown names are rejected by
    # DuckSim.__init__ with the same list argparse would have printed.
    parser.add_argument("--scene", default="ball", metavar="NAME",
                        help="scene to load: "
                             + ", ".join(sorted(SCENES) + sorted(LOCAL_SCENES))
                             + ", or a path to an .xml (default: ball)")
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
