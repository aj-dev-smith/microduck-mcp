"""MCP server exposing the Microduck sim as tools.

Stdio MCP server (Python MCP SDK v2); each tool forwards an intent to the sim
server's Unix socket (start it first: `duck-sim`). Register with any MCP
client, e.g.:

    claude mcp add duck -- uv --directory /path/to/microduck-mcp run duck-mcp

Design notes (see docs/mcp-design-notes.md for the research behind this):
tools — not resources — are the interface, including for reads: agent-driven
state polling and camera frames are tool-shaped, and tools are the one MCP
primitive every client supports. Results use typed structured output; errors
the model should see and recover from are raised as ToolError.
"""

import os
import subprocess
import time
from typing import Any, Literal

from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field

from .client import request

mcp = MCPServer("microduck")

_READ = ToolAnnotations(read_only_hint=True, open_world_hint=False)
_INTENT = ToolAnnotations(read_only_hint=False, destructive_hint=False,
                          idempotent_hint=True, open_world_hint=False)
_EPISODIC = ToolAnnotations(read_only_hint=False, destructive_hint=False,
                            idempotent_hint=False, open_world_hint=False)
_RESET = ToolAnnotations(read_only_hint=False, destructive_hint=True,
                         idempotent_hint=True, open_world_hint=False)

# How long to let an intent take effect before sampling the returned state.
SETTLE_S = 0.3
MAX_DRIVE_DURATION_S = 10.0


class BodyVelocity(BaseModel):
    forward: float = Field(description="Body-frame forward velocity, m/s")
    lateral: float = Field(description="Body-frame leftward velocity, m/s")


class BallSeen(BaseModel):
    """What the head camera saw, NOT where the ball actually is.

    Derived from an orange-blob detection on a 320x240 render from the duck's
    own camera at ~5 Hz — the honest signal the real robot will have, standing
    in for its mediad service. Angles are measured from the camera's optical
    axis, which is mounted 20 deg below the head's forward axis, so a ball on
    the floor a few steps ahead reads elevation ~-20 deg, not 0. Ball out of
    frame, behind the duck, or occluded -> visible false and null fields;
    duck_look pans the camera, so a null is an invitation to look around.
    """

    visible: bool = Field(description="True if the ball was in the last frame")
    distance_m: float | None = Field(
        description="Range from the camera to the ball centre, meters, from the "
        "blob's apparent size. Within ~10% out to 1.5 m; reads long when the "
        "blob is clipped by the frame edge.")
    bearing_deg: float | None = Field(
        description="Horizontal angle off the optical axis, degrees, positive "
        "to the duck's left — same sign as the wz yaw-rate command, so a "
        "positive bearing means turn with positive wz to face the ball.")
    elevation_deg: float | None = Field(
        description="Vertical angle off the optical axis, degrees, positive up")
    ground_distance_m: float | None = Field(
        default=None, description="Floor-plane range from the camera, meters "
        "(slant range corrected by the camera's own height)")
    est_forward_m: float | None = Field(
        default=None, description="Estimated ball position in the TRUNK's yaw "
        "frame, meters forward — camera ray + the robot's kinematics, the "
        "sensed twin of ball_offset_m. Kick pocket: forward~0.09, left~-0.042")
    est_left_m: float | None = Field(
        default=None, description="Estimated ball position, trunk yaw frame, "
        "meters to the left")
    speed_mps: float | None = Field(
        default=None, description="Estimated ball ground speed, m/s, from "
        "differencing the sighting's world-frame estimate across detector "
        "ticks (own-kinematics odometry cancels the duck's motion). Null "
        "until the ball has been tracked for ~0.25 s. Noisy while walking "
        "(the camera bias breathes with the gait); a parked ball reads "
        "near 0, a freshly kicked one ~1 m/s.")
    age_s: float = Field(description="Sim seconds since the ball was last "
                         "SEEN (the detector runs at ~5 Hz, so <=0.2 while "
                         "in view; grows while the ball is out of frame)")


class GoalSeen(BaseModel):
    """What the head camera saw of the GOAL (pitch scene only) — fake mediad
    part 2. The white goal frame is picked out of the same 5 Hz head-camera
    render as the ball, separated from the equally-white pitch lines and
    clouds by ray elevation computed from the robot's own kinematics: only
    the crossbar band lives just under the horizon. The est_* fields are
    dead-reckoned from the last sighting via own odometry — the goal is
    world-fixed, so they stay live while the head is tilted down at the ball
    (which points the camera at the grass and hides the goal entirely).
    """

    visible: bool = Field(description="True if the goal frame was in the last frame")
    bearing_deg: float | None = Field(
        description="Horizontal angle to the goal-mouth centre, degrees,"
        " positive to the duck's left (trunk yaw frame; robust to head pitch)")
    width_deg: float | None = Field(
        description="Angular width of the detected frame, degrees")
    distance_m: float | None = Field(
        description="Range to the mouth from its angular width, meters —"
        " coarse (±30%); null on partial views (goal clipped by frame edge)")
    age_s: float = Field(description="Sim seconds since the goal was last seen")
    est_bearing_deg: float | None = Field(
        default=None, description="Dead-reckoned bearing to the remembered "
        "goal, trunk yaw frame — live every tick once the goal has been "
        "sighted, even with the goal out of frame. Null until first sighted.")
    est_distance_m: float | None = Field(
        default=None, description="Dead-reckoned range to the remembered "
        "goal, meters; null until a ranged sighting has happened.")


class DuckState(BaseModel):
    """Snapshot of the robot. Positions are world-frame; velocities body-frame."""

    model_config = ConfigDict(extra="allow")

    ok: bool
    sim_time_s: float = Field(description="Sim clock, seconds. The sim runs in "
                              "real time; this snapshot is stale on arrival.")
    position_m: list[float] = Field(description="Trunk world position [x, y, z], meters")
    rpy_deg: list[float] = Field(description="Trunk orientation [roll, pitch, yaw], degrees")
    vel_body_mps: BodyVelocity
    yaw_rate_rps: float = Field(description="Yaw rate, rad/s, counterclockwise positive")
    trunk_height_mm: float = Field(description="Trunk height above floor, mm (~116 standing)")
    upright: bool = Field(description="False once tilted past ~45 deg. There is no "
                          "self-recovery policy: if the duck falls, use duck_reset.")
    active_policy: str = Field(description="Which ONNX policy is driving: standing, "
                               "walking, sit, ground_pick, kick_left/right, roulade")
    vel_cmd: list[float] = Field(description="Current sticky velocity intent [vx, vy, wz]")
    sitting: bool
    behavior: str | None = Field(description="Episodic trick currently running, else null")
    ground_pick: bool
    mouth: float | None = Field(
        default=None, description="Beak opening intent, 0 (closed) to 1 "
        "(open); null if the loaded model has no animatable mouth")
    ball_seen: BallSeen | None = Field(
        default=None, description="Camera-derived ball sighting — the sensed "
        "view. Prefer it over ball_position_m when you want the robot to act "
        "on what it can actually perceive.")
    goal_seen: GoalSeen | None = Field(
        default=None, description="Camera-derived goal sighting (pitch scene "
        "only) — how the duck aims. Absent on scenes without a goal.")
    ball_position_m: list[float] | None = Field(
        default=None, description="Ball world position [x, y, z], meters (ball scene only)")
    ball_offset_m: dict[str, float] | None = Field(
        default=None, description="Ball offset from the trunk in the robot's yaw "
        "frame: {forward, left}, meters. Kick staging puts the ball at "
        "forward=0.09, left=±0.042 — aim for that spot before an unstaged kick.")


def _call(req: dict, retries_note: str = "",
          timeout: float = 12.0) -> dict[str, Any]:
    try:
        resp = request({**req, "client": "mcp"}, timeout=timeout)
    except (ConnectionRefusedError, FileNotFoundError, TimeoutError) as e:
        raise ToolError(
            f"Simulator not reachable ({e.__class__.__name__}). Start it with "
            "`duck-sim` (or `uv run mjpython -m microduck_mcp.sim_server --viewer` "
            "for the watchable version) and retry.") from e
    if not resp.get("ok"):
        raise ToolError(resp.get("error", "sim rejected the command") + retries_note)
    return resp


def _state_after(settle_s: float = SETTLE_S) -> dict[str, Any]:
    time.sleep(settle_s)
    return _call({"cmd": "state"})


@mcp.tool(title="Duck telemetry", annotations=_READ)
def duck_state() -> DuckState:
    """Current robot state: pose, body-frame velocity, active policy, whether it
    is upright/sitting/mid-trick, and the ball position. Cheap — poll freely."""
    return DuckState(**_call({"cmd": "state"}))


@mcp.tool(title="Drive the duck", annotations=_INTENT)
def duck_drive(vx: float, vy: float = 0.0, wz: float = 0.0,
               duration_s: float | None = None) -> DuckState:
    """Set the walking velocity intent. vx: forward m/s (max ±0.3; the policy
    tracks ~half the commanded speed, so command 0.25+ for a brisk walk),
    vy: leftward m/s (max ±0.2; lateral tracking is nearly useless — prefer
    turning), wz: counterclockwise yaw rate rad/s (max ±1.5; below ~1.2 the
    duck barely turns in place, so command ±1.5 for point turns — ~45 deg/s).
    Nonzero engages the walking policy; all-zero hands back to standing.

    With duration_s (max 10): drive for that long, then stop and return the
    resulting state — one call instead of drive/poll/stop. Without it the
    intent persists until changed: the robot keeps walking between your tool
    calls, and the sim runs in real time, so returned state is already
    slightly stale when you read it."""
    _call({"cmd": "set_velocity", "vx": vx, "vy": vy, "wz": wz})
    if duration_s is not None:
        time.sleep(min(max(duration_s, 0.0), MAX_DRIVE_DURATION_S))
        _call({"cmd": "set_velocity", "vx": 0.0, "vy": 0.0, "wz": 0.0})
        return DuckState(**_state_after(0.1))
    return DuckState(**_state_after())


@mcp.tool(title="Stop", annotations=_INTENT)
def duck_stop() -> DuckState:
    """Zero all velocity intents — the duck stops walking and stands."""
    _call({"cmd": "set_velocity", "vx": 0.0, "vy": 0.0, "wz": 0.0})
    return DuckState(**_state_after())


@mcp.tool(title="Do a trick", annotations=_EPISODIC)
def duck_trick(name: str, stage_ball: bool = True) -> DuckState:
    """Trigger a trick: 'sit', 'stand', 'ground_pick' (touch beak to floor),
    'kick_left'/'kick_right' (stages the ball at that foot, then kicks), or
    'roulade' (forward roll — NOTE: usually ends with the duck down, since no
    stand-up policy ships yet; follow with duck_reset). Episodic tricks hand
    control back to standing automatically after a few seconds.

    stage_ball=False makes kicks honest: the ball is NOT teleported to the
    foot, so the kick connects only if you've already walked the ball into
    position — check ball_offset_m in duck_state and aim for forward≈0.09,
    left≈+0.042 (kick_left) or -0.042 (kick_right) before triggering."""
    resp = _call({"cmd": "trick", "name": name, "stage_ball": stage_ball})
    if not resp.get("started", True):
        raise ToolError(resp.get("error", f"{name} refused"))
    return DuckState(**_state_after())


@mcp.tool(title="Point the head", annotations=_INTENT)
def duck_look(neck_pitch: float = 0.0, head_pitch: float = 0.0,
              head_yaw: float = 0.0, head_roll: float = 0.0) -> DuckState:
    """Point the head (radians; caps ~±1.4 yaw, ±1.1 pitch, ±0.31 roll). This is
    a command to the balance policy, not a servo write — the body compensates.
    All zeros returns the head to neutral. The gaze intent is sticky — it
    holds between tool calls until changed."""
    _call({"cmd": "look", "neck_pitch": neck_pitch, "head_pitch": head_pitch,
           "head_yaw": head_yaw, "head_roll": head_roll})
    return DuckState(**_state_after())


@mcp.tool(title="Open the beak", annotations=_INTENT)
def duck_mouth(opening: float) -> DuckState:
    """Set the beak opening: 0.0 closed to 1.0 open (clamped). Mirrors the
    real robot's `robot.mouth` verb — a continuous, sticky intent, purely
    expressive (no physics). duck_say drives it automatically while speaking;
    use this directly for a held expression (gape, pant, grin)."""
    _call({"cmd": "mouth", "opening": opening})
    return DuckState(**_state_after(0.1))


@mcp.tool(title="Speak", annotations=_EPISODIC)
def duck_say(text: str, voice_bank: str | None = None) -> DuckState:
    """Speak as the duck: text is rendered into the duck's voice (pitched,
    personality-modulated, chirp grains blended into the stressed syllables),
    played on the host's speakers, while the beak lip-syncs live in the sim
    from the same loudness envelope. Blocks until the utterance finishes
    (~1 s per 12 chars; max 400 chars — keep it punchy, it's a duck).

    voice_bank: directory of voice-bank wavs rendered by the microduck
    `sounds` crate (chirp*.wav is blended in); without it the voice still
    works, just chirpless. Requires macOS `say`, `ffmpeg` and `afplay` on the
    machine running this MCP server."""
    from . import voice
    try:
        ffmpeg = voice.find_ffmpeg()
        wav, traj, duration = voice.render_voice(text, ffmpeg,
                                                 bank_dir=voice_bank)
        try:
            voice.speak(wav, traj, text, duration)
        finally:
            if os.path.exists(wav):
                os.unlink(wav)
    except (voice.FilmError, voice.VoiceError) as e:
        raise ToolError(str(e)) from e
    except subprocess.CalledProcessError as e:
        raise ToolError(f"voice render failed: {e}") from e
    return DuckState(**_call({"cmd": "state"}))


@mcp.tool(title="Duck camera", annotations=_READ)
def duck_camera(view: str = "follow", distance: float = 0.7) -> Image:
    """Render a camera frame of the sim. Views: 'head' (the duck's POV, from
    its own head camera — what duck_state's ball_seen is computed from; pans
    with duck_look), 'follow' (behind the duck), 'front' (facing it), 'side',
    'top'. `distance` in meters (0.4 close-up to ~1.5 wide) for the external
    views; 'head' ignores it. Pair with duck_state for pose numbers."""
    resp = _call({"cmd": "camera", "view": view, "distance": distance})
    return Image(path=resp["frame"])


@mcp.tool(title="Shove the duck", annotations=_EPISODIC)
def duck_push(magnitude: float = 1.0, angle_deg: float | None = None) -> DuckState:
    """Shove the duck: sets trunk velocity to `magnitude` m/s (max 2.0) in a
    world-frame direction (random if angle_deg omitted). Tests push recovery."""
    req: dict[str, Any] = {"cmd": "push", "magnitude": magnitude}
    if angle_deg is not None:
        req["angle_deg"] = angle_deg
    _call(req)
    return DuckState(**_state_after(0.6))


class SeqStep(BaseModel):
    """One step of a duck_sequence: issue a command, then hold for `seconds`."""

    do: Literal["drive", "stop", "trick", "look"]
    seconds: float = Field(default=0.0, ge=0.0, le=10.0,
                           description="How long to hold this step before the next "
                           "(drive keeps walking, trick keeps playing out)")
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0
    name: str | None = Field(default=None, description="Trick name (do='trick')")
    stage_ball: bool = True
    neck_pitch: float = 0.0
    head_pitch: float = 0.0
    head_yaw: float = 0.0
    head_roll: float = 0.0


class SequenceResult(BaseModel):
    steps_run: int
    aborted: str | None = Field(description="Why the sequence stopped early, else null")
    state: DuckState


MAX_SEQ_SECONDS = 30.0


@mcp.tool(title="Run a command sequence", annotations=_EPISODIC)
def duck_sequence(steps: list[SeqStep]) -> SequenceResult:
    """Run steps back-to-back with no client round-trips between them — motion
    flows continuously instead of stop-and-settle after every call. Each step
    issues its command, then holds `seconds` before the next: drive steps keep
    walking through the transition (chain them for arcs and S-curves), 'stop'
    zeroes velocity (give it ~0.5s before a trick), 'trick' waits out its hold
    (kicks need ~3.2s before the next trick is accepted), 'look' aims the head.

    The sequence is OPEN LOOP from the state you planned it on — keep chains
    short (a few seconds), then re-check duck_state and trim. Aborts early if
    the duck falls or a trick is refused; velocity is always zeroed at the end.
    Max 20 steps / 30 s total. Example — walk an arc, stop, honest kick, then
    a roulade celebration:
      [{do: drive, vx: 0.3, wz: 0.8, seconds: 2}, {do: stop, seconds: 0.5},
       {do: trick, name: kick_right, stage_ball: false, seconds: 3.2},
       {do: trick, name: roulade, seconds: 2.5}]"""
    if len(steps) > 20:
        raise ToolError("Too many steps (max 20)")
    if sum(s.seconds for s in steps) > MAX_SEQ_SECONDS:
        raise ToolError(f"Total hold time exceeds {MAX_SEQ_SECONDS:.0f}s")
    aborted = None
    n = 0
    try:
        for s in steps:
            if s.do == "drive":
                _call({"cmd": "set_velocity", "vx": s.vx, "vy": s.vy, "wz": s.wz})
            elif s.do == "stop":
                _call({"cmd": "set_velocity", "vx": 0.0, "vy": 0.0, "wz": 0.0})
            elif s.do == "trick":
                if not s.name:
                    raise ToolError("trick step needs a name")
                resp = _call({"cmd": "trick", "name": s.name,
                              "stage_ball": s.stage_ball})
                if not resp.get("started", True):
                    aborted = f"step {n}: {s.name} refused (busy)"
                    break
            elif s.do == "look":
                _call({"cmd": "look", "neck_pitch": s.neck_pitch,
                       "head_pitch": s.head_pitch, "head_yaw": s.head_yaw,
                       "head_roll": s.head_roll})
            n += 1
            if s.seconds:
                time.sleep(s.seconds)
            if not _call({"cmd": "state"}).get("upright", True):
                aborted = f"step {n - 1}: duck fell"
                break
    finally:
        _call({"cmd": "set_velocity", "vx": 0.0, "vy": 0.0, "wz": 0.0})
    return SequenceResult(steps_run=n, aborted=aborted,
                          state=DuckState(**_state_after(0.2)))


class MachineStatus(BaseModel):
    ok: bool
    name: str | None = None
    source: str | None = Field(default=None, description="Machine source file "
                               "(edit it, then action='reload' to hot-swap)")
    armed: bool | None = None
    node: str | None = Field(default=None, description="Current node")
    nodes: list[str] | None = None
    wake_nodes: list[str] | None = Field(default=None, description="Nodes that "
                                         "park a wake pack on entry")
    say_nodes: list[str] | None = Field(default=None, description="Nodes that "
                                        "speak a line on entry (say = \"...\" "
                                        "in the machine source)")
    note: str | None = None
    wake: dict[str, Any] | None = Field(default=None, description="The wake "
        "pack (action='wait' or a blocking arm/force): reason, node, the "
        "transition that fired (via), a digest snapshot, the recent event "
        "tail. `resolved` is non-null if the machine's deadline default "
        "already ran before anyone listened — the body answered itself.")
    no_wake: bool | None = Field(default=None, description="True: block_s "
        "elapsed with nothing to report — wait again to keep listening")
    waited_s: float | None = None


@mcp.tool(title="Behavior machine", annotations=_EPISODIC)
def duck_machine(action: str, path: str | None = None,
                 node: str | None = None,
                 block_s: float | None = None) -> MachineStatus:
    """Drive the duck's behavior machine — autonomy between your decisions.

    A machine is TOML source (see machines/soccer.toml): nodes bind behaviors
    (search_ball, approach_ball, kick, celebrate, idle) executed at 50 Hz on
    the sim thread, with transitions guarded by expressions over the SENSED
    digest only — ball_seen.* and goal_seen.* (camera-derived), upright,
    active_policy, elapsed_s, plus the referee's goal.scored/goal.count.
    Ground-truth ball position is not in the guard vocabulary: an armed
    machine plays fair by construction. machines/striker.toml is the aiming
    variant: approach_ball with aim=true walks onto the ball->goal line of
    fire before kicking, steered by goal_seen's dead-reckoned bearing.

    Actions: 'load' (path, validates + loads disarmed), 'arm' (start at the
    initial node), 'disarm' (stop, zero velocity), 'reload' (hot-swap edited
    source; keeps armed state), 'force' (node, jump now), 'status', and
    'wait' — BLOCK until the machine wakes you. A node declaring
    wake = "reason" parks a wake pack on entry (reason + digest snapshot +
    event tail); 'wait' returns the oldest pack, or no_wake=true after
    block_s (default 55 s — wait again to keep listening; the machine keeps
    playing either way, and its own deadline transition answers a wake you
    slept through, reported in the pack's `resolved`). Pass block_s with
    'arm' or 'force' to arm-and-listen in one blocking call — the residency
    loop: arm(block_s) -> act on the wake (force/reload/say) -> wait again.
    machines/resident.toml is the idle-life machine built around this.
    While armed the machine owns the velocity intent — duck_drive still works
    but the machine will override it on its next tick. Transitions appear in
    the event feed tagged 'machine', wakes as 'wake'."""
    req: dict[str, Any] = {"cmd": "machine", "action": action}
    if path is not None:
        req["path"] = path
    if node is not None:
        req["node"] = node
    timeout = 12.0
    if action == "wait" and block_s is None:
        block_s = 55.0
    if block_s is not None and action in ("wait", "arm", "force"):
        req["block_s"] = block_s
        timeout = block_s + 15.0
    return MachineStatus(**_call(req, timeout=timeout))


@mcp.tool(title="Reset the sim", annotations=_RESET)
def duck_reset() -> DuckState:
    """Reset the sim: duck back to the origin in its default standing pose,
    ball back to its spawn. Discards the current episode — the escape hatch
    after a fall (there is no stand-up policy yet)."""
    _call({"cmd": "reset"})
    return DuckState(**_state_after())


def main():
    mcp.run()


if __name__ == "__main__":
    main()
