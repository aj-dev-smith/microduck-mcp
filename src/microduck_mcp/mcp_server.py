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

import time
from typing import Any

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
    ball_position_m: list[float] | None = Field(
        default=None, description="Ball world position [x, y, z], meters (ball scene only)")


def _call(req: dict, retries_note: str = "") -> dict[str, Any]:
    try:
        resp = request({**req, "client": "mcp"})
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
    vy: leftward m/s (max ±0.2), wz: counterclockwise yaw rate rad/s (max ±1.5).
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
def duck_trick(name: str) -> DuckState:
    """Trigger a trick: 'sit', 'stand', 'ground_pick' (touch beak to floor),
    'kick_left'/'kick_right' (stages the ball at that foot, then kicks), or
    'roulade' (forward roll — NOTE: usually ends with the duck down, since no
    stand-up policy ships yet; follow with duck_reset). Episodic tricks hand
    control back to standing automatically after a few seconds."""
    resp = _call({"cmd": "trick", "name": name})
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


@mcp.tool(title="Duck camera", annotations=_READ)
def duck_camera(view: str = "follow", distance: float = 0.7) -> Image:
    """Render a camera frame of the sim. Views: 'follow' (behind the duck),
    'front' (facing it), 'side', 'top'. `distance` in meters (0.4 close-up
    to ~1.5 wide). Pair with duck_state for pose numbers."""
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
