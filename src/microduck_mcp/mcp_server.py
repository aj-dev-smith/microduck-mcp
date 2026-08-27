"""MCP server exposing the Microduck sim as tools.

Stdio MCP server; each tool forwards an intent to the sim server's Unix
socket (start it first: `duck-sim`). Register with any MCP client, e.g.:

    claude mcp add duck -- uv --directory /path/to/microduck-mcp run duck-mcp
"""

import json

from mcp.server.mcpserver import Image, MCPServer

from .client import request

mcp = MCPServer("microduck")


def _call(req: dict) -> str:
    try:
        return json.dumps(request(req))
    except (ConnectionRefusedError, FileNotFoundError):
        return json.dumps({"ok": False, "error": "sim server not running — start it with `duck-sim`"})


@mcp.tool()
def duck_state() -> str:
    """Current robot state: position, orientation, body-frame velocity, active
    policy, whether it is upright/sitting/mid-trick, and the ball position."""
    return _call({"cmd": "state"})


@mcp.tool()
def duck_drive(vx: float, vy: float = 0.0, wz: float = 0.0) -> str:
    """Set the walking velocity command and keep it until changed.
    vx: forward m/s (max ±0.3), vy: leftward m/s (max ±0.2),
    wz: counterclockwise yaw rate rad/s (max ±1.5). Nonzero engages the
    walking policy; all-zero hands back to the standing policy."""
    return _call({"cmd": "set_velocity", "vx": vx, "vy": vy, "wz": wz})


@mcp.tool()
def duck_stop() -> str:
    """Zero all velocity commands — the duck stops and stands."""
    return _call({"cmd": "set_velocity", "vx": 0.0, "vy": 0.0, "wz": 0.0})


@mcp.tool()
def duck_trick(name: str) -> str:
    """Trigger a trick: 'sit', 'stand', 'ground_pick' (touch beak to floor),
    'kick_left'/'kick_right' (kick the ball), or 'roulade' (forward roll).
    Episodic tricks hand control back to standing automatically."""
    return _call({"cmd": "trick", "name": name})


@mcp.tool()
def duck_look(neck_pitch: float = 0.0, head_pitch: float = 0.0,
              head_yaw: float = 0.0, head_roll: float = 0.0) -> str:
    """Point the head (radians, roughly ±1.4 yaw, ±1.1 pitch, ±0.31 roll).
    All zeros returns the head to neutral."""
    return _call({"cmd": "look", "neck_pitch": neck_pitch, "head_pitch": head_pitch,
                  "head_yaw": head_yaw, "head_roll": head_roll})


@mcp.tool()
def duck_camera(view: str = "follow", distance: float = 0.7) -> Image:
    """Render a camera frame of the sim. Views: 'follow' (behind the duck),
    'front' (facing it), 'side', 'top'. Returns the image."""
    resp = request({"cmd": "camera", "view": view, "distance": distance})
    if not resp.get("ok"):
        raise RuntimeError(resp.get("error", "camera failed"))
    return Image(path=resp["frame"])


@mcp.tool()
def duck_push(magnitude: float = 1.0, angle_deg: float | None = None) -> str:
    """Shove the duck: set trunk velocity to `magnitude` m/s (max 2.0) in a
    world-frame direction (random if angle_deg omitted). Tests recovery."""
    req = {"cmd": "push", "magnitude": magnitude}
    if angle_deg is not None:
        req["angle_deg"] = angle_deg
    return _call(req)


@mcp.tool()
def duck_reset() -> str:
    """Reset the sim: duck back to the origin in its default stance."""
    return _call({"cmd": "reset"})


def main():
    mcp.run()


if __name__ == "__main__":
    main()
