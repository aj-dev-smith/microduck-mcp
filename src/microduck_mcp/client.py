"""Client for the Microduck sim server's Unix-socket control plane.

Library (`request()`) plus a small CLI:

    duck state
    duck drive 0.15 [vy] [wz]
    duck stop
    duck trick roulade
    duck look 0 0.3 0 0
    duck cam follow
    duck push [magnitude] [angle_deg]
    duck reset
"""

import argparse
import json
import os
import socket
import sys
import tempfile

DEFAULT_SOCKET = os.environ.get(
    "DUCK_SIM_SOCKET", os.path.join(tempfile.gettempdir(), "microduck-sim.sock"))


def request(req: dict, sock_path: str = None, timeout: float = 12.0) -> dict:
    sock_path = sock_path or DEFAULT_SOCKET
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        s.connect(sock_path)
        s.sendall((json.dumps(req) + "\n").encode())
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
    return json.loads(buf.decode())


def main():
    p = argparse.ArgumentParser(description="Microduck sim client")
    p.add_argument("--socket", default=None)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("ping")
    sub.add_parser("state")
    d = sub.add_parser("drive")
    d.add_argument("vx", type=float)
    d.add_argument("vy", type=float, nargs="?", default=0.0)
    d.add_argument("wz", type=float, nargs="?", default=0.0)
    sub.add_parser("stop")
    t = sub.add_parser("trick")
    t.add_argument("name", choices=["sit", "stand", "ground_pick", "kick_left", "kick_right", "roulade"])
    lk = sub.add_parser("look")
    for a in ("neck_pitch", "head_pitch", "head_yaw", "head_roll"):
        lk.add_argument(a, type=float, nargs="?", default=0.0)
    c = sub.add_parser("cam")
    c.add_argument("view", nargs="?", default="follow",
                   choices=["follow", "front", "side", "top", "head"])
    c.add_argument("--distance", type=float, default=0.7)
    ps = sub.add_parser("push")
    ps.add_argument("magnitude", type=float, nargs="?", default=1.0)
    ps.add_argument("angle_deg", type=float, nargs="?", default=None)
    sub.add_parser("reset")
    args = p.parse_args()

    if args.command == "drive":
        req = {"cmd": "set_velocity", "vx": args.vx, "vy": args.vy, "wz": args.wz}
    elif args.command == "stop":
        req = {"cmd": "set_velocity", "vx": 0.0, "vy": 0.0, "wz": 0.0}
    elif args.command == "trick":
        req = {"cmd": "trick", "name": args.name}
    elif args.command == "look":
        req = {"cmd": "look", "neck_pitch": args.neck_pitch, "head_pitch": args.head_pitch,
               "head_yaw": args.head_yaw, "head_roll": args.head_roll}
    elif args.command == "cam":
        req = {"cmd": "camera", "view": args.view, "distance": args.distance}
    elif args.command == "push":
        req = {"cmd": "push", "magnitude": args.magnitude}
        if args.angle_deg is not None:
            req["angle_deg"] = args.angle_deg
    else:
        req = {"cmd": args.command}

    try:
        resp = request(req, sock_path=args.socket)
    except (ConnectionRefusedError, FileNotFoundError):
        print(json.dumps({"ok": False, "error": f"sim server not running (socket {args.socket or DEFAULT_SOCKET})"}))
        sys.exit(1)
    print(json.dumps(resp, indent=2))
    sys.exit(0 if resp.get("ok") else 1)


if __name__ == "__main__":
    main()
