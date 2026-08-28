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
    duck machine load machines/soccer.toml
    duck machine arm | disarm | status | reload | force <node>
    duck machine wait [--block-s 300]     # block until a wake node fires
    duck machine arm --block-s 300        # arm, then block for the first wake
    duck film -o match.mp4
    duck mouth 0.6
    duck say "hello A J" --voice-bank bank/
    duck chirp inquire --voice-bank bank/    # nonverbal: one call from the bank

Every subcommand but `film`, `say` and `chirp` is a one-shot intent over the
socket; `film` runs its own headless sim and writes an mp4 (see film.py), and
`say`/`chirp` play the duck's voice host-side while streaming the beak to the
running sim (see voice.py).
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
    m = sub.add_parser("machine")
    m.add_argument("action", choices=["load", "reload", "status", "arm",
                                      "disarm", "force", "wait"])
    m.add_argument("arg", nargs="?", default=None,
                   help="path for load, node name for force")
    m.add_argument("--block-s", type=float, default=None,
                   help="wait: how long to block for a wake (default 55; "
                   "returns no_wake and you call again). arm/force: block "
                   "for the first wake after the jump.")
    mo = sub.add_parser("mouth", help="set the beak opening")
    mo.add_argument("opening", type=float, help="0 (closed) to 1 (open)")
    # `film`, `say` and `chirp` are not plain socket intents — film boots its
    # own headless sim, the two voices play audio host-side and then perform
    # it — so their flags live next to their implementations.
    from . import film, voice
    film.add_arguments(sub.add_parser("film",
                                      help="film an autonomous match to an mp4"))
    voice.add_arguments(sub.add_parser("say",
                                       help="speak: audio host-side, beak in sim"))
    voice.add_chirp_arguments(sub.add_parser(
        "chirp", help="play one call from the duck's voice bank"))
    args = p.parse_args()

    if args.command == "film":
        sys.exit(film.run(args))
    if args.command == "say":
        sys.exit(voice.run(args))
    if args.command == "chirp":
        sys.exit(voice.run_chirp(args))

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
    elif args.command == "mouth":
        req = {"cmd": "mouth", "opening": args.opening}
    elif args.command == "machine":
        req = {"cmd": "machine", "action": args.action}
        if args.action == "load":
            req["path"] = os.path.abspath(args.arg or "")
        elif args.action == "force":
            req["node"] = args.arg
        if args.action == "wait" and args.block_s is None:
            args.block_s = 55.0
        if args.block_s is not None and args.action in ("wait", "arm", "force"):
            req["block_s"] = args.block_s
    else:
        req = {"cmd": args.command}

    # A blocking machine wait must outlive its block window on the socket.
    timeout = req.get("block_s", 0.0) + 15.0 if "block_s" in req else 12.0
    try:
        resp = request(req, sock_path=args.socket, timeout=timeout)
    except (ConnectionRefusedError, FileNotFoundError):
        print(json.dumps({"ok": False, "error": f"sim server not running (socket {args.socket or DEFAULT_SOCKET})"}))
        sys.exit(1)
    print(json.dumps(resp, indent=2))
    sys.exit(0 if resp.get("ok") else 1)


if __name__ == "__main__":
    main()
