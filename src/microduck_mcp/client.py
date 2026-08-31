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
    duck pet state                           # the desktop overlay's own pose
    duck pet config --px-per-meter 656       # ...and the screen it thinks it is on
    duck pet world                           # park every ledge (the undo)
    duck say "hello A J" --voice-bank bank/
    duck say "we lost the ball" --mood sad   # same duck, different weather
    duck chirp inquire --voice-bank bank/    # nonverbal: one call from the bank
    duck emote head_tilt | duck emote --list # authored gestures (see emotes/)
    duck train tasks                         # what the GPU box can train
    duck train start <TASK_ID> --smoke       # 64 envs / 5 iters, always first
    duck train start <TASK_ID> --num-envs 4096
    duck train status [session] | duck train stop <session>

Every subcommand but `film`, `say`, `chirp` and `train` is a one-shot intent
over the socket; `film` runs its own headless sim and writes an mp4 (see
film.py), `say`/`chirp` play the duck's voice host-side while streaming the
beak to the running sim (see voice.py), and `train` drives RL runs on the GPU
box over ssh (see train.py) — it never touches the sim at all.
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
    # The desktop overlay's half of the socket, so `duck-pet` is inspectable
    # the way `duck machine` is: what does the daemon think the screen looks
    # like, and where does it think the duck is standing on it. `frame` costs
    # a render like any other frame request and hands back the PNG's size
    # rather than its bytes — this is a diagnostic, not a second viewer.
    pet = sub.add_parser("pet", help="the desktop pet's view of the world")
    # `world` with no --rects is the escape hatch: it parks every platform
    # box. `reset` deliberately leaves the pet's world alone (it belongs to
    # the screen, not to the run), so without this a badly-placed ledge could
    # only be undone over HTTP.
    pet.add_argument("action", choices=["state", "config", "frame", "world"])
    pet.add_argument("--px-per-meter", type=float, default=None)
    pet.add_argument("--screen-width-px", type=float, default=None)
    pet.add_argument("--frame-px", type=int, default=None)
    pet.add_argument("--supersample", type=int, default=None)
    pet.add_argument("--wall-margin-m", type=float, default=None)
    pet.add_argument("--corridor-m", type=float, default=None)
    pet.add_argument("--floor-pad-px", type=int, default=None)
    pet.add_argument("--size-px", type=int, default=None,
                     help="frame: render at this size instead of the configured one")
    pet.add_argument("--rects", default=None,
                     help="world: a JSON list of {x, y, w, h} ledges in metres "
                          "of screen floor (default: none, which parks them "
                          "all — the way to undo a bad placement)")
    em = sub.add_parser("emote", help="play an authored gesture")
    em.add_argument("name", nargs="?", default=None,
                    help="emote name — the file stem in the server's emote "
                    "directory (head_tilt, nod, perk_up, droop)")
    em.add_argument("--list", action="store_true", dest="list_emotes",
                    help="list the server's emotes and whether they parse")
    # `film`, `say`, `chirp` and `train` are not plain socket intents — film
    # boots its own headless sim, the two voices play audio host-side and then
    # perform it, and train drives the GPU box over ssh — so their flags live
    # next to their implementations.
    from . import film, train, voice
    film.add_arguments(sub.add_parser("film",
                                      help="film an autonomous match to an mp4"))
    voice.add_arguments(sub.add_parser("say",
                                       help="speak: audio host-side, beak in sim"))
    voice.add_chirp_arguments(sub.add_parser(
        "chirp", help="play one call from the duck's voice bank"))
    train.add_arguments(sub.add_parser(
        "train", help="launch and read RL runs on the GPU box"))
    args = p.parse_args()

    if args.command == "film":
        sys.exit(film.run(args))
    if args.command == "say":
        sys.exit(voice.run(args))
    if args.command == "chirp":
        sys.exit(voice.run_chirp(args))
    if args.command == "train":
        sys.exit(train.run(args))

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
    elif args.command == "emote":
        # No name is the same question as --list: "what can it do?"
        req = ({"cmd": "emote", "name": args.name}
               if args.name and not args.list_emotes
               else {"cmd": "emote", "action": "list"})
    elif args.command == "pet":
        req = {"cmd": f"pet_{args.action}"}
        if args.action == "frame" and args.size_px is not None:
            req["size_px"] = args.size_px
        if args.action == "world":
            try:
                req["rects"] = json.loads(args.rects) if args.rects else []
            except ValueError as e:
                print(json.dumps({"ok": False,
                                  "error": f"--rects is not JSON: {e}"}))
                sys.exit(1)
        if args.action == "config":
            # Only the flags actually given: every key the daemon does not
            # hear about keeps the value it has, so `duck pet config` on its
            # own is a read, not a reset to argparse's idea of a screen.
            for flag in ("px_per_meter", "screen_width_px", "frame_px",
                         "supersample", "wall_margin_m", "corridor_m",
                         "floor_pad_px"):
                val = getattr(args, flag, None)
                if val is not None:
                    req[flag] = val
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
