"""`duck pet up` / `down` / `status`: the desktop pet as a managed pair.

The pet is two processes — a desktop-scene sim daemon and the overlay window
— plus a machine load and an arm, and until now the recipe lived in a note:
three commands in the right order, run from the right directory, and the
whole thing died with the terminal that typed them. This module turns the
recipe into one verb each way:

    duck pet up          # daemon + machine + overlay, detached from the tty
    duck pet down        # kill both halves, sweep strays, remove the socket
    duck pet status      # are they alive, and what node is the duck in

Both children are spawned in their own session (`start_new_session`), so
closing the terminal no longer takes the duck with it — that was the whole
complaint. State lives in ~/.microduck/pet/: a pidfile per half, a log per
half. `down` kills what the pidfiles name, then sweeps the web port and the
socket for strays (tonight's ghost-trail bug was exactly such a stray — an
overlay from an earlier launch that a pattern-based pkill missed), refusing
to touch any process whose command line does not look like ours: the
resident sim on the default socket belongs to someone else and must never
be collateral.

The pet deliberately defaults to its OWN socket (/tmp/duck-pet.sock) and
never to DUCK_SIM_SOCKET — `duck pet up` on a machine with a resident duck
must stand up a second duck, not seize the first one's control plane.
"""

import argparse
import errno
import json
import os
import signal
import subprocess
import sys
import time

from . import client

PET_SOCKET = "/tmp/duck-pet.sock"
PET_PORT = 8410
STATE_DIR = os.path.expanduser("~/.microduck/pet")
# What a sweep is allowed to kill: the guard between "clean up my strays"
# and "shoot whatever squats on my port". Matched against the full command
# line of the process holding the port or socket.
OURS = ("microduck_mcp", "duck-sim", "duck-pet")
# The daemon compiles a scene and loads ONNX before it answers; a cold
# machine with a slow disk has been seen taking most of a minute.
BOOT_TIMEOUT_S = 90.0


def _pidfile(name):
    return os.path.join(STATE_DIR, f"{name}.pid")


def _logfile(name):
    return os.path.join(STATE_DIR, f"{name}.log")


def _read_pid(name):
    """The pid the last `up` wrote, or None — garbage reads as absent."""
    try:
        with open(_pidfile(name)) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def _alive(pid):
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError as e:
        # EPERM means it exists but is not ours to signal — for liveness
        # that is still "alive", and for killing it is "leave it alone".
        return e.errno == errno.EPERM
    return True


def _cmdline(pid):
    try:
        out = subprocess.run(["ps", "-p", str(pid), "-o", "command="],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _is_ours(cmdline):
    return any(tag in cmdline for tag in OURS)


def _port_listeners(port):
    try:
        out = subprocess.run(
            ["lsof", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"],
            capture_output=True, text=True, timeout=10)
        return [int(x) for x in out.stdout.split()]
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return []


def _port_users(port):
    """Everyone on the port — the overlay is a CLIENT of the feed, not a
    listener, so a listener-only sweep leaves stray overlays standing
    (exactly the frozen-ghost failure `down` exists to clear)."""
    try:
        out = subprocess.run(["lsof", "-t", f"-iTCP:{port}"],
                             capture_output=True, text=True, timeout=10)
        return [int(x) for x in out.stdout.split()]
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return []


def _overlay_pids():
    """Stray overlays by name. Safe as a pattern: only the overlay's two
    names — never sim_server, which on the default socket is somebody
    else's resident duck."""
    pids = set()
    for pattern in ("microduck_mcp.pet_app", "duck-pet"):
        try:
            out = subprocess.run(["pgrep", "-f", pattern],
                                 capture_output=True, text=True, timeout=10)
            pids.update(int(x) for x in out.stdout.split())
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
    pids.discard(os.getpid())
    return sorted(pids)


def _socket_owner(sock_path):
    try:
        out = subprocess.run(["lsof", "-t", "-U", "-a", sock_path],
                             capture_output=True, text=True, timeout=10)
        pids = [int(x) for x in out.stdout.split()]
        return pids[0] if pids else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def _ping(sock_path, timeout=3.0):
    try:
        return client.request({"cmd": "ping"}, sock_path=sock_path,
                              timeout=timeout).get("ok", False)
    except (OSError, ValueError):
        return False


def _resolve(path, must_exist=True):
    """Resolve a possibly-relative path against the cwd, then the checkout.

    The documented workspace layout puts microduck_rl and the policies one
    directory above the repo, so the defaults are ../-relative — which only
    works when `duck pet up` is typed from microduck-mcp/. Falling back to
    the checkout root (two levels above this file) makes the same defaults
    work from anywhere, without hardcoding anyone's home directory.
    """
    if os.path.isabs(path):
        return path
    cwd_try = os.path.abspath(path)
    if os.path.exists(cwd_try) or not must_exist:
        return cwd_try
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    repo_try = os.path.abspath(os.path.join(repo_root, path))
    if os.path.exists(repo_try):
        return repo_try
    return cwd_try


def _spawn(name, argv):
    """Start one detached child, log to its file, remember its pid."""
    os.makedirs(STATE_DIR, exist_ok=True)
    log = open(_logfile(name), "ab", buffering=0)
    log.write(f"\n--- duck pet up @ {time.strftime('%Y-%m-%d %H:%M:%S')} "
              f"--- {' '.join(argv)}\n".encode())
    proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log,
                            stderr=log, start_new_session=True)
    log.close()
    with open(_pidfile(name), "w") as f:
        f.write(str(proc.pid))
    return proc.pid


def _kill(pid, label, killed):
    """TERM, a grace window, then KILL. Appends what happened to `killed`."""
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if not _alive(pid):
            killed.append({"pid": pid, "what": label, "how": "term"})
            return
        time.sleep(0.1)
    try:
        os.kill(pid, signal.SIGKILL)
        killed.append({"pid": pid, "what": label, "how": "kill"})
    except OSError:
        killed.append({"pid": pid, "what": label, "how": "term"})


def _emit(payload):
    print(json.dumps(payload, indent=2))
    return 0 if payload.get("ok") else 1


def up(args):
    sock = args.socket or PET_SOCKET
    port = args.port

    daemon_pid = _read_pid("daemon")
    if _alive(daemon_pid) or _ping(sock):
        return _emit({"ok": False, "error": "pet already up",
                      "hint": "duck pet down first, or duck pet status"})
    strangers = [p for p in _port_listeners(port) if p != os.getpid()]
    if strangers:
        return _emit({"ok": False,
                      "error": f"port {port} is already taken "
                               f"(pids {strangers})",
                      "hint": "duck pet down sweeps our strays; anything "
                              "else is yours to move"})

    rl_repo = _resolve(args.rl_repo)
    policies = _resolve(args.policies)
    machine = _resolve(args.machine)
    missing = {k: v for k, v in (("rl-repo", rl_repo),
                                 ("policies", policies),
                                 ("machine", machine))
               if not os.path.exists(v)}
    if missing:
        return _emit({"ok": False, "error": "paths not found",
                      "missing": missing,
                      "hint": "run from microduck-mcp/ or pass the flags"})

    daemon_argv = [sys.executable, "-m", "microduck_mcp.sim_server",
                   "--scene", "desktop", "--web", str(port),
                   "--socket", sock, "--rl-repo", rl_repo,
                   "--policies", policies]
    voice_bank = _resolve(args.voice_bank) if args.voice_bank else None
    if voice_bank and os.path.isdir(voice_bank):
        daemon_argv += ["--voice-bank", voice_bank]
    daemon_pid = _spawn("daemon", daemon_argv)

    deadline = time.monotonic() + BOOT_TIMEOUT_S
    while time.monotonic() < deadline:
        if _ping(sock):
            break
        if not _alive(daemon_pid):
            return _emit({"ok": False, "error": "daemon died during boot",
                          "log": _logfile("daemon")})
        time.sleep(0.5)
    else:
        return _emit({"ok": False,
                      "error": f"daemon not answering after "
                               f"{BOOT_TIMEOUT_S:.0f}s",
                      "log": _logfile("daemon")})

    for req in ({"cmd": "machine", "action": "load", "path": machine},
                {"cmd": "machine", "action": "arm"}):
        resp = client.request(req, sock_path=sock, timeout=15.0)
        if not resp.get("ok"):
            return _emit({"ok": False, "error": "machine setup failed",
                          "request": req, "response": resp})

    overlay_pid = _spawn("overlay", [sys.executable, "-m",
                                     "microduck_mcp.pet_app",
                                     "--port", str(port)])
    # The overlay fails fast when it does (no screen, bad port) — a couple
    # of seconds is enough to tell a launch from a crash-on-arrival.
    time.sleep(2.0)
    if not _alive(overlay_pid):
        return _emit({"ok": False, "error": "overlay died on launch",
                      "log": _logfile("overlay"),
                      "daemon": {"pid": daemon_pid, "socket": sock}})

    return _emit({"ok": True, "daemon": {"pid": daemon_pid, "socket": sock,
                                         "web": port},
                  "overlay": {"pid": overlay_pid},
                  "machine": {"path": machine, "armed": True},
                  "logs": STATE_DIR,
                  "note": "detached — survives this terminal; "
                          "duck pet down to stop"})


def down(args):
    sock = args.socket or PET_SOCKET
    killed, spared = [], []

    # The overlay first: killing the daemon under a live overlay leaves it
    # a few seconds of frozen last-frame ghost before it notices.
    for name in ("overlay", "daemon"):
        pid = _read_pid(name)
        if _alive(pid):
            _kill(pid, name, killed)
        try:
            os.unlink(_pidfile(name))
        except OSError:
            pass

    # Sweep the port and the socket for strays — launches that predate the
    # pidfiles, or survived a crash that lost them. Only what looks like
    # ours: a stranger on the port gets reported, never shot.
    done = {k["pid"] for k in killed}
    for pid in _overlay_pids() + _port_users(args.port):
        if pid in done or not _alive(pid):
            continue
        done.add(pid)
        cmd = _cmdline(pid)
        if _is_ours(cmd):
            _kill(pid, "stray", killed)
        else:
            spared.append({"pid": pid, "cmd": cmd[:120]})
    owner = _socket_owner(sock)
    if owner is not None and _alive(owner):
        cmd = _cmdline(owner)
        if _is_ours(cmd):
            _kill(owner, "socket stray", killed)
        else:
            spared.append({"pid": owner, "cmd": cmd[:120]})
            return _emit({"ok": False, "killed": killed, "spared": spared,
                          "error": f"{sock} belongs to a process that is "
                                   "not ours; leaving it (and the socket "
                                   "file) alone"})
    try:
        os.unlink(sock)
    except OSError:
        pass

    return _emit({"ok": True, "killed": killed, "spared": spared,
                  "note": "pet is down" if killed else
                          "nothing was running"})


def status(args):
    sock = args.socket or PET_SOCKET
    halves = {}
    for name in ("overlay", "daemon"):
        pid = _read_pid(name)
        halves[name] = {"pid": pid, "alive": _alive(pid)}
    answering = _ping(sock)
    out = {"ok": answering and halves["overlay"]["alive"],
           "socket": sock, "daemon_answering": answering, **halves}
    if answering:
        try:
            m = client.request({"cmd": "machine", "action": "status"},
                               sock_path=sock, timeout=5.0)
            out["machine"] = {k: m.get(k) for k in ("armed", "node")
                              if k in m}
        except (OSError, ValueError):
            pass
    return _emit(out)


def add_arguments(p: argparse.ArgumentParser):
    p.add_argument("--port", type=int, default=PET_PORT,
                   help="the daemon's web port and the overlay's feed")
    p.add_argument("--rl-repo", default=os.environ.get(
        "MICRODUCK_RL_REPO", "../microduck_rl"))
    p.add_argument("--policies", default=os.environ.get(
        "MICRODUCK_POLICIES", "../microduck/policies"))
    p.add_argument("--voice-bank", default=os.environ.get(
        "DUCK_VOICE_BANK", "../voicebank"),
        help="chirps and coos; skipped quietly if the directory is absent")
    p.add_argument("--machine", default="machines/pet.toml",
                   help="the behavior loaded and armed by `up`")
    return p


def run(args):
    return {"up": up, "down": down, "status": status}[args.action](args)
