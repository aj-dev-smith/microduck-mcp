"""Remote training: launch and read RL runs on the GPU box.

The other half of the north star. `machines/` let the duck act between the
agent's decisions; this lets the agent grow the duck a NEW behavior — author an
env in microduck_rl, train it on the 4090, read the run, export the ONNX the
sim hot-swaps. This module is the train step, and only the train step: it does
not touch the sim, and nothing in here imports MuJoCo.

    duck_train_start   ->  tmux new-session on the box, tee'd to ~/logs/
    duck_train_status  ->  tmux ls + tail + parse (wandb URL, iter, reward, ETA)
    duck_train_stop    ->  SIGINT the trainer, then kill the session
    duck_train_tasks   ->  `uv run list-envs`, cached

## Why the script goes over stdin

A training command is `cd && uv run train ... 2>&1 | tee log`, launched inside
`tmux new-session`, over ssh. Quoting that three times over is how you get a
run that dies on a stray quote twelve hours later. So nothing is quoted for a
remote shell at all: we build a plain shell SCRIPT and feed it to `bash -l -s`
on stdin. ssh never parses it, PowerShell (on the fallback route) never parses
it, and the tests can read the exact text that will run.

Same reasoning one level down: the tmux session runs a `~/logs/<session>.sh`
written by that script, not an argument to `tmux new-session`. The runner
script is also a record — the exact command a run was launched with survives
on the box next to its log.

## Sessions and logs

One tmux session per run, named `duck-train-<slug>`; refusing to start a
second run for a task is the whole point of the naming. The log is
`~/logs/train_<slug>_<timestamp>.log`, so a session's logs are discoverable
from its name alone (no manifest to go stale) and a re-run never clobbers the
previous run's evidence.

## `--video True`, not `--video`

mjlab configures tyro with `FlagConversionOff`, so booleans take a VALUE.
`--video` alone is a parse error — it killed a 12 h StandUp run at second one,
which is exactly the failure mode a tool exists to make impossible.
"""

import os
import re
import shlex
import subprocess
import time
from datetime import datetime

# The box, as a bare ssh alias that lands in Ubuntu. The fallback route goes
# in through Windows PowerShell and hops into WSL; it works because the script
# travels on stdin (PowerShell only ever parses the wsl invocation itself).
SSH_HOST = os.environ.get("DUCK_TRAIN_HOST", "duck-4090-wsl")
VIA_WSL = os.environ.get("DUCK_TRAIN_VIA_WSL", "") not in ("", "0", "false")

REPO = "~/microduck_rl"
UV = "~/.local/bin/uv"
LOG_DIR = "~/logs"
SESSION_PREFIX = "duck-train"

# A smoke test is 64 envs / 5 iterations (microduck_rl's AGENTS.md): it catches
# ~95% of config errors for cents, and no long run should be launched without
# one. Cheap enough that the tool defaults to offering it, not hiding it.
SMOKE_NUM_ENVS = 64
SMOKE_ITERATIONS = 5

DEFAULT_NUM_ENVS = 4096
TAIL_LINES = 200

# ssh, not the trainer, is what we are waiting on: launching is a few seconds,
# `list-envs` imports the whole task registry and is not.
SSH_TIMEOUT_S = 30.0
TASKS_TIMEOUT_S = 180.0

# How long the trainer gets between the SIGINT and the session being killed.
STOP_GRACE_S = 5.0

_ANSI = re.compile(r"\x1b\[[0-9;]*m")

# What a task id is allowed to look like. Registered ids are of the form
# `Mjlab-StandUp-Flat-MicroDuck`, and refusing everything else here — before
# the string is ever written into a shell script — is why nothing downstream
# has to think about quoting.
_TASK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class TrainError(RuntimeError):
    """Something the agent should read and act on — a refusal or a dead box."""


# --------------------------------------------------------------------------
# Names
# --------------------------------------------------------------------------

def check_task_id(task_id: str) -> str:
    if not task_id or not task_id.strip():
        raise TrainError("which task? (duck_train_tasks lists what the box has)")
    task_id = task_id.strip()
    if not _TASK_ID.match(task_id):
        raise TrainError(
            f"{task_id!r} is not a task id — they look like "
            f"`Mjlab-StandUp-Flat-MicroDuck` (letters, digits, . _ -). "
            f"duck_train_tasks lists the real ones.")
    return task_id


def slug_for(task_id: str) -> str:
    """`Mjlab-StandUp-Flat-MicroDuck` -> `standup-flat-microduck`.

    Only the `Mjlab-` prefix comes off — every other word stays, because the
    families differ by one word (`-Backlash-`, `-Rollers`) and a slug that
    collapses two tasks into one session name would let a run silently refuse
    to start next to its own twin.
    """
    s = re.sub(r"(?i)^mjlab[-_]", "", check_task_id(task_id))
    s = re.sub(r"[^0-9A-Za-z]+", "-", s).strip("-").lower()
    if not s:
        raise TrainError(f"{task_id!r} has no name left once punctuation is "
                         f"removed — that is not a task id")
    return s


def session_for(task_id: str) -> str:
    return f"{SESSION_PREFIX}-{slug_for(task_id)}"


def slug_of_session(session: str) -> str | None:
    """The slug a `duck-train-<slug>` session was named for, else None.

    None is the honest answer for the bare `duck-train` session that predates
    this tool: it is a training session, we just cannot tell which task from
    its name.
    """
    if session.startswith(SESSION_PREFIX + "-"):
        return session[len(SESSION_PREFIX) + 1:] or None
    return None


def log_path(task_id: str, when: datetime | None = None) -> str:
    stamp = (when or datetime.now()).strftime("%Y%m%d-%H%M%S")
    return f"{LOG_DIR}/train_{slug_for(task_id)}_{stamp}.log"


def log_globs(session: str) -> list[str]:
    """Where to look for a session's log, newest first when globbed by `ls -t`.

    Derived from the session name rather than remembered, so status works on a
    run this process did not start — including the pre-tool `duck-train`
    session, whose log is whatever `~/train_*.log` is freshest.
    """
    slug = slug_of_session(session)
    if slug:
        return [f"{LOG_DIR}/train_{slug}_*.log"]
    return [f"{LOG_DIR}/train_*.log", "~/train_*.log"]


# --------------------------------------------------------------------------
# Command construction — every remote command is a script, quoted once
# --------------------------------------------------------------------------

def ssh_argv(host: str = None, via_wsl: bool = None) -> list[str]:
    """argv for a login bash on the box that reads its script from stdin."""
    host = SSH_HOST if host is None else host
    via_wsl = VIA_WSL if via_wsl is None else via_wsl
    remote = "wsl -d Ubuntu -- bash -l -s" if via_wsl else "bash -l -s"
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
            host, remote]


def runner_script(task_id: str, logfile: str, num_envs: int = DEFAULT_NUM_ENVS,
                  video: bool = True, iterations: int | None = None,
                  extra_args: str | None = None) -> str:
    """The script tmux runs: one training command, tee'd, with its exit code.

    `set -o pipefail` is load-bearing — without it the recorded status is
    tee's, and every run in history looks like it succeeded.
    """
    task_id = check_task_id(task_id)
    if num_envs < 1:
        raise TrainError(f"num_envs is {num_envs}")
    cmd = [UV, "run", "train", shlex.quote(task_id),
           "--env.scene.num-envs", str(int(num_envs))]
    if iterations is not None:
        cmd += ["--agent.max_iterations", str(int(iterations))]
    if video:
        # A VALUE, not a flag: mjlab runs tyro with FlagConversionOff.
        cmd += ["--video", "True"]
    if extra_args:
        cmd += [shlex.quote(a) for a in shlex.split(extra_args)]
    return (
        "set -o pipefail\n"
        f"cd {REPO} || exit 1\n"
        f"echo \"[duck-train] {task_id} started $(date -Is)\" | tee -a {logfile}\n"
        f"{' '.join(cmd)} 2>&1 | tee -a {logfile}\n"
        f"echo \"[duck-train] exit rc=$? at $(date -Is)\" >> {logfile}\n"
    )


def start_script(session: str, runner_path: str, runner: str) -> str:
    """Refuse-or-launch, decided on the box so the check cannot go stale.

    The refusal is a sentence and a non-zero exit: a session already running
    for this task is a fact about the world, not a failure of the caller, and
    it must never be resolved by replacing the run that is already in there.
    """
    # No `set -e`: under it, an `exit 3` from inside an `if` reports 1 to ssh,
    # and a refusal that comes back as a generic failure is a refusal nobody
    # can branch on. Failures are handled where they happen instead.
    return (
        f"mkdir -p {LOG_DIR} || exit 5\n"
        f"if tmux has-session -t {shlex.quote(session)} 2>/dev/null; then\n"
        f"  echo \"REFUSED session {session} already exists\"\n"
        f"  exit 3\n"
        f"fi\n"
        f"cat > {runner_path} <<'DUCK_RUNNER_EOF'\n"
        f"{runner}"
        f"DUCK_RUNNER_EOF\n"
        f"tmux new-session -d -s {shlex.quote(session)} "
        f"bash -l {runner_path} || exit 6\n"
        f"sleep 1\n"
        f"echo \"INIT=$(ps -p 1 -o comm= 2>/dev/null)\"\n"
        f"tmux has-session -t {shlex.quote(session)} 2>/dev/null "
        f"&& echo STARTED || echo DIED\n"
    )


def status_script(session: str | None = None, tail: int = TAIL_LINES) -> str:
    """List the training sessions; for one of them, also tail its log.

    Read-only by construction — nothing here starts, signals or kills
    anything, so pointing it at a run in progress is always safe.
    """
    s = ("echo '=== sessions'\n"
         "tmux ls -F '#{session_name}\t#{session_created}\t"
         "#{session_windows}' 2>/dev/null || true\n")
    if not session:
        return s
    globs = " ".join(log_globs(session))
    return s + (
        f"echo '=== session {session}'\n"
        f"tmux has-session -t {shlex.quote(session)} 2>/dev/null "
        f"&& echo ALIVE || echo GONE\n"
        f"LOG=$(ls -t {globs} 2>/dev/null | head -1)\n"
        f"echo \"=== log ${{LOG:-none}}\"\n"
        f"[ -n \"$LOG\" ] && tail -n {int(tail)} \"$LOG\"\n"
        f"exit 0\n"
    )


def stop_script(session: str, grace_s: float = STOP_GRACE_S) -> str:
    """Ctrl-C the trainer, give it the grace period, then kill the session.

    The SIGINT is not a save: mjlab's trainer has no interrupt handler, so
    what survives is the last periodic checkpoint (`save_interval`, every 50
    iterations by default). The grace period is for the process to unwind
    wandb and close its files, not to write a model.
    """
    q = shlex.quote(session)
    return (
        f"if ! tmux has-session -t {q} 2>/dev/null; then\n"
        f"  echo 'NO SESSION {session}'\n"
        f"  exit 4\n"
        f"fi\n"
        f"tmux send-keys -t {q} C-c\n"
        f"sleep {float(grace_s):g}\n"
        f"tmux kill-session -t {q} 2>/dev/null || true\n"
        f"tmux has-session -t {q} 2>/dev/null && echo STILL_ALIVE || echo STOPPED\n"
    )


def tasks_script() -> str:
    return f"cd {REPO} && {UV} run list-envs\n"


# --------------------------------------------------------------------------
# Parsing — fixtures in, facts out
# --------------------------------------------------------------------------

def strip_ansi(text: str) -> str:
    """rsl_rl bolds its iteration banner; a regex should not have to care."""
    return _ANSI.sub("", text)


def parse_sessions(text: str) -> list[dict]:
    """`tmux ls -F` rows -> the training sessions, in name order.

    Sessions that are not ours are skipped rather than reported: this tool
    speaks for the duck's runs, and someone's editor session is none of its
    business.
    """
    out = []
    for line in strip_ansi(text).splitlines():
        parts = line.strip().split("\t")
        name = parts[0].strip()
        if not name or not name.startswith(SESSION_PREFIX):
            continue
        created = None
        if len(parts) > 1 and parts[1].strip().isdigit():
            created = datetime.fromtimestamp(
                int(parts[1].strip())).isoformat(timespec="seconds")
        out.append({"session": name, "slug": slug_of_session(name),
                    "created": created})
    return sorted(out, key=lambda d: d["session"])


def parse_wandb(text: str) -> tuple[str | None, str | None]:
    """The run's wandb URL and its `entity/project/run_id` path.

    The path is the point: it is the argument `scripts/export.py
    --wandb-run-path` wants, so pulling it out of the log is one manual step
    removed from train -> export -> ONNX -> the sim.
    """
    # wandb has printed this line three ways over the years — bare
    # "View run at <url>", "View run <name> at <url>", and the current
    # "View run <name> at: <url>". "View project at" is a different line and
    # must not match. Last one wins: a resumed run logs twice.
    urls = re.findall(r"View run\b[^\n]*?\bat:?\s+(https?://\S+)",
                      strip_ansi(text))
    if not urls:
        return None, None
    url = urls[-1].rstrip(".,)")
    m = re.search(r"//[^/]+/([^/\s]+)/([^/\s]+)/runs/([^/\s?#]+)", url)
    return url, ("/".join(m.groups()) if m else None)


# Lines that mean the run is not going to produce a policy. Ordered: the first
# match wins, so the specific diagnoses come before the generic traceback.
_FAILURES = (
    (re.compile(r"Missing value for argument '?\(?'?(--[\w.-]+)"),
     "CLI parse error: {0} needs a value (mjlab's tyro has "
     "FlagConversionOff — booleans take `True`/`False`)"),
    (re.compile(r"(?i)(no space left on device)"), "disk full on the box: {0}"),
    (re.compile(r"(?i)(CUDA out of memory)"), "{0} — lower --env.scene.num-envs"),
    (re.compile(r"(?i)^\s*(Unrecognized (?:or unused )?arguments?.*)$", re.M), "{0}"),
    (re.compile(r"(?i)^\s*(Invalid value for.*)$", re.M), "{0}"),
    (re.compile(r"Traceback \(most recent call last\)"),
     "the trainer raised — see the tail"),
)


def parse_log(text: str) -> dict:
    """The facts an agent wants from a training log, from the log's own tail.

    Everything is read LAST-match-first, because a log is append-only and the
    latest iteration block is the state of the run. Missing keys are missing,
    not zero: an iteration that has not happened must not read as iteration 0.
    """
    t = strip_ansi(text)
    out: dict = {}

    it = re.findall(r"Learning iteration\s+(\d+)\s*/\s*(\d+)", t)
    if it:
        out["iteration"] = int(it[-1][0])
        out["total_iterations"] = int(it[-1][1])

    for key, pat, cast in (
        ("mean_reward", r"^\s*Mean reward:\s*(-?[\d.]+)\s*$", float),
        ("mean_episode_length", r"^\s*Mean episode length:\s*(-?[\d.]+)\s*$", float),
        ("steps_per_second", r"^\s*Steps per second:\s*([\d.]+)\s*$", float),
        ("total_steps", r"^\s*Total steps:\s*(\d+)\s*$", int),
    ):
        m = re.findall(pat, t, re.M)
        if m:
            out[key] = cast(m[-1])

    for key, pat in (("eta", r"^\s*ETA:\s*(\d+:\d\d:\d\d)\s*$"),
                     ("elapsed", r"^\s*Time elapsed:\s*(\d+:\d\d:\d\d)\s*$")):
        m = re.findall(pat, t, re.M)
        if m:
            out[key] = m[-1]

    url, run_path = parse_wandb(t)
    if url:
        out["wandb_url"] = url
    if run_path:
        out["wandb_run_path"] = run_path

    m = re.findall(r"^\[duck-train\] exit rc=(-?\d+) at (\S+)", t, re.M)
    if m:
        out["exit_rc"] = int(m[-1][0])
        out["exit_at"] = m[-1][1]

    for pat, msg in _FAILURES:
        hit = pat.search(t)
        if hit:
            out["failed"] = msg.format(*(g.strip() for g in hit.groups())) \
                if hit.groups() else msg
            break
    if out.get("exit_rc") == 0:
        out.pop("failed", None)
    return out


def parse_status(text: str, session: str | None = None) -> dict:
    """The status script's stdout -> sessions, aliveness, logfile, log facts."""
    body = strip_ansi(text)
    blocks: dict = {}
    key = None
    for line in body.splitlines():
        if line.startswith("=== "):
            key = line[4:].strip()
            blocks[key] = []
            continue
        if key is not None:
            blocks[key].append(line)

    out: dict = {"sessions": parse_sessions("\n".join(blocks.get("sessions", [])))}
    if not session:
        return out
    out["session"] = session
    marker = f"session {session}"
    alive_lines = [ln.strip() for ln in blocks.get(marker, []) if ln.strip()]
    out["alive"] = alive_lines[:1] == ["ALIVE"]
    log_key = next((k for k in blocks if k.startswith("log ")), None)
    if log_key:
        name = log_key[4:].strip()
        out["logfile"] = None if name == "none" else name
        tail = "\n".join(blocks[log_key]).strip()
        out["tail"] = tail or None
        out.update(parse_log(tail))
    return out


def parse_tasks(text: str) -> list[str]:
    """The `list-envs` table -> task ids.

    Row shape is `| 25 | Mjlab-StandUp-Flat-MicroDuck |`; the header row and
    the mdp-patch chatter mjlab prints on import are not tasks.
    """
    out = []
    for line in strip_ansi(text).splitlines():
        m = re.match(r"^\|\s*\d+\s*\|\s*(\S+)\s*\|", line.strip())
        if m:
            out.append(m.group(1))
    return out


# --------------------------------------------------------------------------
# Talking to the box
# --------------------------------------------------------------------------

def wrap_script(script: str) -> str:
    """Brace the script so bash parses all of it before running any of it.

    A script arriving on stdin is read as it runs, so an early `exit` leaves
    the rest of it unread — the local writer takes an EPIPE and ssh reports
    its own status instead of the script's. A refusal then comes back as 1
    rather than 3, which is exactly the kind of lie a caller builds a wrong
    branch on. A brace group is parsed to its closing brace first, so stdin is
    already drained by the time anything can exit.
    """
    return "{\n" + script.rstrip("\n") + "\n}\n"


def run_script(script: str, timeout: float = SSH_TIMEOUT_S,
               host: str = None, via_wsl: bool = None) -> tuple[int, str]:
    """Run a script on the box; return (exit code, stdout+stderr).

    Failures to REACH the box are raised (the agent can do nothing with a
    parse of nothing); failures OF the script come back as its exit code,
    because "that session already exists" is an answer, not an outage.
    """
    argv = ssh_argv(host, via_wsl)
    try:
        p = subprocess.run(argv, input=wrap_script(script), capture_output=True,
                           text=True, timeout=timeout)
    except FileNotFoundError as e:
        raise TrainError("no `ssh` on this machine — the training tools drive "
                         "the GPU box over ssh") from e
    except subprocess.TimeoutExpired as e:
        raise TrainError(
            f"the box did not answer within {timeout:.0f}s "
            f"({' '.join(argv[-2:])})") from e
    # WSL under PowerShell hands back UTF-16; the NULs are not data.
    text = (p.stdout or "").replace("\0", "") + (p.stderr or "").replace("\0", "")
    low = text.lower()
    if "could not resolve hostname" in low or "name or service not known" in low:
        raise TrainError(
            f"ssh host {argv[-2]!r} does not resolve. The training tools want "
            f"an alias that lands directly in Ubuntu on the 4090 box; set one "
            f"up in ~/.ssh/config, or set DUCK_TRAIN_HOST=duck-4090 and "
            f"DUCK_TRAIN_VIA_WSL=1 to go in through PowerShell instead.")
    if p.returncode == 255 and ("connection" in low or "timed out" in low
                                or "permission denied" in low):
        raise TrainError(f"cannot reach the GPU box over ssh: {text.strip()[:300]}")
    return p.returncode, text


# --------------------------------------------------------------------------
# The four verbs
# --------------------------------------------------------------------------

def start(task_id: str, num_envs: int = DEFAULT_NUM_ENVS, video: bool = True,
          extra_args: str | None = None, smoke: bool = False,
          iterations: int | None = None, dry_run: bool = False) -> dict:
    """Launch a run in its own tmux session. Refuses to displace one."""
    session = session_for(task_id)
    if smoke:
        num_envs = SMOKE_NUM_ENVS
        iterations = SMOKE_ITERATIONS if iterations is None else iterations
        video = False        # five iterations of video is five iterations of nothing
    logfile = log_path(task_id)
    runner = runner_script(task_id, logfile, num_envs=num_envs, video=video,
                           iterations=iterations, extra_args=extra_args)
    runner_path = f"{LOG_DIR}/{session}.sh"
    script = start_script(session, runner_path, runner)
    result = {"ok": True, "session": session, "task_id": task_id,
              "logfile": logfile, "num_envs": num_envs, "smoke": smoke,
              "iterations": iterations, "script": script, "started": False}
    if dry_run:
        result["note"] = ("dry run — nothing was sent to the box; `script` is "
                          "exactly what would have run")
        return result
    rc, text = run_script(script)
    if rc == 3 or "REFUSED" in text:
        raise TrainError(
            f"a run for {task_id} is already going in tmux session {session!r}. "
            f"Read it with duck_train_status, or duck_train_stop it first — "
            f"this tool will not replace a live run.")
    if rc != 0 or "STARTED" not in text:
        raise TrainError(f"launch failed (rc={rc}): {text.strip()[:400]}")
    # The script survives on the box as ~/logs/<session>.sh; echoing it back
    # on every successful launch is tokens for something nobody reads.
    result.pop("script", None)
    result["started"] = True
    result["note"] = (
        f"tmux session {session} is running; output is tee'd to {logfile}. "
        f"Nothing blocks — poll duck_train_status. First iteration takes a "
        f"few minutes (JIT + scene compile) before the log says anything.")
    init = re.search(r"^INIT=(.*)$", text, re.M)
    result["distro_init"] = init.group(1).strip() if init else None
    if result["distro_init"] and "systemd" not in result["distro_init"]:
        result["warning"] = (
            f"the box's WSL distro runs {result['distro_init']!r} as PID 1, "
            f"not systemd — WSL tears the distro down seconds after the last "
            f"session ends, taking detached tmux with it. This run will NOT "
            f"survive on its own. Either hold a session open for the duration "
            f"or enable `systemd=true` under [boot] in /etc/wsl.conf on the "
            f"box (one distro restart; do it between runs).")
    return result


def status(session: str | None = None, tail: int = TAIL_LINES) -> dict:
    rc, text = run_script(status_script(session, tail))
    out = parse_status(text, session)
    out["ok"] = True
    if session and not out.get("sessions") and not out.get("logfile"):
        out["note"] = (f"no tmux session {session!r} and no log for it — either "
                       f"it never started or the box was rebooted")
    return out


def stop(session: str, grace_s: float = STOP_GRACE_S) -> dict:
    rc, text = run_script(stop_script(session, grace_s),
                          timeout=SSH_TIMEOUT_S + grace_s)
    if rc == 4 or "NO SESSION" in text:
        raise TrainError(f"no tmux session {session!r} on the box "
                         f"(duck_train_status lists what is running)")
    stopped = "STOPPED" in text
    return {"ok": stopped, "session": session, "stopped": stopped,
            "note": ("interrupted and killed. What survives is the last "
                     "periodic checkpoint (every 50 iterations by default) — "
                     "the trainer does not save on Ctrl-C."
                     if stopped else
                     "sent Ctrl-C but the session is still there; it may be "
                     "unwinding. Call again to kill it.")}


_TASKS_CACHE: dict = {}


def tasks(refresh: bool = False) -> dict:
    """The box's live task registry. Cached: `list-envs` imports every env."""
    if refresh or "ids" not in _TASKS_CACHE:
        rc, text = run_script(tasks_script(), timeout=TASKS_TIMEOUT_S)
        ids = parse_tasks(text)
        if not ids:
            raise TrainError(f"`uv run list-envs` listed no tasks (rc={rc}): "
                             f"{text.strip()[:400]}")
        _TASKS_CACHE.update(ids=ids, at=time.time())
    return {"ok": True, "tasks": _TASKS_CACHE["ids"],
            "cached_at": datetime.fromtimestamp(
                _TASKS_CACHE["at"]).isoformat(timespec="seconds"),
            "count": len(_TASKS_CACHE["ids"])}


# --------------------------------------------------------------------------
# `duck train ...` — the same four verbs, for humans and for testing
# --------------------------------------------------------------------------

def add_arguments(p) -> None:
    p.add_argument("action", choices=["start", "status", "stop", "tasks"])
    p.add_argument("arg", nargs="?", default=None,
                   help="task id for start, session name for status/stop")
    p.add_argument("--num-envs", type=int, default=DEFAULT_NUM_ENVS,
                   help="parallel envs (--smoke overrides this)")
    p.add_argument("--iterations", type=int, default=None,
                   help="--agent.max_iterations cap")
    p.add_argument("--smoke", action="store_true",
                   help=f"cheap validation run: {SMOKE_NUM_ENVS} envs, "
                        f"{SMOKE_ITERATIONS} iterations — always run one first")
    p.add_argument("--no-video", action="store_false", dest="video",
                   help="skip training-video capture")
    p.add_argument("--extra", default=None,
                   help="extra flags for the trainer, verbatim")
    p.add_argument("--tail", type=int, default=TAIL_LINES,
                   help="status: log lines to read back")
    p.add_argument("--refresh", action="store_true", help="tasks: re-read the registry")
    p.add_argument("--dry-run", action="store_true",
                   help="start: print the script instead of sending it")


def run(args) -> int:
    import json
    try:
        if args.action == "start":
            out = start(args.arg, num_envs=args.num_envs, video=args.video,
                        extra_args=args.extra, smoke=args.smoke,
                        iterations=args.iterations, dry_run=args.dry_run)
        elif args.action == "status":
            out = status(args.arg, tail=args.tail)
        elif args.action == "stop":
            if not args.arg:
                raise TrainError("stop which session? (`duck train status` lists them)")
            out = stop(args.arg)
        else:
            out = tasks(refresh=args.refresh)
    except TrainError as e:
        print(json.dumps({"ok": False, "error": str(e)}, indent=2))
        return 1
    print(json.dumps(out, indent=2))
    return 0 if out.get("ok") else 1
