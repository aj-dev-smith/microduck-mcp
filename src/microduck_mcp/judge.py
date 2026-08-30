"""Rollout judgment: an outside eye on what training actually produced.

The training tools read the run's NUMBERS (train.py); this module watches the
run's VIDEOS. The distinction is load-bearing: total reward can climb on
regularizers while the trick never happens, and the 2026 reward-design
literature (RDA and friends) found the same thing the hard way — a reward
loop steered only by scalars drifts from intent, and a vision model watching
rendered rollouts catches it. mjlab already records a clip every ~2000 env
steps of every `--video True` run; this module fetches the newest one off the
box and asks Gemini what the policy actually does in it.

Gemini, not the orchestrating model, for two reasons: a video review costs a
fraction of a cent there (free tier covers a whole run's worth), and a judge
with different blind spots than the reward's author is a feature in a loop
whose failure mode is self-deception.

## The population rubric

An mjlab training video shows SEVERAL envs at once, each starting in a
different state. The first draft of this judge scored "the duck" and
confidently described whichever one it found salient — a fallen one, while
its neighbor stood perfectly. The rubric therefore demands a per-subject
account and only credits the task when a subject that started outside the
goal state visibly achieves it. Verdicts are structured JSON so a machine
step can branch on them; the summary is for the human.

## Auth

Reads GEMINI_API_KEY from the environment (a free-tier AI Studio key is
enough). No key material is ever logged or returned.
"""

import json
import os
import re
import subprocess
import tempfile

from . import train

# Newest first: 3.7-flash is cheapest-best but 503s under free-tier load;
# 3.6-flash has answered every time and judges rollouts indistinguishably.
MODELS = ("gemini-3.7-flash", "gemini-3.6-flash", "gemini-flash-latest")

# Where mjlab writes rollout clips on the box, relative to the training repo.
VIDEO_GLOB = f"{train.REPO}/logs/rsl_rl/*/*/videos/train/*.mp4"

SCP_TIMEOUT_S = 60.0
JUDGE_TIMEOUT_S = 120.0

_STEP = re.compile(r"rl-video-step-(\d+)\.mp4$")

# Cheat patterns that generalize across duck tasks; a caller adds
# task-specific ones via watch_for.
COMMON_CHEATS = (
    "flail: high-energy thrashing with no progress toward the task",
    "freeze: doing nothing at all (a reward-tax victim)",
    "jitter: rapid trembling while nominally succeeding (hurts real servos)",
    "physics-surfing: vibrating against the ground/contacts to move in ways "
    "no real robot could",
)

RUBRIC = """\
You are reviewing a rollout video from a reinforcement-learning training run.
The robot is a ~25 cm tall bipedal duck (14 servos: two legs, a neck, a head).

The TASK being trained: {task}

IMPORTANT: the video typically shows SEVERAL ducks at once — independent
parallel simulations of the same policy, each starting in a different state
(some may start already in the goal state). Judge the POPULATION, duck by
duck: count them and report what each one does. The policy is only credited
with the task if a duck that started OUTSIDE the goal state visibly achieves
it. Judge what the policy actually does — not what a reward curve claims.

Watch for these cheat/failure patterns:
{cheats}

Respond with JSON only, matching this schema:
{{
  "ducks_seen": <how many ducks are visible>,
  "per_duck": ["<one short phrase per duck: start state -> what it did>"],
  "task_achieved": <true only if a duck starting outside the goal state visibly achieved the task>,
  "quality": <0-10 for the population: 0 = never close, 10 = smooth deliberate success>,
  "failure_modes": ["<zero or more of the named patterns, or your own>"],
  "cheating_suspected": <true if the goal is gamed rather than achieved>,
  "summary": "<2-3 sentences, concrete, about what you saw>"
}}
"""


class JudgeError(RuntimeError):
    """Something the agent should read and act on — no key, no clip, no answer."""


def step_of(path: str) -> int | None:
    """`.../rl-video-step-20000.mp4` -> 20000; None for foreign names."""
    m = _STEP.search(path)
    return int(m.group(1)) if m else None


def find_clips_script(match: str | None = None, limit: int = 5) -> str:
    """Shell that lists the newest rollout clips on the box, one path per line.

    `ls -t` over the glob, because the box has no manifest to consult — the
    filesystem is the record, same as the training logs. `match` narrows to
    runs whose path mentions it (e.g. "microduck_stand"); it is embedded with
    grep -F so it is never a pattern.
    """
    filt = f" | grep -F {train_shquote(match)}" if match else ""
    return (f"ls -t {VIDEO_GLOB} 2>/dev/null{filt} | head -n {int(limit)}\n")


def train_shquote(s: str) -> str:
    """shlex.quote, named for where the convention comes from."""
    import shlex
    return shlex.quote(s)


def latest_clip(match: str | None = None) -> str:
    """Remote path of the newest rollout clip on the box, or a refusal."""
    rc, out = train.run_script(find_clips_script(match))
    lines = [ln.strip() for ln in out.splitlines() if ln.strip().endswith(".mp4")]
    if rc != 0 or not lines:
        raise JudgeError(
            "no rollout clips on the box"
            + (f" matching {match!r}" if match else "")
            + " — was the run launched with video=True? "
            "(duck_train_start records them by default)")
    return lines[0]


def fetch_clip(remote_path: str, dest_dir: str | None = None) -> str:
    """Copy one clip off the box; returns the local path.

    scp with the same host alias as the training ssh — the clips are small
    (~100 KB: 320x240, four seconds), so this is the cheap step.
    """
    dest_dir = dest_dir or tempfile.mkdtemp(prefix="duck-rollout-")
    local = os.path.join(dest_dir, os.path.basename(remote_path))
    argv = ["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
            f"{train.SSH_HOST}:{remote_path}", local]
    try:
        p = subprocess.run(argv, capture_output=True, text=True,
                           timeout=SCP_TIMEOUT_S)
    except subprocess.TimeoutExpired as e:
        raise JudgeError(f"scp of {remote_path} timed out") from e
    if p.returncode != 0 or not os.path.exists(local):
        raise JudgeError(f"could not fetch {remote_path}: "
                         f"{(p.stderr or p.stdout).strip()[:300]}")
    return local


def review(clip_path: str, task: str, watch_for: list[str] | None = None,
           model: str | None = None) -> dict:
    """Ask Gemini what the policy in this clip actually does.

    Tries the MODELS chain unless one is named: the newest flash model 503s
    under free-tier load spikes, and a judge that answers beats a judge that
    is 2% better. The import is deferred so a client that never judges never
    pays for it, exactly as duck_say defers the voice.
    """
    if not os.environ.get("GEMINI_API_KEY"):
        raise JudgeError("GEMINI_API_KEY is not set — the judge needs a "
                         "(free-tier is fine) Gemini API key in the environment")
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        raise JudgeError("google-genai is not installed — "
                         "`uv sync` in microduck-mcp") from e

    cheats = "\n".join(f"- {c}" for c in (*COMMON_CHEATS, *(watch_for or ())))
    prompt = RUBRIC.format(task=task.strip(), cheats=cheats)
    data = open(clip_path, "rb").read()

    client = genai.Client()
    last: Exception | None = None
    for m in ((model,) if model else MODELS):
        try:
            resp = client.models.generate_content(
                model=m,
                contents=[types.Part.from_bytes(data=data, mime_type="video/mp4"),
                          prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json", temperature=0.2),
            )
            verdict = json.loads(resp.text)
            verdict["model"] = m
            return verdict
        except Exception as e:  # 503s, quota, malformed JSON: try the next
            last = e
    raise JudgeError(f"no judge model answered ({type(last).__name__}: "
                     f"{str(last)[:200]})")


def latest(task: str, match: str | None = None,
           watch_for: list[str] | None = None) -> dict:
    """The whole verb: newest clip on the box -> fetched -> judged."""
    remote = latest_clip(match)
    local = fetch_clip(remote)
    verdict = review(local, task, watch_for=watch_for)
    return {"ok": True, "clip": remote, "step": step_of(remote),
            "local": local, **verdict}
