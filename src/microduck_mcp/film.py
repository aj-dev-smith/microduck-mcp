"""Film an autonomous striker match: broadcast camera + head-cam PiP -> mp4.

`duck film` runs cold-start matches on the pitch scene and keeps one that ends
in a goal. Each frame is a composite of four honest things:

  * the BROADCAST camera — a touchline shot that tracks the duck and the ball,
    and swings around to the west side for the celebration so the goal frame
    stops blocking the money shot;
  * the DUCK CAM picture-in-picture — the 70 deg head camera the detectors
    actually run on, so a viewer sees what the robot sees;
  * the SENSED-STATE HUD — `ball_seen.*` and `goal_seen.est_*` straight out of
    the machine digest, the entire vocabulary the behavior machine steers on;
  * the CONTROL-SURFACE FEED — real events off the sim's command ring: MCP
    load/arm calls, machine transitions, and the guard expression that fired
    each one, plus the referee's goal call.

It also has a voice. The soundtrack (soundtrack.py) is cut from the take's own
event timeline on the same sim clock the frames are sampled on: the duck says
a line as the machine arms, chirps on the kick, and the referee's goal — only
the referee's goal — gets the wheee, with the celebration line behind it. The
beak moves with the words in the picture, off the same trajectory that drives
the mix. Audio is an enhancement: anything that cannot be rendered is noted on
stderr and the film is cut silently rather than not at all.

Unlike every other `duck` subcommand this one does not talk to a running sim
over the control socket: filming needs raw frame buffers at 17 fps, per-take
resets and chosen spawns, none of which are (or should be) socket intents. So
`duck film` boots its own headless `DuckSim` in-process, films, and exits. A
sim you already have running is left alone.

Frames are piped raw into ffmpeg — an external binary, checked for before any
of the slow work starts.
"""

import argparse
import bisect
import math
import os
import shutil
import subprocess
import sys
import tempfile
from typing import NamedTuple

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import sim_server as ss

W, H = 960, 540
PIP_W, PIP_H = 264, 198
FPS_EVERY = 3               # render every 3 control ticks -> 16.67 fps
FPS = ss.CONTROL_HZ / FPS_EVERY
CRF = 21

DEFAULT_OUTPUT = "duck_match.mp4"
DEFAULT_MACHINE = "machines/striker.toml"
DEFAULT_CAP_S = 150.0

# Cold-start spawns that give the striker a fair match: a metre-ish out, off
# the goal's centre line, facing somewhere other than the ball. Hand-checked,
# but not a script — float nondeterminism in the policy runner means the same
# spawn plays out differently run to run, which is why a shoot takes takes.
MATCH_SPAWNS = ((-1.11, -0.45, 32.0), (-2.14, -0.45, 2.0), (-0.92, -0.76, -81.0),
                (-1.33, -0.69, -147.0), (-1.75, 0.20, 161.0))

# Control-surface feed: one colour per client, matching the AX debug page.
FEED_COLORS = {"mcp": (140, 255, 160), "machine": (120, 220, 255),
               "referee": (255, 215, 40)}
FEED_PREFIXES = {"mcp": "mcp>     ", "machine": "machine> ",
                 "referee": "referee> "}
FEED_TAIL = 8   # events scanned for the feed (the guard may precede the line)
FEED_ROWS = 2   # lines actually drawn
HUD_COLORS = ((255, 190, 90), (255, 190, 90), (150, 220, 255))

CREDIT = "behavior machine by Claude, via MCP"
TITLE_CARD = ("CLAUDE PLAYS STRIKER",
              "a robot duck driven over MCP  ·  microduck-mcp",
              "the behavior machine was written, calibrated and",
              "armed by Claude — it plays off the duck's own camera")
END_CARD = ("DUCK 1 - 0 GOAL",
            "sensed, aimed and scored — no ground truth",
            "github.com/aj-dev-smith/microduck-mcp")
CARD_S = 2.8

# The cold open: the goal frame, held, so it becomes the timeline thumbnail.
# Everything the soundtrack places is offset by it — the match starts here.
HOOK_S = 1.0
HOOK_FRAMES = int(HOOK_S * FPS)

SANS_CANDIDATES = ("/System/Library/Fonts/Helvetica.ttc",
                   "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
MONO_CANDIDATES = ("/System/Library/Fonts/Menlo.ttc",
                   "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")

FFMPEG_MISSING = (
    "ffmpeg not found ({binary!r} is not on PATH).\n"
    "duck film pipes raw frames into ffmpeg to encode the mp4; it is an "
    "external binary, not a Python dependency.\n"
    "  macOS:         brew install ffmpeg\n"
    "  Debian/Ubuntu: sudo apt install ffmpeg\n"
    "  elsewhere:     https://ffmpeg.org/download.html\n"
    "Already have one somewhere else? Point at it with --ffmpeg /path/to/ffmpeg."
)


class FilmError(RuntimeError):
    """Anything that stops a shoot before it starts (or an encode failing)."""


def _soundtrack():
    """The soundtrack module, imported late on purpose.

    voice.py imports this module for the ffmpeg preflight and soundtrack.py
    imports voice.py, so a top-level import here would close the ring. The
    sound is also strictly optional to filming, which is the shape a lazy
    import is for.
    """
    from . import soundtrack
    return soundtrack


def frame_clock(frame_times, hook_frames: int = HOOK_FRAMES, fps: float = FPS):
    """Take-relative sim time -> time on the finished film's clock.

    The cut holds a goal frame as a cold open before the match starts, so the
    two clocks differ by that hook. Times are resolved through the frames that
    were actually kept — a cue lands on the frame the viewer is looking at
    when the thing happened, not on an arithmetic guess about the sample rate.
    """
    def at(t_s: float) -> float:
        if not frame_times:
            return hook_frames / fps
        k = min(bisect.bisect_left(frame_times, t_s), len(frame_times) - 1)
        return (hook_frames + k) / fps
    return at


# ---------- take selection ----------

KEEP, FALLBACK, DISCARD = "keep", "fallback", "discard"


class Take(NamedTuple):
    """One filmed episode and how it ended."""
    index: int
    spawn: tuple
    path: str
    scored: bool
    node: str
    duration_s: float


def classify_take(take: Take, prefer_won: bool = True) -> str:
    """keep / fallback / discard.

    The take we want scored AND landed the celebration (`won`): the duck rolls,
    gets back up and stands over its goal. A goal that ends with the duck on
    its side (`down`) is still a goal, so it is kept as plan B rather than
    thrown away — with --select goal it is a keeper outright. No goal, no take.
    """
    if not take.scored:
        return DISCARD
    if take.node == "won" or not prefer_won:
        return KEEP
    return FALLBACK


def select_take(takes, prefer_won: bool = True):
    """The take that makes the film: first keeper, else first fallback."""
    fallback = None
    for take in takes:
        verdict = classify_take(take, prefer_won)
        if verdict == KEEP:
            return take
        if verdict == FALLBACK and fallback is None:
            fallback = take
    return fallback


# ---------- overlays (pure formatting: what the AI knows, on screen) ----------

def _f(v, fmt="{:+.3f}") -> str:
    """A digest field, or a placeholder — nulls are honest and must show."""
    return "  --  " if v is None else fmt.format(v)


def hud_lines(digest: dict) -> list:
    """The sensed-state HUD, three lines, exactly the machine's vocabulary."""
    return [
        f"ball fwd {_f(digest['ball_seen.est_forward_m'])} "
        f"left {_f(digest['ball_seen.est_left_m'])}",
        f"     spd {_f(digest['ball_seen.speed_mps'], '{:.3f}')} m/s",
        f"goal brg {_f(digest['goal_seen.est_bearing_deg'], '{:+.1f}')}deg "
        f"dist {_f(digest['goal_seen.est_distance_m'], '{:.2f}')}",
    ]


def feed_lines(events):
    """Command feed -> ([(prefix, text, colour)], guard expression or None).

    Reads the sim's own event ring, so what scrolls past is the real control
    surface: the MCP calls that armed the machine, the transitions the machine
    fired, the referee's call. The guard is the `when` of the most recent
    transition in view — the film's whole point is that the expression which
    fired is legible next to the behavior it produced.
    """
    lines = []
    guard = None
    for ev in list(events)[-FEED_TAIL:]:
        client = ev["client"]
        args = ev.get("args", {})
        if client == "mcp":
            text = (f"duck machine {args.get('action', '')} "
                    f"{args.get('path', '') or ''}").rstrip()
        elif client == "machine":
            text = f"{args.get('from')} {ev['cmd']}"
            guard = args.get("when")
        elif client == "referee":
            text = f"GOAL! #{args.get('count')}"
        else:
            continue
        lines.append((FEED_PREFIXES[client], text, FEED_COLORS[client]))
    return lines, guard


# ---------- ffmpeg ----------

def find_ffmpeg(binary: str = "ffmpeg") -> str:
    """Absolute path to ffmpeg, or FilmError with somewhere to go next.

    Called before the model loads: a missing encoder should cost a second, not
    a two-minute match filmed into a broken pipe.
    """
    path = shutil.which(binary)
    if path is None:
        raise FilmError(FFMPEG_MISSING.format(binary=binary))
    return path


def encode(frames, out_path: str, ffmpeg: str, width: int = W, height: int = H,
           fps: float = FPS, crf: int = CRF):
    """Pipe raw RGB frames into ffmpeg -> a faststart h264 mp4."""
    proc = subprocess.Popen(
        [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", f"{fps}",
         "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf),
         "-movflags", "+faststart", out_path], stdin=subprocess.PIPE)
    try:
        for frame in frames:
            proc.stdin.write(frame.tobytes())
    except BrokenPipeError:
        pass
    proc.stdin.close()
    if proc.wait() != 0:
        raise FilmError(f"ffmpeg exited {proc.returncode} encoding {out_path}")
    return out_path


# ---------- the broadcast camera ----------

class BroadcastCamera:
    """Match phase: fixed azimuth from the touchline, smoothed lookat between
    duck and ball, distance adapting to keep both in frame (no yaw coupling —
    the search spins stay watchable).
    Celebration phase: swings around to the WEST side and closes in, so the
    duck tumbles front-and-centre with the net as a backdrop instead of the
    goal frame blocking the money shot.

    The numbers were tuned by eye against real footage. They are not knobs.
    """

    def __init__(self, sim):
        self.sim = sim
        self.look = None
        self.dist = 1.8
        self.az = -145.0
        self.el = -24.0

    def mjv(self, celebrating: bool = False):
        sim = self.sim
        a, ba = sim.qpos_adr, sim.policy.ball_qpos_adr
        duck = sim.data.qpos[a:a + 2]
        ball = sim.data.qpos[ba:ba + 2]
        if celebrating:
            mid = np.array([duck[0], duck[1], 0.10])
            want_d, want_az, want_el = 0.95, 0.0, -14.0
        else:
            mid = np.array([(duck[0] + ball[0]) / 2,
                            (duck[1] + ball[1]) / 2, 0.10])
            sep = float(np.hypot(duck[0] - ball[0], duck[1] - ball[1]))
            want_d = min(2.6, max(1.1, 1.1 * sep + 0.9))
            want_az, want_el = -145.0, -24.0
        if self.look is None:
            self.look = mid
        ease = 0.06 if celebrating else 0.03
        self.look = (1 - ease) * self.look + ease * mid
        self.dist += max(-0.035, min(0.035, want_d - self.dist))
        self.az += max(-4.5, min(4.5, want_az - self.az))   # ~75 deg/s swing
        self.el += max(-0.6, min(0.6, want_el - self.el))
        cam = mujoco.MjvCamera()
        cam.lookat[:] = self.look
        cam.distance = self.dist
        cam.azimuth = self.az
        cam.elevation = self.el
        return cam


def _font(candidates, size: int):
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


class Fonts(NamedTuple):
    big: object
    med: object
    huge: object
    title: object
    mono: object
    mono_s: object

    @classmethod
    def load(cls) -> "Fonts":
        return cls(_font(SANS_CANDIDATES, 44), _font(SANS_CANDIDATES, 26),
                   _font(SANS_CANDIDATES, 110), _font(SANS_CANDIDATES, 52),
                   _font(MONO_CANDIDATES, 17), _font(MONO_CANDIDATES, 14))


# ---------- the shoot ----------

class MatchFilm:
    """One headless sim, rendered into takes.

    Reaches into DuckSim's internals (the event ring, the machine digest, the
    head-camera id) on purpose: the film is a view of the sim's own state, and
    everything it shows a viewer is something the sim already publishes to the
    AX debug page.
    """

    def __init__(self, sim, ffmpeg: str, machine_path: str,
                 machine_label: str = None, cards: bool = True,
                 quiet: bool = False, kit=None):
        self.sim = sim
        self.ffmpeg = ffmpeg
        self.machine_path = machine_path
        self.kit = kit          # a soundtrack.SoundKit, or None for a silent film
        # What the feed shows for the load call: the path as a human would
        # type it, not the absolute one the loader resolved to.
        self.machine_label = machine_label or machine_path
        self.cards = cards
        self.quiet = quiet
        self.dt = ss.DECIMATION * ss.TIMESTEP
        self.renderer = mujoco.Renderer(sim.model, height=H, width=W)
        self.pip = mujoco.Renderer(sim.model, height=PIP_H, width=PIP_W)
        self.fonts = Fonts.load()

    def close(self):
        for r in (self.renderer, self.pip):
            try:
                r.close()
            except Exception:
                pass

    def _say(self, msg: str):
        if not self.quiet:
            print(msg, flush=True)

    def tick(self, mouth: float = 0.0):
        """One 50 Hz control step — run_loop's body, minus socket and pacing.

        `mouth` is the commanded beak opening for this step, so the duck's
        mouth moves with its lines on camera. It rides on a mocap plate with
        no collision geometry and no dofs, exactly as it does live: the walk
        policy's observation is untouched, and a silent film steps the same
        physics as a talking one.
        """
        sim = self.sim
        sim.policy.update_ground_pick_phase(self.dt)
        sim.policy.update_behavior(self.dt)
        sim.policy.apply_action(sim.policy.infer())
        sim.mouth_opening = float(min(1.0, max(0.0, mouth)))
        sim.mouth_tick()
        for _ in range(ss.DECIMATION):
            mujoco.mj_step(sim.model, sim.data)
        sim.sim_time += self.dt
        sim.sense()
        sim.referee_tick()
        sim.machine_tick()

    def render_frame(self, cam, node, goals, goal_age_s, t, celebrating=False,
                     mouth=None):
        sim, f = self.sim, self.fonts
        self.renderer.update_scene(sim.data, camera=cam.mjv(celebrating))
        img = Image.fromarray(self.renderer.render())
        # head-cam PiP (what the duck actually sees)
        if sim._head_cam_id >= 0:
            self.pip.update_scene(sim.data, camera=sim._head_cam_id)
            pip = Image.fromarray(self.pip.render())
            bx, by = W - PIP_W - 16, 16
            d0 = ImageDraw.Draw(img)
            d0.rectangle([bx - 3, by - 3, bx + PIP_W + 3, by + PIP_H + 3],
                         outline=(255, 255, 255), width=3)
            img.paste(pip, (bx, by))
            d0.text((bx + 8, by + PIP_H - 30), "duck cam", font=f.med,
                    fill=(255, 255, 255))
        d = ImageDraw.Draw(img)
        # score bug
        d.rectangle([16, 16, 320, 66], fill=(10, 40, 16))
        d.text((28, 24), f"DUCK  {goals} - 0  GOAL", font=f.med,
               fill=(255, 255, 255))
        d.text((28 + 330, 24), f"t={t:5.1f}s", font=f.med, fill=(240, 240, 240))
        # sensed-state HUD under the duck cam: exactly what the AI knows
        bx, by = W - PIP_W - 16, 16 + PIP_H + 10
        d.rectangle([bx - 3, by, W - 13, by + (66 if mouth is None else 88)],
                    fill=(12, 12, 12))
        for i, (line, col) in enumerate(zip(hud_lines(sim._machine_digest()),
                                            HUD_COLORS)):
            d.text((bx + 6, by + 4 + 20 * i), line, font=f.mono_s, fill=col)
        if mouth is not None:
            # The beak trajectory, while the duck is talking: the same signal
            # that opens the mouth on camera and places the chirps in the
            # voice — so the HUD is showing the mix, not illustrating it.
            ty = by + 4 + 20 * 3
            d.text((bx + 6, ty), "beak", font=f.mono_s, fill=(200, 200, 200))
            x0, x1 = bx + 6 + 44, W - 20
            d.rectangle([x0, ty + 3, x1, ty + 13], outline=(90, 90, 90))
            fill_w = (x1 - x0 - 2) * min(1.0, max(0.0, mouth))
            if fill_w >= 1:
                d.rectangle([x0 + 1, ty + 4, x0 + 1 + fill_w, ty + 12],
                            fill=(255, 190, 90))
        # control-surface feed: the real MCP commands + machine transitions
        bar_y = H - 96
        d.rectangle([0, bar_y, W, H], fill=(8, 8, 8))
        lines, guard = feed_lines(sim.events)
        y = bar_y + 6
        for prefix, text, col in lines[-FEED_ROWS:]:
            d.text((16, y), prefix, font=f.mono, fill=(150, 150, 150))
            d.text((16 + 90, y), text[:88], font=f.mono, fill=col)
            y += 22
        if guard:
            d.text((16, y), "guard>   ", font=f.mono, fill=(150, 150, 150))
            g = guard if len(guard) <= 105 else guard[:102] + "..."
            d.text((16 + 90, y), g, font=f.mono_s, fill=(200, 200, 200))
        d.text((16, bar_y - 30), f"machine: {node}", font=f.med,
               fill=(120, 220, 255))
        d.text((W - d.textlength(CREDIT, font=f.med) - 16, bar_y - 34), CREDIT,
               font=f.med, fill=(235, 235, 235))
        # goal splash
        if goal_age_s is not None and goal_age_s < 2.5:
            tw = d.textlength("GOAL!", font=f.huge)
            d.text(((W - tw) / 2 + 4, H / 2 - 90 + 4), "GOAL!", font=f.huge,
                   fill=(0, 0, 0))
            d.text(((W - tw) / 2, H / 2 - 90), "GOAL!", font=f.huge,
                   fill=(255, 215, 40))
        return np.asarray(img)

    def card(self, lines, seconds: float, sub_from: int = 1):
        img = Image.new("RGB", (W, H), (10, 14, 10))
        d = ImageDraw.Draw(img)
        y = H / 2 - 34 * len(lines) - 10
        for i, text in enumerate(lines):
            font = self.fonts.title if i < sub_from else self.fonts.med
            tw = d.textlength(text, font=font)
            col = (255, 255, 255) if i < sub_from else (170, 200, 175)
            d.text(((W - tw) / 2, y), text, font=font, fill=col)
            y += 74 if i < sub_from else 40
        return [np.asarray(img)] * int(seconds * FPS)

    def start_take(self, spawn):
        """Reset to a cold start at `spawn`, then load and arm the machine."""
        sim = self.sim
        x, y, yaw = spawn
        sim._handle_reset()
        # Stale transitions from the previous take would otherwise scroll past
        # in this take's feed, over guards that never fired here.
        sim.events.clear()
        half = math.radians(yaw) / 2
        a = sim.qpos_adr
        sim.data.qpos[a:a + 2] = (x, y)
        sim.data.qpos[a + 3:a + 7] = (math.cos(half), 0, 0, math.sin(half))
        mujoco.mj_forward(sim.model, sim.data)
        for req, shown in (({"cmd": "machine", "action": "load",
                             "path": self.machine_path}, self.machine_label),
                           ({"cmd": "machine", "action": "arm"}, None)):
            resp = sim.handle(dict(req))
            if not resp.get("ok"):
                raise FilmError(f"machine {req['action']} failed: "
                                f"{resp.get('error')}")
            # Logged as if it came over the socket — it is the same call the
            # runbook makes by hand, and the feed should say so.
            sim._log_event("mcp", {**req, **({"path": shown} if shown else {})},
                           resp)

    def _cue_speech(self, speaking, beat: str, start_s: float):
        """Schedule a rendered line's beak trajectory from `start_s`.

        The lines are rendered before the shoot, so the mouth can be animated
        live during the take from the same trajectory the mix will use — the
        picture and the track are cut from one render, not synced afterwards.
        """
        voicing = self.kit.voicings.get(beat) if self.kit else None
        if voicing is not None:
            speaking.append((start_s, voicing))

    def film_take(self, index: int, spawn, out_path: str,
                  cap_s: float = DEFAULT_CAP_S) -> Take:
        """Run one match, and cut it (picture and sound) if it produced a goal."""
        st = _soundtrack()
        sim = self.sim
        self.start_take(spawn)
        frames = []       # buffered so the goal moment can cold-open the cut
        frame_times = []  # each frame's take-relative sim time: the sound clock
        beats = [st.Beat(st.ARM, 0.0, "arm")]
        speaking = []     # [(start_s, Voicing)] — what the beak is up to
        self._cue_speech(speaking, "arm", 0.0)
        cam = BroadcastCamera(sim)
        t0 = sim.sim_time
        goal_t = None
        end_t = None
        node = sim.machine.current
        mouth = None
        i = 0
        while sim.sim_time - t0 < cap_s:
            self.tick(mouth or 0.0)
            i += 1
            t_rel = sim.sim_time - t0
            goals = sim.referee.count
            if goals > 0 and goal_t is None:
                goal_t = sim.sim_time
                beats.append(st.Beat(st.GOAL, t_rel))
                self._cue_speech(speaking, "goal",
                                 t_rel + st.GOAL_LINE_DELAY_S)
                self._say(f"  GOAL on film at t={t_rel:.1f}s")
            if sim.machine.current != node:
                node = sim.machine.current
                beats.append(st.Beat(st.NODE, t_rel, node))
            if end_t is None and (node in ("won", "down") and goals > 0):
                end_t = sim.sim_time + 3.0  # linger on the final pose
            if end_t is None and node == "down" and goals == 0:
                break  # fell without scoring: not the take
            mouth = st.mouth_at(speaking, t_rel)
            if i % FPS_EVERY == 0:
                ga = None if goal_t is None else sim.sim_time - goal_t
                celebrating = goals > 0 or node == "celebrate"
                frames.append(self.render_frame(cam, node, goals, ga, t_rel,
                                                celebrating, mouth))
                frame_times.append(t_rel)
            if end_t is not None and sim.sim_time >= end_t:
                break
        scored = sim.referee.count > 0
        take = Take(index, tuple(spawn), out_path, scored, sim.machine.current,
                    round(sim.sim_time - t0, 1))
        if scored:
            seq = self.cut(frames, goal_t, t0)
            encode(seq, out_path, self.ffmpeg)
            self.dub(out_path, beats, frame_times, len(seq))
        self._say(f"take {index} {tuple(spawn)}: scored={scored} "
                  f"node={take.node} len={take.duration_s:.1f}s")
        return take

    def dub(self, path: str, beats, frame_times, n_frames: int):
        """Cut the soundtrack to this take's beats and mux it onto the film.

        Wrapped end to end: a soundtrack is worth having and never worth
        losing a filmed goal over, so every failure here downgrades to a
        silent film plus a note.
        """
        if self.kit is None or not self.kit.audible:
            return
        st = _soundtrack()
        try:
            cues = st.plan_cues(beats, self.kit.lines)
            track = self.kit.track(cues, n_frames / FPS,
                                   frame_clock(frame_times))
            st.mux(path, track, self.ffmpeg)
            self._say(f"  audio: {st.scored_note(cues)}")
        except Exception as e:
            print(f"note: no soundtrack on take ({e}) — the film is silent",
                  file=sys.stderr)

    def cut(self, frames, goal_t, t0):
        """Assemble for the scroll-by: the GOAL! money shot as a 1 s cold open
        (it becomes the timeline thumbnail), then the match from the top,
        credits at the very end."""
        hook_i = min(len(frames) - 1,
                     int((goal_t - t0 + 1.1) * FPS)) if goal_t else 0
        seq = [frames[hook_i]] * HOOK_FRAMES + frames
        if self.cards:
            seq += self.card(TITLE_CARD, CARD_S) + self.card(END_CARD, CARD_S)
        return seq


# ---------- CLI ----------

def take_path(out_path: str, index: int, keep: bool) -> str:
    """Where take N is written. Beside the output, so the winner is a rename
    on the same filesystem; hidden unless the shoot is keeping its rushes."""
    directory = os.path.dirname(os.path.abspath(out_path))
    stem = os.path.splitext(os.path.basename(out_path))[0]
    dot = "" if keep else "."
    return os.path.join(directory, f"{dot}{stem}.take{index}.mp4")


def resolve_machine(path: str) -> str:
    """A machine path relative to the cwd, or to the repo the code lives in —
    so `duck film` works from anywhere without spelling out machines/."""
    if os.path.isfile(path):
        return os.path.abspath(path)
    repo_relative = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), path)
    if os.path.isfile(repo_relative):
        return repo_relative
    raise FilmError(f"machine not found: {path} (looked in the working "
                    f"directory and at {repo_relative})")


def add_arguments(parser: argparse.ArgumentParser):
    """Flags for `duck film`. Zero-config by design: with the sibling repo
    layout the defaults already film a striker match into ./duck_match.mp4."""
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT,
                        help=f"mp4 to write (default {DEFAULT_OUTPUT})")
    parser.add_argument("--takes", type=int, default=len(MATCH_SPAWNS),
                        metavar="N",
                        help=f"how many spawns to try, in order, until one is "
                             f"a keeper (1-{len(MATCH_SPAWNS)})")
    parser.add_argument("--select", choices=("won", "goal"), default="won",
                        help="'won' (default) keeps a take that scored AND "
                             "landed the celebration, falling back to any "
                             "goal; 'goal' keeps the first take that scores")
    parser.add_argument("--cap-seconds", type=float, default=DEFAULT_CAP_S,
                        metavar="S",
                        help=f"give up on a take after this much sim time "
                             f"(default {DEFAULT_CAP_S:g})")
    parser.add_argument("--machine", default=DEFAULT_MACHINE, metavar="PATH",
                        help=f"behavior machine to arm (default {DEFAULT_MACHINE})")
    parser.add_argument("--scene", choices=sorted(ss.SCENES), default="pitch",
                        help="scene to film (needs a goal; default pitch)")
    parser.add_argument("--rl-repo", default=os.environ.get("MICRODUCK_RL_REPO",
                                                            "../microduck_rl"),
                        help="Path to a microduck_rl clone (scenes + PolicyInference)")
    parser.add_argument("--policies", default=os.environ.get("MICRODUCK_POLICIES",
                                                             "../microduck/policies"),
                        help="Directory of ONNX policies (microduck repo's policies/)")
    parser.add_argument("--ffmpeg", default=os.environ.get("DUCK_FFMPEG", "ffmpeg"),
                        help="ffmpeg binary to encode with (default: ffmpeg on PATH)")
    parser.add_argument("--no-cards", action="store_true",
                        help="drop the title and credit cards; match footage only")
    parser.add_argument("--no-audio", action="store_true",
                        help="film silently: no voice, no wheee, no chirps")
    parser.add_argument("--voice-bank", default=os.environ.get("DUCK_VOICE_BANK"),
                        metavar="DIR",
                        help="voice-bank wavs from the microduck `sounds` "
                             "crate (chirp*.wav, wheee.wav); without it the "
                             "shoot renders its own from the crate beside "
                             "--policies, and films chirpless if it cannot")
    parser.add_argument("--tts-voice", default=_soundtrack().DEFAULT_TTS_VOICE,
                        metavar="NAME",
                        help="TTS voice the duck's own voice is built from")
    parser.add_argument("--line-arm", default=None, metavar="TEXT",
                        help="what the duck says as the machine arms "
                             "(empty string: say nothing)")
    parser.add_argument("--line-goal", default=None, metavar="TEXT",
                        help="what the duck says after a goal (empty string: "
                             "let the wheee speak for itself)")
    parser.add_argument("--keep-takes", action="store_true",
                        help="keep every scoring take next to the output "
                             "instead of only the chosen one")
    parser.add_argument("--quiet", action="store_true",
                        help="only print the path of the finished film")


def script_lines(args) -> dict:
    """The film's script: the defaults, with anything the caller overrode.

    An explicitly empty line is a deletion, not a default — `--line-arm ""`
    means the film opens without a word.
    """
    lines = dict(_soundtrack().DEFAULT_LINES)
    for beat in ("arm", "goal"):
        override = getattr(args, f"line_{beat}", None)
        if override is not None:
            lines[beat] = override
    return lines


def build_kit(args, ffmpeg: str, work_dir: str):
    """Render the shoot's sound, or return None and say what went wrong.

    Zero-config: with no --voice-bank the shoot renders its own from the
    `sounds` crate in the repo that --policies already points into. Cargo
    missing, crate not building, no TTS on this platform — each is a note on
    stderr and a quieter film, never a failed shoot.
    """
    if args.no_audio:
        return None
    st = _soundtrack()
    try:
        bank = args.voice_bank
        if not bank:
            repo = st.find_sounds_repo(args.policies)
            if repo is not None:
                if not args.quiet:
                    print(f"rendering the voice bank from {repo} "
                          f"(first build can take a minute)", flush=True)
                bank = st.render_bank(repo, os.path.join(work_dir, "bank"))
            if not bank:
                print("note: no voice bank — pass --voice-bank DIR, or point "
                      "--policies into a microduck checkout with the `sounds` "
                      "crate", file=sys.stderr)
        kit = st.SoundKit(script_lines(args), ffmpeg, bank, args.tts_voice)
    except Exception as e:
        print(f"note: no sound this shoot ({e}) — filming silently",
              file=sys.stderr)
        return None
    for note in kit.notes:
        print(f"note: {note}", file=sys.stderr)
    return kit if kit.audible else None


def run(args) -> int:
    """`duck film`. Returns a process exit code."""
    try:
        ffmpeg = find_ffmpeg(args.ffmpeg)
        machine_path = resolve_machine(args.machine)
        if not 1 <= args.takes <= len(MATCH_SPAWNS):
            raise FilmError(f"--takes must be 1..{len(MATCH_SPAWNS)} "
                            f"(one per known spawn); got {args.takes}")
    except FilmError as e:
        print(f"duck film: {e}", file=sys.stderr)
        return 1

    frames_dir = tempfile.mkdtemp(prefix="duck-film-")
    shoot = None
    takes = []
    try:
        # Before the model loads, like the ffmpeg preflight: rendering the
        # voice is the one other slow thing that can degrade, and the shoot
        # should say so up front rather than after a two-minute match.
        kit = build_kit(args, ffmpeg, frames_dir)
        sim = ss.DuckSim(args.rl_repo, args.policies, args.scene, frames_dir)
        if sim.referee is None or sim.policy.ball_qpos_adr is None:
            raise FilmError(f"scene {args.scene!r} has no goal to score in — "
                            "film needs --scene pitch")
        shoot = MatchFilm(sim, ffmpeg, machine_path, machine_label=args.machine,
                          cards=not args.no_cards, quiet=args.quiet, kit=kit)
        for index, spawn in enumerate(MATCH_SPAWNS[:args.takes]):
            take = shoot.film_take(
                index, spawn, take_path(args.output, index, args.keep_takes),
                cap_s=args.cap_seconds)
            takes.append(take)
            if classify_take(take, args.select == "won") == KEEP:
                break
    except FilmError as e:
        print(f"duck film: {e}", file=sys.stderr)
        return 1
    except (FileNotFoundError, OSError) as e:
        print(f"duck film: {e}", file=sys.stderr)
        return 1
    finally:
        if shoot is not None:
            shoot.close()
        shutil.rmtree(frames_dir, ignore_errors=True)

    chosen = select_take(takes, args.select == "won")
    for take in takes:
        if take.scored and take is not chosen and not args.keep_takes:
            os.unlink(take.path)
    if chosen is None:
        print("duck film: no take scored — nothing to cut. Try --takes 5, or "
              "a longer --cap-seconds.", file=sys.stderr)
        return 1
    if args.keep_takes:
        shutil.copyfile(chosen.path, args.output)
    else:
        os.replace(chosen.path, args.output)
    if not args.quiet:
        verdict = ("goal + landed celebration" if chosen.node == "won"
                   else f"goal, ended in {chosen.node}")
        print(f"kept take {chosen.index} ({verdict})")
    print(args.output)
    return 0
