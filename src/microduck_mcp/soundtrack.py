"""The film's soundtrack: the duck's own voice, cut to the match's timeline.

`duck film` already records what the robot knew; this records what it had to
say about it. The track is not scored to the picture by ear — it is built from
the take's OWN event timeline, the same sim clock the frames are sampled on:

    beats (sim time)        cues                    segments (video time)
    ----------------        ----                    ---------------------
    arm      t=0.0    -->   speak "..."       -->   wav at 1.00 s
    -> kick  t=31.2   -->   chirp accent      -->   chirp at 32.2 s
    GOAL!    t=44.9   -->   WHEEE             -->   wheee at 45.9 s
                            speak "..."       -->   wav at 47.8 s

A beat is something that actually happened in the take (the machine armed, a
node entered, the referee's call). A cue is what to play about it. Cues are
planned by a pure function — `plan_cues` — which is where the one hard rule of
this film's sound design lives:

    THE WHEEE IS RESERVED FOR EARNED GOALS.

Not the arm, not a kick, not the cold open, not decoration: it fires once, at
the moment the referee says the ball crossed the line, or it does not fire at
all. That rule is a test, not a habit (see tests/test_soundtrack.py).

Speech comes from voice.py — the ratified pipeline, unchanged: macOS `say`,
pitched into the duck's own synthesized personality, chirp grains blended into
the stressed syllables. There is no music bed and no choir; every sound in the
film is either the duck talking or the duck's own voice bank. The voice is
honestly synthetic and never pretends to be a person.

Mixing is done here in numpy rather than in an ffmpeg filter graph: placing a
wav at a sample offset is exact, needs no `adelay`/`amix` incantation whose
failure modes are silent, and is testable without an encoder. ffmpeg is used
for one thing at the end — muxing the finished track onto the finished video
with `-c:v copy`, so the picture is never re-encoded for the sake of sound.

Everything here degrades instead of failing. No TTS, no voice bank, no cargo,
a filter that errors out: the film still gets cut, silently, with a note on
stderr. Audio is an enhancement to `duck film`, never a new way for it to die.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from typing import NamedTuple

import numpy as np

from . import voice
from .voice import SR

# Beat kinds — what happened in the take.
ARM, GOAL, NODE = "arm", "goal", "node"
# Cue kinds — what to play about it.
SPEAK, WHEEE, CHIRP = "speak", "wheee", "chirp"

# The bank's `wheee` is a sustained ~4.8 s call, tuned for a robot announcing
# itself across a room. The film wants a sting: the first WHEEE_MAX_S of it,
# faded, so the celebration line has air to land in. An edit of the asset's
# length, not of the rule about when it may play.
WHEEE_MAX_S = 1.6
WHEEE_FADE_S = 0.25

# Levels. Speech carries the film; the wheee is an exclamation under it; the
# chirp accents are punctuation and must never compete with either.
SPEECH_GAIN = 1.0
WHEEE_GAIN = 0.55
CHIRP_GAIN = 0.35
PEAK_CEILING = 0.97

# The celebration line waits for the wheee to clear — the duck yells first and
# talks second, which is the correct order for a goal.
GOAL_LINE_DELAY_S = 1.9
# Chirp accents on the kick swing. Capped: a striker that needs six attempts
# should not sound like a smoke alarm.
MAX_KICK_CHIRPS = 3

# Zero-config script. Short, charming, and true to what the machine is doing:
# it hunts the ball first and aims at the goal second, and the aiming is the
# achievement worth bragging about.
DEFAULT_LINES = {
    "arm": "Right. Ball first, then the goal.",
    "goal": "Goal! I aimed that one myself.",
}

# The voice bank, rendered on demand by the microduck `sounds` crate.
BANK_SEED = 42
BANK_VARIANTS = (1, 2, 3, 4)
CARGO_TIMEOUT_S = 300


class Beat(NamedTuple):
    """Something that happened in the take, at take-relative sim time."""
    kind: str
    t_s: float
    detail: str = ""


class Cue(NamedTuple):
    """Something to play about a beat, at take-relative sim time."""
    t_s: float
    kind: str
    text: str = ""


class Voicing(NamedTuple):
    """A rendered line: the audio and the beak trajectory that matches it.

    One render, two consumers — the mix and the duck's own mouth — so the beak
    in the picture cannot drift against the voice on the track.
    """
    text: str
    samples: np.ndarray
    mouth: np.ndarray
    duration_s: float


MOUTH_RATE_HZ = voice.MOUTH_RATE_HZ
DEFAULT_TTS_VOICE = voice.DEFAULT_TTS_VOICE


def mouth_at(speaking, t_s: float, rate_hz: int = MOUTH_RATE_HZ) -> float | None:
    """Beak opening at take time `t_s`, or None if nobody is talking.

    `speaking` is [(start_s, Voicing)]. None and 0.0 are different answers: a
    shut beak mid-sentence is a closed plosive, silence between lines is not
    speech at all — the HUD draws the first and hides the second.
    """
    opening = None
    for start, v in speaking:
        i = int((t_s - start) * rate_hz)
        if 0 <= i < len(v.mouth):
            opening = max(opening or 0.0, float(v.mouth[i]))
    return opening


# ---------- the plan (pure: beats in, cues out) ----------

def plan_cues(beats, lines=None, goal_line_delay_s: float = GOAL_LINE_DELAY_S,
              max_kick_chirps: int = MAX_KICK_CHIRPS) -> list:
    """The take's beats -> the sounds to play, in take-relative sim time.

    The whole sound design, in one readable function:

      * the machine arming opens the film with a line;
      * entering `kick` gets a chirp accent, capped — effort, not commentary;
      * the referee's goal gets the WHEEE, once, and only here, followed by the
        celebration line once the wheee has cleared.

    An empty line string is a line the caller deleted: no cue, no silence
    where a cue would have been.
    """
    lines = DEFAULT_LINES if lines is None else lines
    cues = []
    scored = False
    chirps = 0
    for beat in beats:
        if beat.kind == ARM:
            text = (lines.get("arm") or "").strip()
            if text:
                cues.append(Cue(beat.t_s, SPEAK, text))
        elif beat.kind == GOAL:
            if scored:
                continue        # one goal, one wheee: a latch, not a counter
            scored = True
            cues.append(Cue(beat.t_s, WHEEE))
            text = (lines.get("goal") or "").strip()
            if text:
                cues.append(Cue(beat.t_s + goal_line_delay_s, SPEAK, text))
        elif beat.kind == NODE and beat.detail == "kick":
            if chirps < max_kick_chirps:
                chirps += 1
                cues.append(Cue(beat.t_s, CHIRP))
    return sorted(cues, key=lambda c: (c.t_s, c.kind))


# ---------- the mix ----------

def trim_sting(x: np.ndarray, max_s: float = WHEEE_MAX_S,
               fade_s: float = WHEEE_FADE_S) -> np.ndarray:
    """A long sustained call, cut down to a sting with a fade on the tail."""
    n = int(max_s * SR)
    if len(x) <= n:
        return x
    out = x[:n].astype(np.float32).copy()
    fade = min(int(fade_s * SR), len(out))
    out[len(out) - fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    return out


def mix(segments, total_s: float, sr: int = SR) -> np.ndarray:
    """Place gained segments at their timestamps on one `total_s` bed.

    Segments landing past the end are truncated (the film stops when it stops);
    one landing before t=0 keeps only the part that is still in the film. The
    sum is scaled down if it would clip, never up — a quiet film is a film, a
    clipped one is a fault.
    """
    out = np.zeros(max(1, int(round(total_s * sr))), dtype=np.float32)
    for t_s, samples, gain in segments:
        start = int(round(t_s * sr))
        if start >= len(out) or not len(samples):
            continue
        seg = np.asarray(samples, dtype=np.float32)
        if start < 0:
            seg = seg[-start:]
            start = 0
        seg = seg[: len(out) - start]
        if len(seg):
            out[start: start + len(seg)] += gain * seg
    peak = float(np.abs(out).max()) if len(out) else 0.0
    if peak > PEAK_CEILING:
        out *= PEAK_CEILING / peak
    return out


# ---------- the voice bank ----------

def find_sounds_repo(policies_dir: str) -> str | None:
    """The microduck repo holding the `sounds` crate, inferred from --policies.

    `--policies .../microduck/policies` already points inside the repo whose
    crate renders the bank, so zero-config filming can find the duck's voice
    without a second path to spell out.
    """
    env = os.environ.get("MICRODUCK_SOUNDS_REPO")
    if env:
        return env if os.path.isfile(os.path.join(env, "Cargo.toml")) else None
    repo = os.path.dirname(os.path.abspath(policies_dir))
    return repo if os.path.isfile(os.path.join(repo, "Cargo.toml")) else None


def render_bank(sounds_repo: str, out_dir: str,
                timeout_s: int = CARGO_TIMEOUT_S) -> str | None:
    """Render a voice bank with the `sounds` crate. None if it cannot be done.

    Never raises: a missing cargo, a crate that does not build, a render that
    hangs — all of them mean "film without chirps", not "no film".
    """
    if shutil.which("cargo") is None:
        return None
    os.makedirs(out_dir, exist_ok=True)
    jobs = [("chirp", f"chirp{v}.wav", ["--variant", str(v)])
            for v in BANK_VARIANTS] + [("wheee", "wheee.wav", [])]
    for tag, name, extra in jobs:
        cmd = ["cargo", "run", "-q", "-p", "sounds", "--", "render", tag,
               os.path.join(out_dir, name), "--seed", str(BANK_SEED)] + extra
        try:
            done = subprocess.run(cmd, cwd=sounds_repo, timeout=timeout_s,
                                  capture_output=True)
        except (OSError, subprocess.SubprocessError):
            return None
        if done.returncode != 0:
            return None
    return out_dir


def _bank_wav(bank_dir: str | None, *names) -> np.ndarray | None:
    """First readable wav among `names` in the bank, or None."""
    if not bank_dir:
        return None
    for name in names:
        path = os.path.join(bank_dir, name)
        if os.path.isfile(path):
            try:
                return voice.load_wav48(path)
            except Exception:   # wrong rate, truncated, not a RIFF at all —
                return None     # a bank wav that will not load is a quiet film
    return None


# ---------- the kit ----------

class SoundKit:
    """Everything the shoot can make noise with: rendered lines and bank sounds.

    Built once per shoot (the lines are the same in every take, so the TTS cost
    is paid once), and built defensively: any stage that fails leaves its slot
    empty and adds a note, so the caller can film on regardless. `notes` is the
    honest account of what the audience will not be hearing and why.
    """

    def __init__(self, lines: dict, ffmpeg: str, bank_dir: str | None = None,
                 tts_voice: str = voice.DEFAULT_TTS_VOICE):
        self.lines = dict(DEFAULT_LINES if lines is None else lines)
        self.ffmpeg = ffmpeg
        self.bank_dir = bank_dir
        self.notes = []
        self.voicings = {}
        for beat, text in self.lines.items():
            text = (text or "").strip()
            if not text:
                continue
            # Only lines that actually rendered go in: an empty slot is what
            # `audible` and the beak animation both read as "no line here".
            voicing = self._render_line(text, bank_dir, tts_voice)
            if voicing is not None:
                self.voicings[beat] = voicing
        full = _bank_wav(bank_dir, "wheee.wav")
        self.wheee = trim_sting(full) if full is not None else None
        # A different variant than the one blended into the speech (voice.py
        # takes the first chirp*.wav): a standalone accent should not sound
        # like a syllable that wandered off.
        self.chirp = _bank_wav(bank_dir, "chirp2.wav", "chirp1.wav")
        if bank_dir and self.wheee is None:
            self.notes.append(f"no wheee*.wav in {bank_dir} — the goal will "
                              f"land without one")
        if not bank_dir:
            self.notes.append("no voice bank — chirpless voice, and no wheee")

    def _render_line(self, text: str, bank_dir, tts_voice) -> Voicing | None:
        tmp = os.path.join(tempfile.gettempdir(),
                           f"duck-film-line-{os.getpid()}-{abs(hash(text))}.wav")
        try:
            wav, traj, duration = voice.render_voice(
                text, self.ffmpeg, voice=tts_voice, bank_dir=bank_dir,
                out_wav=tmp)
            samples = voice.load_wav48(wav)
        except Exception as e:      # TTS, ffmpeg filters, a wav that will not
            self.notes.append(f"no voice for {text!r}: {e}")   # load — all the
            return None                                        # same verdict
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        return Voicing(text, samples, traj, duration)

    @property
    def audible(self) -> bool:
        """Is there anything at all to put on the track?"""
        return bool(self.voicings) or self.wheee is not None \
            or self.chirp is not None

    def voicing(self, text: str) -> Voicing | None:
        for v in self.voicings.values():
            if v is not None and v.text == text:
                return v
        return None

    def segments(self, cues, at=None):
        """Cues -> (time, samples, gain), dropping whatever did not render.

        `at` maps a take-relative sim time to a time on the finished film's
        clock (the cut adds a cold open, so the two are not the same); without
        it the cues stay on sim time, which is what the tests read.
        """
        at = (lambda t: t) if at is None else at
        out = []
        for cue in cues:
            if cue.kind == SPEAK:
                v = self.voicing(cue.text)
                if v is not None:
                    out.append((at(cue.t_s), v.samples, SPEECH_GAIN))
            elif cue.kind == WHEEE and self.wheee is not None:
                out.append((at(cue.t_s), self.wheee, WHEEE_GAIN))
            elif cue.kind == CHIRP and self.chirp is not None:
                out.append((at(cue.t_s), self.chirp, CHIRP_GAIN))
        return out

    def track(self, cues, total_s: float, at=None) -> np.ndarray:
        return mix(self.segments(cues, at), total_s)


# ---------- muxing ----------

def mux(video_path: str, track: np.ndarray, ffmpeg: str,
        out_path: str | None = None) -> str:
    """Lay the track onto the cut video. The picture is copied, never re-encoded.

    Writes beside the video and moves into place, so a failed mux leaves the
    silent film exactly as it was.
    """
    out_path = out_path or video_path
    tmp_wav = video_path + ".track.wav"
    tmp_mp4 = video_path + ".sound.mp4"
    try:
        voice.save_wav48(tmp_wav, track)
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", video_path,
                        "-i", tmp_wav, "-map", "0:v:0", "-map", "1:a:0",
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                        "-shortest", "-movflags", "+faststart", tmp_mp4],
                       check=True)
        os.replace(tmp_mp4, out_path)
    finally:
        for path in (tmp_wav, tmp_mp4):
            if os.path.exists(path):
                os.unlink(path)
    return out_path


def scored_note(cues) -> str:
    """One line for the shoot log: what the audience will hear, and when."""
    if not cues:
        return "silent"
    parts = []
    for cue in cues:
        what = cue.kind if cue.kind != SPEAK else f"say {cue.text!r}"
        parts.append(f"{cue.t_s:.1f}s {what}")
    return "  ·  ".join(parts)
