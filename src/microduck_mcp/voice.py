r"""The duck's speaking voice: `duck say` — words out the speakers, beak in sync.

Pipeline (offline render, then a timed performance):

    text --say--> speech --ffmpeg--> duck-modulated 48 kHz wav
         --envelope--> syllable nuclei --> chirp grains blended INTO the
                                           stressed syllables
                   \--> mouth trajectory (the SAME envelope, resampled with a
                        fast attack and a slower release)

then the wav plays host-side (afplay — the sim has no speaker) while the mouth
trajectory streams to the running sim as `mouth` intents against the playback
clock. One envelope drives both the chirp placement and the beak, which is
what keeps them honest to each other: the beak opens exactly where the voice
leans.

`duck chirp` is the same performance with the render stage cut out: a tag from
the voice bank (alarm, greet, inquire, peck, chirp, coo, wheee) played straight,
the beak driven by that call's own envelope. The bank is the duck's OWN
vocabulary — the words are the borrowed part — so a chirp is the honest
reaction and a sentence is the translation.

The duckness lives IN the voice, not under it: the speech is pitched up and
run through the modulation parameters of the duck's own synthesized
personality (see the constants below), and the chirps are grains of a real
voice-bank chirp riding the syllables the enunciation already hits hardest —
an accent, not punctuation. TTS is a stand-in boundary: macOS `say` today, a
phoneme-timed engine (e.g. Piper) later.
"""

import argparse
import glob
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave

import numpy as np

from .film import FilmError, find_ffmpeg

SR = 48000

# Duck modulation, from the robot's own voice synthesis: seed 42's Personality
# (`sounds show --seed 42` in pollen-robotics/microduck) has vibrato at 8.6 Hz
# and amplitude modulation at 25.9 Hz; depths here are scaled for speech (the
# synth's full depths are tuned for calls, not words). Pitch-up ~2 semitones
# (54240/48000) sits the voice above an adult human without chipmunking it.
# Recipe ratified as the v4 POC (2026-08-28).
PITCH_ASETRATE = 54240
PITCH_TEMPO = 0.885  # duration compensation for the asetrate pitch-up
VIBRATO_HZ, VIBRATO_DEPTH = 8.6, 0.10
TREMOLO_HZ, TREMOLO_DEPTH = 25.9, 0.18

# Chirp blending (the v4 POC numbers): 90 ms grains at gain 0.75, placed on
# the syllables with the sharpest attack, each grain shaped by the word's own
# envelope so it inherits the articulation instead of interrupting it.
GRAIN_S = 0.09
GRAIN_GAIN = 0.75
GRAIN_PRE_S = 0.03
GRAIN_FADE_S = 0.012

# Envelope / mouth. The envelope runs at 1 kHz (block-averaged |x|, then a
# 25 Hz one-pole); the mouth resamples it to MOUTH_RATE_HZ with an asymmetric
# smoother — beaks snap open and ease shut. The floor keeps breath noise from
# fluttering the beak, and silence closes it completely.
ENV_RATE = 1000
ENV_CUTOFF_HZ = 25.0
MOUTH_RATE_HZ = 40
MOUTH_ATTACK_S = 0.015
MOUTH_RELEASE_S = 0.090
MOUTH_FLOOR = 0.10

DEFAULT_TTS_VOICE = "Samantha"
MAX_SAY_CHARS = 400


class VoiceError(Exception):
    """A problem worth a clean sentence, not a stack trace."""


# ---------- wav I/O ----------

def load_wav48(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        if w.getframerate() != SR:
            raise VoiceError(f"{path}: expected {SR} Hz, got {w.getframerate()}")
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        x = x.astype(np.float32)
        if w.getnchannels() == 2:
            x = x.reshape(-1, 2).mean(axis=1)
        return x / 32768.0


def save_wav48(path: str, x: np.ndarray):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())


# ---------- the render pipeline ----------

def find_say() -> str:
    """The TTS boundary. macOS `say` for now; swap here for Piper et al."""
    path = shutil.which("say")
    if path is None:
        raise VoiceError(
            "no `say` binary — the TTS stage currently uses macOS text-to-"
            "speech. On another OS, render a wav elsewhere or wire in a TTS "
            "engine at voice.find_say/tts_render.")
    return path


def tts_render(text: str, out_path: str, voice: str = DEFAULT_TTS_VOICE):
    subprocess.run([find_say(), "-v", voice, "-o", out_path, text], check=True)


def duck_modulate(src: str, dst_wav: str, ffmpeg: str):
    """Pitch the speech up and run it through the personality's modulation."""
    chain = (f"aresample={SR},asetrate={PITCH_ASETRATE},aresample={SR},"
             f"atempo={PITCH_TEMPO},"
             f"vibrato=f={VIBRATO_HZ}:d={VIBRATO_DEPTH},"
             f"tremolo=f={TREMOLO_HZ}:d={TREMOLO_DEPTH},"
             f"aformat=sample_rates={SR}:channel_layouts=mono")
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", src,
                    "-af", chain, dst_wav], check=True)


def envelope(x: np.ndarray) -> np.ndarray:
    """Loudness envelope at ENV_RATE Hz: block-averaged |x|, 25 Hz one-pole.

    The same signal a live mouth-driver would compute chunk by chunk — and the
    one signal both the chirp placement and the beak trajectory read.
    """
    block = SR // ENV_RATE
    n = len(x) // block * block
    coarse = np.abs(x[:n]).reshape(-1, block).mean(axis=1)
    a = 1.0 - math.exp(-2 * math.pi * ENV_CUTOFF_HZ / ENV_RATE)
    env = np.empty_like(coarse)
    acc = 0.0
    for i, v in enumerate(coarse):
        acc += a * (v - acc)
        env[i] = acc
    return env


def syllable_peaks(env: np.ndarray, min_gap_s: float = 0.18,
                   floor: float = 0.35, attack_win_s: float = 0.06) -> list:
    """Stressed-syllable nuclei: local envelope maxima with a sharp attack.

    Returns [(env_index, amplitude, attack)] for the syllables leaned on
    hardest — the top half by attack ramp, at least two. Deterministic.
    """
    if not len(env) or env.max() <= 0.0:
        return []
    gap = int(min_gap_s * ENV_RATE)
    win = int(attack_win_s * ENV_RATE)
    thresh = floor * env.max()
    peaks = []
    i = win
    while i < len(env) - 1:
        if env[i] > thresh and env[i] >= env[i - 1] and env[i] > env[i + 1]:
            peaks.append((i, float(env[i]), float(env[i] - env[i - win])))
            i += gap
        else:
            i += 1
    peaks.sort(key=lambda p: -p[2])
    return sorted(peaks[: max(2, len(peaks) // 2)])


def bank_wavs(bank_dir: str | None, tag: str) -> list:
    """Every `<tag>*.wav` in the bank, sorted — the tag's variants.

    Sorted, so a tag names the same wav on every host and every run: the bank
    is content, and content the duck reacts with must not depend on readdir
    order. Empty list for no bank and for a bank without that tag; the callers
    decide which of those is fatal.
    """
    if not bank_dir or not os.path.isdir(bank_dir):
        return []
    return sorted(glob.glob(os.path.join(bank_dir, f"{tag}*.wav")))


def bank_wav_path(bank_dir: str | None, tag: str, variant: int = 0) -> str:
    """The wav a `<tag>[, variant]` names, or a VoiceError saying why not."""
    if not tag or not tag.isidentifier():
        raise VoiceError(f"{tag!r} is not a voice-bank tag (try alarm, greet, "
                         "inquire, peck, chirp, coo, wheee)")
    if not bank_dir:
        raise VoiceError(
            "no voice bank — pass --voice-bank DIR (or set DUCK_VOICE_BANK) "
            "pointing at wavs rendered by the microduck `sounds` crate")
    hits = bank_wavs(bank_dir, tag)
    if not hits:
        raise VoiceError(f"no {tag}*.wav in {bank_dir} — render it with the "
                         f"`sounds` crate, or pick a tag the bank has")
    if not 0 <= variant < len(hits):
        raise VoiceError(f"{tag} has {len(hits)} variant(s) in {bank_dir}; "
                         f"there is no variant {variant}")
    return hits[variant]


def load_chirp(bank_dir: str | None) -> np.ndarray | None:
    """First chirp wav from a voice bank rendered by the `sounds` crate.

    No bank, no chirps: the voice degrades to unblended speech with a note,
    because a duck that cannot chirp should still be able to talk.
    """
    if not bank_dir:
        return None
    hits = bank_wavs(bank_dir, "chirp")
    if not hits:
        print(f"note: no chirp*.wav in {bank_dir} — speaking without chirps",
              file=sys.stderr)
        return None
    return load_wav48(hits[0])


def blend_chirps(speech: np.ndarray, chirp: np.ndarray | None,
                 env: np.ndarray) -> tuple[np.ndarray, list]:
    """Ride chirp grains on the stressed syllables. Returns (audio, hits)."""
    hits = syllable_peaks(env)
    if chirp is None or not hits:
        return speech, hits if chirp is not None else []
    grain = chirp[: int(GRAIN_S * SR)].copy()
    fade = int(GRAIN_FADE_S * SR)
    grain[:fade] *= np.linspace(0, 1, fade)
    grain[-fade:] *= np.linspace(1, 0, fade)
    peak = float(np.abs(grain).max()) or 1.0
    out = speech.copy()
    block = SR // ENV_RATE
    for ei, amp, _ in hits:
        start = max(0, ei * block - int(GRAIN_PRE_S * SR))
        seg = out[start: start + len(grain)]
        # local envelope, upsampled to shape the grain by the word itself
        idx = (start + np.arange(len(seg))) / block
        shape = np.interp(idx, np.arange(len(env)), env) / (amp or 1.0)
        seg += grain[: len(seg)] * (GRAIN_GAIN * amp / peak) * shape
    return out, hits


def mouth_trajectory(env: np.ndarray, rate_hz: int = MOUTH_RATE_HZ) -> np.ndarray:
    """Beak openings in [0, 1] at rate_hz, from the shared envelope.

    Fast attack, slower release (a beak snaps open and eases shut), a noise
    floor so breath does not flutter it, normalized per utterance, and a
    guaranteed 0.0 on silence.
    """
    if not len(env) or env.max() <= 0.0:
        return np.zeros(0, dtype=np.float32)
    aa = 1.0 - math.exp(-1.0 / (MOUTH_ATTACK_S * ENV_RATE))
    ar = 1.0 - math.exp(-1.0 / (MOUTH_RELEASE_S * ENV_RATE))
    sm = np.empty_like(env)
    acc = 0.0
    for i, v in enumerate(env):
        acc += (aa if v > acc else ar) * (v - acc)
        sm[i] = acc
    floor = MOUTH_FLOOR * sm.max()
    span = sm.max() - floor
    opening = np.clip((sm - floor) / span, 0.0, 1.0) if span > 0 else np.zeros_like(sm)
    step = ENV_RATE // rate_hz
    return opening[::step].astype(np.float32)


def render_voice(text: str, ffmpeg: str, voice: str = DEFAULT_TTS_VOICE,
                 bank_dir: str | None = None, out_wav: str | None = None):
    """text -> (wav_path, mouth trajectory, duration_s). The whole render."""
    if not text.strip():
        raise VoiceError("nothing to say")
    if len(text) > MAX_SAY_CHARS:
        raise VoiceError(f"text too long ({len(text)} chars, max {MAX_SAY_CHARS})")
    chirp = load_chirp(bank_dir)
    with tempfile.TemporaryDirectory(prefix="duck-say-") as tmp:
        raw = os.path.join(tmp, "speech.aiff")
        mod = os.path.join(tmp, "speech48.wav")
        tts_render(text, raw, voice=voice)
        duck_modulate(raw, mod, ffmpeg)
        speech = load_wav48(mod)
    env = envelope(speech)
    audio, hits = blend_chirps(speech, chirp, env)
    if hits:
        print(f"chirps on {len(hits)} syllables", file=sys.stderr)
    traj = mouth_trajectory(env)
    wav_path = out_wav or os.path.join(
        tempfile.gettempdir(), f"duck_say_{os.getpid()}.wav")
    save_wav48(wav_path, audio)
    return wav_path, traj, len(audio) / SR


# ---------- the performance ----------

def perform(wav_path: str, traj: np.ndarray, annotation: dict,
            sock_path: str | None = None, play_audio: bool = True,
            rate_hz: int = MOUTH_RATE_HZ) -> None:
    """Play a wav host-side while streaming the beak to the sim.

    The shared performance behind both of the duck's voices: `duck say` hands
    it a rendered sentence, `duck chirp` a call straight out of the bank. One
    shared t0 and absolute per-sample deadlines (the sim loop's own pacing
    rule): drift never accumulates, and the beak cannot slide against the
    audio because both run off the same clock.

    `annotation` is the intent that puts the act on the control surface before
    a sound is made — and the server's chance to REFUSE it (the wheee is
    reserved for goals the duck actually scored), which is why its reply is
    read rather than merely drained: a refused sound is not played.
    """
    from .client import DEFAULT_SOCKET
    import json
    import socket as socketlib

    client = annotation.get("client", "say")
    sock = socketlib.socket(socketlib.AF_UNIX, socketlib.SOCK_STREAM)
    sock.settimeout(5.0)
    try:
        sock.connect(sock_path or DEFAULT_SOCKET)
    except (ConnectionRefusedError, FileNotFoundError) as e:
        sock.close()
        raise VoiceError(
            f"sim server not running (socket {sock_path or DEFAULT_SOCKET}) — "
            "start it with `duck-sim` (mjpython + --viewer to watch the beak), "
            "or pass --audio-only to just hear the voice") from e
    f = sock.makefile("rwb")

    def send(req: dict) -> dict:
        f.write((json.dumps(req) + "\n").encode())
        f.flush()
        line = f.readline()  # ack; keep the pipe drained
        try:
            return json.loads(line)
        except (ValueError, TypeError):
            return {"ok": True}

    player = None
    try:
        resp = send(annotation)
        if not resp.get("ok"):
            raise VoiceError(resp.get("error", "the sim refused the sound"))
        if play_audio:
            afplay = shutil.which("afplay")
            if afplay is None:
                raise VoiceError("no `afplay` — use --audio-only elsewhere, "
                                 "or play the --wav-out file yourself")
            player = subprocess.Popen([afplay, wav_path])
        t0 = time.perf_counter()
        for i, opening in enumerate(traj):
            deadline = t0 + i / rate_hz
            now = time.perf_counter()
            if deadline > now:
                time.sleep(deadline - now)
            send({"cmd": "mouth", "client": client, "opening": float(opening)})
    finally:
        try:
            send({"cmd": "mouth", "client": client, "opening": 0.0})
        finally:
            f.close()
            sock.close()
        if player is not None:
            player.wait()


def speak(wav_path: str, traj: np.ndarray, text: str, duration_s: float,
          sock_path: str | None = None, play_audio: bool = True,
          rate_hz: int = MOUTH_RATE_HZ) -> None:
    """`duck say`'s performance: the rendered line, annotated as speech."""
    perform(wav_path, traj,
            {"cmd": "say", "client": "say", "text": text,
             "duration_s": round(duration_s, 2)},
            sock_path=sock_path, play_audio=play_audio, rate_hz=rate_hz)


# ---------- the nonverbal voice ----------

def chirp_render(bank_dir: str | None, tag: str, variant: int = 0):
    """(wav path, mouth trajectory) for one call out of the voice bank.

    The same two signals speech produces, one stage shorter: a bank wav is
    already the duck's own voice at 48 kHz, so there is nothing to synthesize
    and nothing to modulate. The beak still comes from the call's own
    envelope — a duck that chirped with its beak shut would be a speaker.
    """
    path = bank_wav_path(bank_dir, tag, variant)
    return path, mouth_trajectory(envelope(load_wav48(path)))


def add_chirp_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("tag", help="voice-bank tag: alarm, greet, inquire, "
                        "peck, chirp, coo — plus wheee, which the sim grants "
                        "only when the referee has a goal on the board")
    parser.add_argument("--variant", type=int, default=0, metavar="N",
                        help="which wav when the bank holds several for the "
                        "tag (sorted; default the first)")
    parser.add_argument("--voice-bank", default=os.environ.get("DUCK_VOICE_BANK"),
                        metavar="DIR",
                        help="directory of voice-bank wavs rendered by the "
                        "microduck `sounds` crate")
    parser.add_argument("--audio-only", action="store_true",
                        help="skip the sim: just play the call")


def run_chirp(args) -> int:
    """`duck chirp`. Returns a process exit code."""
    try:
        wav, traj = chirp_render(args.voice_bank, args.tag, args.variant)
        if args.audio_only:
            afplay = shutil.which("afplay")
            if afplay is None:
                raise VoiceError(f"no `afplay`; the call is at {wav}")
            subprocess.run([afplay, wav], check=True)
        else:
            perform(wav, traj,
                    {"cmd": "chirp", "client": "chirp", "tag": args.tag,
                     "variant": args.variant},
                    sock_path=getattr(args, "socket", None))
    except VoiceError as e:
        print(f"duck chirp: {e}", file=sys.stderr)
        return 1
    return 0


# ---------- the machine's own voice ----------

class SayPlayer:
    """The duck talking on its own initiative, off the sim thread.

    `duck say` is a performance a person asks for and waits through. This is
    the same pipeline for lines the behavior machine decides to say while
    nobody is asking: entering a node with a `say` annotation hands the line
    here, and the 50 Hz control loop carries on within microseconds — the TTS
    render alone takes about a second, which is fifty control steps the duck
    is not allowed to spend on talking.

    One line at a time. A line arriving while another is playing is DROPPED,
    not queued: a duck that talks over itself sounds broken, and a backlog of
    stale celebrations played after the fact is worse than silence.

    The only thing the worker touches back in the sim is `mouth_opening` — a
    mocap plate with no collision geometry and no dofs, exactly what the
    socket `mouth` intent writes. The policy's observation never sees it, so a
    talking duck and a silent one step identical physics.
    """

    def __init__(self, ffmpeg: str, bank_dir: str | None = None,
                 tts_voice: str = DEFAULT_TTS_VOICE, player: str = None):
        self.ffmpeg = ffmpeg
        self.bank_dir = bank_dir
        self.tts_voice = tts_voice
        self.player = player or shutil.which("afplay")
        self._busy = threading.Lock()

    @classmethod
    def available(cls, ffmpeg: str = "ffmpeg", bank_dir: str | None = None,
                  tts_voice: str = DEFAULT_TTS_VOICE):
        """A player, or None with a reason on stderr. Never raises.

        A robot that cannot speak is a robot, not an error: every missing
        piece of the toolchain means the machine's lines stay on the control
        surface and out of the air.
        """
        try:
            resolved = find_ffmpeg(ffmpeg)
            find_say()
            if shutil.which("afplay") is None:
                raise VoiceError("no `afplay` to play the rendered voice")
        except (FilmError, VoiceError) as e:
            print(f"note: the duck has no voice this session ({e}) — machine "
                  f"`say` lines will still show on the control surface",
                  file=sys.stderr)
            return None
        return cls(resolved, bank_dir=bank_dir, tts_voice=tts_voice)

    @property
    def busy(self) -> bool:
        """Is a line playing right now? The mouth's ownership question: while
        this is True the beak belongs to the words, and an emote gesturing at
        the same time leaves it alone."""
        return self._busy.locked()

    def speak(self, text: str, sim=None) -> bool:
        """Start a line. False if the duck is already mid-sentence."""
        if not self._busy.acquire(blocking=False):
            return False
        threading.Thread(target=self._run, args=(text, sim),
                         daemon=True, name="duck-say").start()
        return True

    def _run(self, text: str, sim):
        wav = None
        try:
            fd, wav = tempfile.mkstemp(prefix="duck-say-", suffix=".wav")
            os.close(fd)
            _, traj, _ = render_voice(text, self.ffmpeg, voice=self.tts_voice,
                                      bank_dir=self.bank_dir, out_wav=wav)
            self._perform(wav, traj, sim)
        except Exception as e:
            # The line is already on the control surface; failing to voice it
            # must not take the sim down with it.
            print(f"note: the duck could not say {text!r} ({e})",
                  file=sys.stderr)
        finally:
            if wav and os.path.exists(wav):
                os.unlink(wav)
            self._busy.release()

    def _perform(self, wav_path: str, traj: np.ndarray, sim,
                 rate_hz: int = MOUTH_RATE_HZ):
        """Play, and walk the beak against the same absolute clock as speak()."""
        proc = subprocess.Popen([self.player, wav_path]) if self.player else None
        try:
            t0 = time.perf_counter()
            for i, opening in enumerate(traj):
                deadline = t0 + i / rate_hz
                now = time.perf_counter()
                if deadline > now:
                    time.sleep(deadline - now)
                if sim is not None:
                    sim.mouth_opening = float(opening)
        finally:
            if sim is not None:
                sim.mouth_opening = 0.0
            if proc is not None:
                proc.wait()


# ---------- CLI ----------

def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("text", help="what the duck says (English in, duck out)")
    parser.add_argument("--voice-bank", default=os.environ.get("DUCK_VOICE_BANK"),
                        metavar="DIR",
                        help="directory of voice-bank wavs rendered by the "
                        "microduck `sounds` crate (chirp*.wav is used); "
                        "without it the voice has no chirps")
    parser.add_argument("--voice", default=DEFAULT_TTS_VOICE,
                        help=f"TTS voice (default {DEFAULT_TTS_VOICE})")
    parser.add_argument("--wav-out", default=None, metavar="PATH",
                        help="also keep the rendered wav here")
    parser.add_argument("--audio-only", action="store_true",
                        help="skip the sim: just render and play the voice")
    parser.add_argument("--ffmpeg", default="ffmpeg",
                        help="ffmpeg binary to use (default: ffmpeg on PATH)")


def run(args) -> int:
    """`duck say`. Returns a process exit code."""
    try:
        ffmpeg = find_ffmpeg(args.ffmpeg)
        wav, traj, duration = render_voice(
            args.text, ffmpeg, voice=args.voice,
            bank_dir=args.voice_bank, out_wav=args.wav_out)
        if args.audio_only:
            afplay = shutil.which("afplay")
            if afplay is None:
                raise VoiceError(f"no `afplay`; rendered wav is at {wav}")
            subprocess.run([afplay, wav], check=True)
        else:
            speak(wav, traj, args.text, duration,
                  sock_path=getattr(args, "socket", None))
    except (FilmError, VoiceError) as e:
        print(f"duck say: {e}", file=sys.stderr)
        return 1
    finally:
        if args.wav_out is None:
            tmp = os.path.join(tempfile.gettempdir(), f"duck_say_{os.getpid()}.wav")
            if os.path.exists(tmp):
                os.unlink(tmp)
    return 0
