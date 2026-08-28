"""Emotes: short authored gestures, kept as data.

The duck has words (`duck say`) and calls (`duck chirp`); this is the third
channel, the one with no sound in it at all. An emote is a keyframed pose over
time — head and beak — authored in TOML beside the machines it is triggered
from, rendered to per-channel arrays at the control rate and played by the sim
against its own clock:

    emotes/nod.toml  --parse-->  keys  --render-->  50 Hz channel arrays
                                                    |
                                 policy.head_offset <+> sim.mouth_opening

Data, not code, for the same reason machines are: expressiveness is authorship,
and authorship wants a file you can edit while the duck is standing there.
Emotes are re-read when their file changes (EmoteLibrary caches by mtime), so
tuning a gesture is a save away — no reload verb, no restart.

The vocabulary is deliberately tiny. Four head channels and the beak, the same
four the `look` intent writes, in the same order and with the same sign
convention: POSITIVE PITCH LOOKS DOWN (see machine.bhv_approach_ball, which
tilts the camera down at the ball with head_down = +0.5). Radians throughout,
mouth in [0, 1]. Nothing here can command a joint, a velocity or a policy: an
emote is a pose, so the worst a bad one can do is look silly.

Rejections are sentences (ocarina's house style) and happen at parse time,
where the author is watching, rather than mid-gesture where nobody is.
"""

import glob
import os
import tomllib

import numpy as np

# The head channels, in policy.head_offset order. Trained per-joint caps are
# neck/head_pitch ±1.1, head_yaw ±1.4, head_roll ±0.31 (microduck_rl's
# infer_policy) — an emote is clamped to policy.head_max when it is APPLIED,
# never here: this module renders a gesture, the sim decides what its neck can
# do with it.
HEAD_CHANNELS = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")
CHANNELS = HEAD_CHANNELS + ("mouth",)

# How a key is travelled INTO: a cosine ease (the default — bodies accelerate),
# a straight ramp, or no travel at all until the key lands.
EASES = ("smooth", "linear", "hold")

_KEY_FIELDS = frozenset(("t", "ease")) | frozenset(CHANNELS)

# The control rate, spelled out rather than imported from sim_server: an emote
# file must be validatable anywhere — a test, an editor hook, a laptop with no
# MuJoCo on it — and this module's imports stay light enough for that.
RENDER_HZ = 50

# A gesture, not a performance. Eight seconds is already long enough that the
# duck stops looking expressive and starts looking stuck, and an emote holds
# the head away from whatever behavior wants it for the whole time.
MAX_DURATION_S = 8.0


class EmoteError(ValueError):
    pass


class Emote:
    """A validated gesture: its keys, and the sampled channels they mean."""

    def __init__(self, spec: dict, source_path: str = "<inline>",
                 expect_name: str | None = None):
        self.source_path = source_path
        e = spec.get("emote")
        if not isinstance(e, dict):
            raise EmoteError(f"{source_path}: missing [emote] table")
        self.name = e.get("name")
        if not isinstance(self.name, str) or not self.name:
            raise EmoteError(f"{source_path}: [emote] needs a name")
        if expect_name is not None and self.name != expect_name:
            raise EmoteError(
                f"emote {self.name!r} lives in {os.path.basename(source_path)} "
                f"— the name and the file must agree, because a machine names "
                f"an emote by its file")
        self.sound = e.get("sound")
        if self.sound is not None and not isinstance(self.sound, str):
            raise EmoteError(f"{self.name}: sound must be a voice-bank tag")
        keys = spec.get("key")
        if not isinstance(keys, list) or len(keys) < 2:
            raise EmoteError(
                f"{self.name}: an emote needs at least 2 [[key]] tables — one "
                f"pose held still is a look, not a gesture")
        self.keys = [self._key(i, k) for i, k in enumerate(keys)]
        if self.keys[0]["t"] != 0.0:
            raise EmoteError(f"{self.name}: the first key must be t = 0.0 "
                             f"(the pose the gesture starts from)")
        for prev, k in zip(self.keys, self.keys[1:]):
            if k["t"] <= prev["t"]:
                raise EmoteError(
                    f"{self.name}: key times must increase — {k['t']} does not "
                    f"come after {prev['t']}")
        self.duration = self.keys[-1]["t"]
        if self.duration > MAX_DURATION_S:
            raise EmoteError(
                f"{self.name}: {self.duration}s long (max {MAX_DURATION_S}) — "
                f"an emote borrows the head, and a long borrow is a behavior")
        self._rendered = {}

    def _key(self, i: int, k: dict) -> dict:
        """One [[key]] table, validated into plain floats."""
        if not isinstance(k, dict):
            raise EmoteError(f"{self.name}: key {i} is not a table")
        unknown = sorted(set(k) - _KEY_FIELDS)
        if unknown:
            raise EmoteError(
                f"{self.name}: key {i} has no channel {unknown[0]!r} "
                f"(channels: {', '.join(CHANNELS)}, plus t and ease)")
        if "t" not in k:
            raise EmoteError(f"{self.name}: key {i} has no t")
        out = {"ease": k.get("ease", "smooth")}
        if out["ease"] not in EASES:
            raise EmoteError(f"{self.name}: key {i} eases {out['ease']!r} "
                             f"(one of: {', '.join(EASES)})")
        for field in ("t",) + CHANNELS:
            if field not in k:
                continue
            v = k[field]
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise EmoteError(f"{self.name}: key {i}'s {field} is {v!r}, "
                                 f"not a number")
            out[field] = float(v)
        if out["t"] < 0.0:
            raise EmoteError(f"{self.name}: key {i} is at t = {out['t']}")
        if not 0.0 <= out.get("mouth", 0.0) <= 1.0:
            raise EmoteError(f"{self.name}: key {i}'s mouth is {out['mouth']} "
                             f"(the beak opens 0 to 1)")
        return out

    @classmethod
    def load(cls, path: str) -> "Emote":
        with open(path, "rb") as f:
            try:
                spec = tomllib.load(f)
            except tomllib.TOMLDecodeError as e:
                raise EmoteError(f"{path}: TOML does not parse: {e}") from e
        stem = os.path.splitext(os.path.basename(path))[0]
        return cls(spec, source_path=path, expect_name=stem)

    def render(self, rate_hz: int = RENDER_HZ) -> dict:
        """channel -> float32 array at rate_hz, t = 0 through the last key.

        An omitted channel CARRIES the previous key's value (the first key
        starts everything at 0.0), so an author writes only the channels the
        gesture is about — a nod says head_pitch four times and never mentions
        the beak, and the beak stays where the emote found it.

        Cheap enough to run in a request path: a few hundred floats per
        channel, cached per rate because the same gesture plays many times.
        """
        if rate_hz not in self._rendered:
            self._rendered[rate_hz] = self._render(rate_hz)
        return self._rendered[rate_hz]

    def _render(self, rate_hz: int) -> dict:
        n = int(round(self.duration * rate_hz)) + 1
        t = np.arange(n, dtype=np.float64) / rate_hz
        out = {}
        for ch in CHANNELS:
            at_key, v = [], 0.0
            for k in self.keys:
                v = k.get(ch, v)
                at_key.append(v)
            y = np.full(n, at_key[0], dtype=np.float32)
            for i in range(1, len(self.keys)):
                t0, t1 = self.keys[i - 1]["t"], self.keys[i]["t"]
                seg = (t >= t0) & (t <= t1)
                u = (t[seg] - t0) / (t1 - t0)
                ease = self.keys[i]["ease"]
                if ease == "hold":
                    # No travel: the old pose stands until the key lands, and
                    # arrives as a step. A held beat between two gestures.
                    w = (u >= 1.0).astype(np.float64)
                elif ease == "linear":
                    w = u
                else:
                    w = 0.5 - 0.5 * np.cos(np.pi * u)
                y[seg] = at_key[i - 1] + (at_key[i] - at_key[i - 1]) * w
            out[ch] = y
        return out

    def summary(self) -> dict:
        return {"name": self.name, "duration_s": round(self.duration, 3),
                "sound": self.sound, "keys": len(self.keys),
                "source": self.source_path}


class EmoteLibrary:
    """An `emotes/` directory, re-read whenever a file changes on disk.

    Same deal as machine source: the files are the truth and the process is
    just holding them. Parsed emotes are cached by mtime, so a triggered
    gesture costs a stat rather than a parse, and an edit costs one parse —
    which matters because the trigger can land on the 50 Hz sim thread.

    A file that does not parse is remembered as its error, not dropped: the
    listing says which emotes are broken and why, instead of quietly having
    fewer emotes than the directory does.
    """

    def __init__(self, dir_path: str):
        self.dir = os.path.abspath(dir_path)
        self._cache = {}    # name -> (mtime, Emote | EmoteError)

    def path_for(self, name: str) -> str:
        return os.path.join(self.dir, f"{name}.toml")

    def names(self) -> list:
        return sorted(os.path.splitext(os.path.basename(p))[0]
                      for p in glob.glob(os.path.join(self.dir, "*.toml")))

    def get(self, name: str) -> Emote:
        """The named emote, or an EmoteError saying what is wrong with it."""
        if not isinstance(name, str) or not name.isidentifier():
            raise EmoteError(f"{name!r} is not an emote name")
        path = self.path_for(name)
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            have = self.names()
            raise EmoteError(
                f"no emote named {name!r} in {self.dir}"
                + (f" (have: {', '.join(have)})" if have else
                   " — the directory holds no emotes")) from None
        hit = self._cache.get(name)
        if hit is None or hit[0] != mtime:
            try:
                hit = (mtime, Emote.load(path))
            except (EmoteError, OSError) as e:
                hit = (mtime, e)
            self._cache[name] = hit
        if isinstance(hit[1], Exception):
            raise hit[1]
        return hit[1]

    def listing(self) -> list:
        """Every emote in the directory with its validity — the `list` action.

        Deliberately reports the broken ones too: an emote a machine names and
        cannot play is exactly what somebody needs to see.
        """
        out = []
        for name in self.names():
            try:
                out.append({**self.get(name).summary(), "valid": True})
            except EmoteError as e:
                out.append({"name": name, "valid": False, "error": str(e)})
        return out
