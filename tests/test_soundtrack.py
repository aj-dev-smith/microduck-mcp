"""Offline tests for the film's soundtrack: the plan, the rule, the mix.

No TTS and no encoder here. What is testable without either is everything that
decides *what* is heard and *when* — above all the one rule this sound design
has, which is that the wheee belongs to earned goals and to nothing else. It
is a test rather than a habit precisely because it is the kind of thing a
later "just a little one on the kick" would quietly erode.

    uv run --with pytest pytest tests/ --ignore=tests/live
"""

import os
import shutil
import tempfile
import unittest

import numpy as np

from microduck_mcp import soundtrack as st
from microduck_mcp.soundtrack import (ARM, CHIRP, GOAL, NODE, SPEAK, WHEEE,
                                      Beat, Cue, SoundKit, Voicing, mix,
                                      mouth_at, plan_cues, trim_sting)

SR = st.SR


def match(goal_at=None, kicks=(), extra=()):
    """A take's beats: the machine arms, kicks happen, maybe someone scores."""
    beats = [Beat(ARM, 0.0, "arm")]
    beats += [Beat(NODE, t, "kick") for t in kicks]
    if goal_at is not None:
        beats.append(Beat(GOAL, goal_at))
        beats.append(Beat(NODE, goal_at + 0.1, "celebrate"))
    beats += list(extra)
    return sorted(beats, key=lambda b: b.t_s)


def kinds(cues):
    return [c.kind for c in cues]


class TheWheeeRule(unittest.TestCase):
    """`wheee` is reserved for earned goals. This is the whole rule."""

    def test_a_goal_gets_the_wheee_at_the_moment_it_is_called(self):
        cues = plan_cues(match(goal_at=44.9))
        wheee = [c for c in cues if c.kind == WHEEE]
        self.assertEqual(len(wheee), 1)
        self.assertAlmostEqual(wheee[0].t_s, 44.9)

    def test_no_goal_means_no_wheee_however_long_the_match(self):
        cues = plan_cues(match(kicks=(12.0, 31.5, 60.0, 88.0)))
        self.assertNotIn(WHEEE, kinds(cues))

    def test_the_arm_never_gets_one(self):
        # A film that opens on a wheee is decoration; the duck has not done
        # anything yet.
        cues = plan_cues([Beat(ARM, 0.0, "arm")])
        self.assertNotIn(WHEEE, kinds(cues))

    def test_a_kick_never_gets_one(self):
        cues = plan_cues(match(kicks=(10.0,)))
        self.assertNotIn(WHEEE, kinds(cues))
        self.assertIn(CHIRP, kinds(cues))   # the kick gets an accent, not a wheee

    def test_no_other_node_can_earn_one(self):
        nodes = ("search", "approach", "kick", "watch", "backoff",
                 "celebrate", "won", "down")
        cues = plan_cues([Beat(NODE, float(i), n)
                          for i, n in enumerate(nodes)])
        self.assertNotIn(WHEEE, kinds(cues))

    def test_a_second_referee_call_does_not_earn_a_second_one(self):
        # The referee latches while the ball sits in the net; the sound must
        # not stutter along with it.
        cues = plan_cues([Beat(ARM, 0.0), Beat(GOAL, 20.0), Beat(GOAL, 20.4)])
        self.assertEqual(kinds(cues).count(WHEEE), 1)


class TheScript(unittest.TestCase):
    def test_the_machine_arming_opens_the_film_with_a_line(self):
        cues = plan_cues(match())
        self.assertEqual(cues[0].kind, SPEAK)
        self.assertEqual(cues[0].t_s, 0.0)
        self.assertEqual(cues[0].text, st.DEFAULT_LINES["arm"])

    def test_the_duck_yells_first_and_talks_second(self):
        cues = plan_cues(match(goal_at=30.0))
        wheee = next(c for c in cues if c.kind == WHEEE)
        line = [c for c in cues if c.kind == SPEAK][-1]
        self.assertGreater(line.t_s, wheee.t_s)
        # ...and the line waits for the sting to clear rather than talking over it
        self.assertGreaterEqual(line.t_s - wheee.t_s, st.WHEEE_MAX_S)

    def test_lines_are_overridable(self):
        cues = plan_cues(match(goal_at=5.0),
                         {"arm": "here we go", "goal": "get in"})
        said = [c.text for c in cues if c.kind == SPEAK]
        self.assertEqual(said, ["here we go", "get in"])

    def test_an_emptied_line_is_a_deletion_not_a_silence(self):
        cues = plan_cues(match(goal_at=5.0), {"arm": "", "goal": "  "})
        self.assertEqual(kinds(cues), [WHEEE])   # the goal still earns its sting

    def test_chirp_accents_are_capped(self):
        cues = plan_cues(match(kicks=(10.0, 20.0, 30.0, 40.0, 50.0)))
        self.assertEqual(kinds(cues).count(CHIRP), st.MAX_KICK_CHIRPS)

    def test_cues_come_out_in_time_order(self):
        cues = plan_cues(match(goal_at=40.0, kicks=(38.0, 12.0)))
        self.assertEqual([c.t_s for c in cues], sorted(c.t_s for c in cues))

    def test_a_silent_take_plans_nothing_to_play(self):
        self.assertEqual(plan_cues([], {}), [])


class TheSting(unittest.TestCase):
    """The bank's wheee is a sustained call; the film wants a sting."""

    def test_a_long_call_is_cut_and_faded(self):
        x = np.ones(int(5.0 * SR), dtype=np.float32)
        out = trim_sting(x)
        self.assertEqual(len(out), int(st.WHEEE_MAX_S * SR))
        self.assertAlmostEqual(float(out[-1]), 0.0, places=5)
        self.assertAlmostEqual(float(out[0]), 1.0, places=5)

    def test_a_short_one_is_left_alone(self):
        x = np.ones(int(0.4 * SR), dtype=np.float32)
        self.assertTrue(np.array_equal(trim_sting(x), x))


class TheMix(unittest.TestCase):
    def test_segments_land_on_their_timestamps(self):
        blip = np.full(SR // 10, 0.5, dtype=np.float32)
        out = mix([(1.0, blip, 1.0)], total_s=3.0)
        self.assertEqual(len(out), 3 * SR)
        self.assertEqual(float(out[0]), 0.0)
        self.assertAlmostEqual(float(out[SR]), 0.5, places=5)
        self.assertAlmostEqual(float(out[SR + SR // 10 + 10]), 0.0)

    def test_gain_is_applied(self):
        blip = np.ones(100, dtype=np.float32)
        out = mix([(0.0, blip, 0.25)], total_s=1.0)
        self.assertAlmostEqual(float(out[0]), 0.25)

    def test_overlapping_segments_sum(self):
        blip = np.full(SR, 0.2, dtype=np.float32)
        out = mix([(0.0, blip, 1.0), (0.0, blip, 1.0)], total_s=1.0)
        self.assertAlmostEqual(float(out[0]), 0.4, places=5)

    def test_a_segment_running_past_the_end_is_truncated(self):
        blip = np.ones(3 * SR, dtype=np.float32)
        out = mix([(0.5, blip, 1.0)], total_s=1.0)
        self.assertEqual(len(out), SR)

    def test_a_segment_starting_before_the_film_keeps_only_what_is_left(self):
        ramp = np.arange(SR, dtype=np.float32) / SR / 2
        out = mix([(-0.5, ramp, 1.0)], total_s=1.0)
        self.assertAlmostEqual(float(out[0]), 0.25, places=3)

    def test_a_segment_landing_past_the_end_is_dropped_not_an_error(self):
        out = mix([(9.0, np.ones(100, dtype=np.float32), 1.0)], total_s=1.0)
        self.assertEqual(float(np.abs(out).max()), 0.0)

    def test_a_sum_that_would_clip_is_scaled_down_never_up(self):
        loud = np.ones(SR, dtype=np.float32)
        out = mix([(0.0, loud, 1.0)] * 4, total_s=1.0)
        self.assertLessEqual(float(np.abs(out).max()), st.PEAK_CEILING + 1e-6)
        quiet = mix([(0.0, np.full(SR, 0.1, dtype=np.float32), 1.0)], total_s=1.0)
        self.assertAlmostEqual(float(np.abs(quiet).max()), 0.1, places=5)

    def test_nothing_to_play_is_silence_of_the_right_length(self):
        out = mix([], total_s=2.0)
        self.assertEqual(len(out), 2 * SR)
        self.assertEqual(float(np.abs(out).max()), 0.0)


def voicing(text="hello", duration_s=1.0, mouth=None):
    n = int(duration_s * SR)
    m = np.linspace(0, 1, int(duration_s * st.MOUTH_RATE_HZ), dtype=np.float32) \
        if mouth is None else np.asarray(mouth, dtype=np.float32)
    return Voicing(text, np.full(n, 0.5, dtype=np.float32), m, duration_s)


def kit(voicings=None, wheee=True, chirp=True, lines=None):
    """A SoundKit with rendered sound stubbed in — no TTS, no bank on disk."""
    k = object.__new__(SoundKit)
    k.lines = dict(st.DEFAULT_LINES if lines is None else lines)
    k.ffmpeg = "ffmpeg"
    k.bank_dir = None
    k.notes = []
    k.voicings = {} if voicings is None else voicings
    k.wheee = np.ones(int(0.5 * SR), dtype=np.float32) if wheee else None
    k.chirp = np.ones(int(0.1 * SR), dtype=np.float32) if chirp else None
    return k


class TheBeak(unittest.TestCase):
    def test_nobody_talking_is_not_the_same_as_a_shut_beak(self):
        # None hides the HUD row; 0.0 draws it closed. A closed plosive
        # mid-sentence is not silence.
        self.assertIsNone(mouth_at([], 1.0))
        self.assertIsNone(mouth_at([(0.0, voicing())], 5.0))
        self.assertIsNotNone(mouth_at([(0.0, voicing())], 0.5))

    def test_it_follows_the_rendered_trajectory(self):
        v = voicing(duration_s=1.0, mouth=[0.0, 0.5, 1.0, 0.25])
        opened = [mouth_at([(0.0, v)], i / st.MOUTH_RATE_HZ) for i in range(4)]
        self.assertEqual(opened, [0.0, 0.5, 1.0, 0.25])

    def test_it_starts_when_the_line_does(self):
        v = voicing(duration_s=1.0, mouth=[1.0, 1.0])
        self.assertIsNone(mouth_at([(10.0, v)], 9.9))
        self.assertEqual(mouth_at([(10.0, v)], 10.0), 1.0)

    def test_overlapping_lines_take_the_wider_opening(self):
        a = voicing("a", mouth=[0.2, 0.2])
        b = voicing("b", mouth=[0.9, 0.9])
        self.assertAlmostEqual(mouth_at([(0.0, a), (0.0, b)], 0.0), 0.9,
                               places=5)


class Segments(unittest.TestCase):
    def test_every_cue_becomes_a_placed_sound(self):
        k = kit({"arm": voicing(st.DEFAULT_LINES["arm"])})
        cues = plan_cues(match(goal_at=10.0, kicks=(5.0,)),
                         {"arm": st.DEFAULT_LINES["arm"], "goal": ""})
        segs = k.segments(cues)
        self.assertEqual([round(t, 1) for t, _, _ in segs], [0.0, 5.0, 10.0])
        self.assertEqual([g for _, _, g in segs],
                         [st.SPEECH_GAIN, st.CHIRP_GAIN, st.WHEEE_GAIN])

    def test_a_line_that_never_rendered_is_dropped_not_faked(self):
        segs = kit().segments([Cue(0.0, SPEAK, "unrendered")])
        self.assertEqual(segs, [])

    def test_a_bankless_kit_still_places_the_speech(self):
        k = kit({"arm": voicing("hi")}, wheee=False, chirp=False)
        segs = k.segments([Cue(0.0, SPEAK, "hi"), Cue(1.0, WHEEE),
                           Cue(2.0, CHIRP)])
        self.assertEqual(len(segs), 1)

    def test_the_cut_clock_moves_every_cue_together(self):
        k = kit({"arm": voicing("hi")})
        segs = k.segments([Cue(0.0, SPEAK, "hi"), Cue(4.0, WHEEE)],
                          at=lambda t: t + 1.0)
        self.assertEqual([t for t, _, _ in segs], [1.0, 5.0])

    def test_a_kit_with_nothing_rendered_is_not_audible(self):
        self.assertFalse(kit(wheee=False, chirp=False).audible)
        self.assertTrue(kit(wheee=False, chirp=True).audible)
        self.assertTrue(kit({"arm": voicing()}, wheee=False, chirp=False).audible)

    def test_a_track_is_as_long_as_the_film(self):
        k = kit({"arm": voicing("hi", duration_s=0.5)})
        track = k.track([Cue(0.0, SPEAK, "hi")], total_s=4.0)
        self.assertEqual(len(track), 4 * SR)


class TheVoiceBank(unittest.TestCase):
    def test_the_crate_is_found_beside_the_policies(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = os.path.join(tmp, "microduck")
            os.makedirs(os.path.join(repo, "policies"))
            self.assertIsNone(st.find_sounds_repo(
                os.path.join(repo, "policies")))
            open(os.path.join(repo, "Cargo.toml"), "w").close()
            self.assertEqual(st.find_sounds_repo(
                os.path.join(repo, "policies")), repo)

    def test_no_cargo_means_no_bank_not_an_exception(self):
        real = shutil.which
        shutil.which = lambda name: None if name == "cargo" else real(name)
        try:
            self.assertIsNone(st.render_bank("/nowhere", "/nowhere/bank"))
        finally:
            shutil.which = real

    def test_a_crate_that_cannot_be_run_degrades_to_none(self):
        if shutil.which("cargo") is None:
            self.skipTest("no cargo on this machine")
        with tempfile.TemporaryDirectory() as tmp:
            # A directory with no Cargo.toml: cargo exits non-zero, and the
            # shoot must read that as "film chirpless", not as a crash.
            self.assertIsNone(st.render_bank(tmp, os.path.join(tmp, "bank"),
                                             timeout_s=60))

    def test_a_missing_bank_wav_is_a_none_not_a_stack_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(st._bank_wav(tmp, "wheee.wav"))
            self.assertIsNone(st._bank_wav(None, "wheee.wav"))
            broken = os.path.join(tmp, "wheee.wav")
            with open(broken, "w") as f:
                f.write("not a wav")
            self.assertIsNone(st._bank_wav(tmp, "wheee.wav"))


class TheShootLog(unittest.TestCase):
    def test_it_reads_as_a_timeline(self):
        note = st.scored_note(plan_cues(match(goal_at=12.0)))
        self.assertIn("0.0s say", note)
        self.assertIn("12.0s wheee", note)

    def test_a_silent_film_says_so(self):
        self.assertEqual(st.scored_note([]), "silent")


if __name__ == "__main__":
    unittest.main()
