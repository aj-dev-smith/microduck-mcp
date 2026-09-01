"""Offline tests for emotes: the file format, the render, and the library.

An emote is data, so the whole of it is testable without a sim: parse a table,
reject the ways it can be wrong, and check that the numbers coming out at
50 Hz are the gesture the author wrote. Arbitration — who owns the head, who
owns the beak — lives in test_emote_sim.py, where a bare DuckSim plays them.

    uv run --with pytest pytest tests/ --ignore=tests/live
"""

import os
import tempfile
import textwrap
import unittest

import numpy as np

from microduck_mcp.emote import (CHANNELS, MAX_DURATION_S, RENDER_HZ, Emote,
                                 EmoteError, EmoteLibrary)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMOTES = os.path.join(REPO, "emotes")

GOOD = {"emote": {"name": "tilt", "sound": "inquire"},
        "key": [{"t": 0.0}, {"t": 0.4, "head_roll": 0.3, "ease": "smooth"}]}


def spec(*keys, **emote):
    return {"emote": {"name": "t", **emote}, "key": list(keys)}


class FileFormat(unittest.TestCase):
    def test_a_good_emote_parses(self):
        e = Emote(GOOD)
        self.assertEqual((e.name, e.sound), ("tilt", "inquire"))
        self.assertEqual(e.duration, 0.4)

    def test_a_soundless_emote_is_silent_not_broken(self):
        self.assertIsNone(Emote(spec({"t": 0.0}, {"t": 0.5})).sound)

    def test_the_name_must_match_the_file(self):
        with self.assertRaisesRegex(EmoteError, "name and the file"):
            Emote(GOOD, source_path="wave.toml", expect_name="wave")

    def test_one_key_is_a_look_not_a_gesture(self):
        with self.assertRaisesRegex(EmoteError, "at least 2"):
            Emote(spec({"t": 0.0}))

    def test_it_must_start_at_zero(self):
        with self.assertRaisesRegex(EmoteError, "first key"):
            Emote(spec({"t": 0.2}, {"t": 0.6}))

    def test_key_times_must_increase(self):
        for bad in (({"t": 0.0}, {"t": 0.4}, {"t": 0.4}),
                    ({"t": 0.0}, {"t": 0.6}, {"t": 0.3})):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(EmoteError, "must increase"):
                    Emote(spec(*bad))

    def test_a_gesture_longer_than_the_ceiling_is_a_behavior(self):
        with self.assertRaisesRegex(EmoteError, "borrow"):
            Emote(spec({"t": 0.0}, {"t": MAX_DURATION_S + 0.1}))

    def test_the_beak_opens_zero_to_one(self):
        for mouth in (-0.1, 1.4):
            with self.subTest(mouth=mouth):
                with self.assertRaisesRegex(EmoteError, "0 to 1"):
                    Emote(spec({"t": 0.0}, {"t": 0.5, "mouth": mouth}))

    def test_an_unknown_ease_is_refused(self):
        with self.assertRaisesRegex(EmoteError, "eases"):
            Emote(spec({"t": 0.0}, {"t": 0.5, "ease": "bounce"}))

    def test_a_typo_does_not_pass_silently(self):
        # The failure mode this exists to prevent: `head_ptich` parsing fine,
        # doing nothing, and the author blaming the policy.
        with self.assertRaisesRegex(EmoteError, "head_ptich"):
            Emote(spec({"t": 0.0}, {"t": 0.5, "head_ptich": 0.3}))

    def test_a_channel_must_be_a_number(self):
        with self.assertRaisesRegex(EmoteError, "not a number"):
            Emote(spec({"t": 0.0}, {"t": 0.5, "head_yaw": "left"}))

    def test_a_sound_must_be_a_bank_tag(self):
        with self.assertRaisesRegex(EmoteError, "voice-bank tag"):
            Emote(spec({"t": 0.0}, {"t": 0.5}, sound=3))

    def test_the_table_and_the_name_are_required(self):
        with self.assertRaisesRegex(EmoteError, r"\[emote\]"):
            Emote({"key": [{"t": 0.0}, {"t": 0.5}]})
        with self.assertRaisesRegex(EmoteError, "needs a name"):
            Emote({"emote": {}, "key": [{"t": 0.0}, {"t": 0.5}]})


class Render(unittest.TestCase):
    def test_it_samples_at_the_control_rate_inclusive_of_the_end(self):
        traj = Emote(spec({"t": 0.0}, {"t": 1.2, "head_pitch": 0.3})).render()
        self.assertEqual(set(traj), set(CHANNELS))
        self.assertEqual(len(traj["head_pitch"]), int(1.2 * RENDER_HZ) + 1)

    def test_linear_is_the_straight_line_between_two_keys(self):
        traj = Emote(spec({"t": 0.0},
                          {"t": 1.0, "head_yaw": 0.4, "ease": "linear"})).render()
        y = traj["head_yaw"]
        self.assertAlmostEqual(float(y[0]), 0.0, places=6)
        self.assertAlmostEqual(float(y[RENDER_HZ // 2]), 0.2, places=6)
        self.assertAlmostEqual(float(y[RENDER_HZ]), 0.4, places=6)

    def test_smooth_shares_the_endpoints_and_bulges_between(self):
        # A cosine ease: same endpoints as linear, exactly half way at the
        # midpoint, and slower than linear at both ends (that IS the ease).
        traj = Emote(spec({"t": 0.0}, {"t": 1.0, "head_yaw": 0.4})).render()
        y = traj["head_yaw"]
        self.assertAlmostEqual(float(y[0]), 0.0, places=6)
        self.assertAlmostEqual(float(y[RENDER_HZ]), 0.4, places=6)
        self.assertAlmostEqual(float(y[RENDER_HZ // 2]), 0.2, places=6)
        self.assertLess(float(y[5]), 0.4 * 0.1)          # 0.1 of the way in
        self.assertGreater(float(y[-5]), 0.4 * 0.9)

    def test_hold_does_not_travel_until_the_key_lands(self):
        traj = Emote(spec({"t": 0.0},
                          {"t": 0.5, "head_pitch": 0.3, "ease": "linear"},
                          {"t": 1.0, "head_pitch": 0.0, "ease": "hold"})).render()
        y = traj["head_pitch"]
        held = y[int(0.5 * RENDER_HZ):int(1.0 * RENDER_HZ)]
        np.testing.assert_allclose(held, 0.3, atol=1e-6)
        self.assertAlmostEqual(float(y[-1]), 0.0, places=6)

    def test_an_omitted_channel_carries_the_previous_value(self):
        traj = Emote(spec({"t": 0.0},
                          {"t": 0.5, "head_roll": 0.3},
                          {"t": 1.0, "ease": "hold"},
                          {"t": 1.5, "head_roll": 0.0})).render()
        y = traj["head_roll"]
        np.testing.assert_allclose(y[int(0.5 * RENDER_HZ):int(1.0 * RENDER_HZ)],
                                   0.3, atol=1e-6)

    def test_channels_nobody_wrote_stay_at_zero(self):
        traj = Emote(spec({"t": 0.0}, {"t": 1.0, "head_pitch": 0.3})).render()
        for ch in ("neck_pitch", "head_yaw", "head_roll", "mouth"):
            with self.subTest(channel=ch):
                self.assertEqual(float(np.abs(traj[ch]).max()), 0.0)

    def test_the_render_is_cached_per_rate(self):
        e = Emote(GOOD)
        self.assertIs(e.render(), e.render())
        self.assertIsNot(e.render(), e.render(100))


class ShippedEmotes(unittest.TestCase):
    """The six authored gestures. Their timings are taste; their SIGNS are
    physics, and positive pitch looks down."""

    def emote(self, name):
        return EmoteLibrary(EMOTES).get(name)

    def test_all_of_them_load_and_are_named_for_their_files(self):
        names = EmoteLibrary(EMOTES).names()
        self.assertEqual(names, ["double_take", "droop", "head_tilt",
                                 "new_brain", "nod", "nuzzle", "perk_up",
                                 "shiver", "yawn"])
        for name in names:
            with self.subTest(emote=name):
                self.assertEqual(self.emote(name).name, name)

    def test_the_nod_dips_down_twice_and_ends_level(self):
        y = self.emote("nod").render()["head_pitch"]
        self.assertGreater(float(y.max()), 0.25)      # down is positive
        self.assertAlmostEqual(float(y[0]), 0.0, places=6)
        self.assertAlmostEqual(float(y[-1]), 0.0, places=6)
        peaks = [i for i in range(1, len(y) - 1)
                 if y[i] >= y[i - 1] and y[i] > y[i + 1]]
        self.assertEqual(len(peaks), 2, "a nod is two dips")

    def test_perk_up_lifts_the_head_and_the_neck_with_it(self):
        traj = self.emote("perk_up").render()
        self.assertLess(float(traj["head_pitch"].min()), -0.2)   # up is negative
        self.assertLess(float(traj["neck_pitch"].min()), 0.0)
        self.assertAlmostEqual(float(traj["head_pitch"][-1]), 0.0, places=6)

    def test_the_tilt_stays_inside_the_trained_roll(self):
        # head_roll was trained to ±0.31; past that the command is not a
        # bigger tilt, it is an untrained one.
        y = self.emote("head_tilt").render()["head_roll"]
        self.assertGreater(float(y.max()), 0.25)
        self.assertLessEqual(float(y.max()), 0.31)

    def test_the_beak_stays_shut_unless_the_gesture_is_about_a_sound(self):
        # Only gestures with a voice-bank tag to shape may crack the beak:
        # droop's coo, nuzzle's contented one, the yawn (all the way — that
        # is what a yawn is), the double take's open-mouthed second look. A
        # beak that opens on a silent gesture is a duck mouthing at nothing —
        # which is why the shiver, silent by design, is held to it too.
        for name in ("nod", "head_tilt", "perk_up", "new_brain", "shiver"):
            with self.subTest(emote=name):
                self.assertEqual(
                    float(np.abs(self.emote(name).render()["mouth"]).max()), 0.0)
        traj = self.emote("droop").render()
        self.assertAlmostEqual(float(traj["mouth"].max()), 0.15, places=6)
        self.assertGreater(float(traj["head_pitch"].max()), 0.4)  # and looks down

    def test_the_sounded_ones_name_bank_tags(self):
        # The cozy shelf's split is deliberate: the yawn coos (a sleepy
        # sound, not a call), the double take inquires on its second look,
        # and the shiver is silent — a shiver with a sound effect is a
        # cartoon, a silent one is a bird.
        sounds = {n: self.emote(n).sound for n in EmoteLibrary(EMOTES).names()}
        self.assertEqual(sounds, {"droop": "coo", "head_tilt": "inquire",
                                  "nod": None, "perk_up": "greet",
                                  "new_brain": "chirp", "nuzzle": "coo",
                                  "yawn": "coo", "shiver": None,
                                  "double_take": "inquire"})


class Library(unittest.TestCase):
    def write(self, d, name, body):
        path = os.path.join(d, f"{name}.toml")
        with open(path, "w") as f:
            f.write(textwrap.dedent(body))
        return path

    def setUp(self):
        box = tempfile.TemporaryDirectory()
        self.addCleanup(box.cleanup)
        self.tmp = box.name
        self.write(self.tmp, "wave", """
            [emote]
            name = "wave"
            [[key]]
            t = 0.0
            [[key]]
            t = 0.5
            head_yaw = 0.4
            """)

    def test_it_finds_and_parses_an_emote(self):
        self.assertEqual(EmoteLibrary(self.tmp).get("wave").duration, 0.5)

    def test_an_unknown_name_lists_what_there_is(self):
        with self.assertRaisesRegex(EmoteError, "have: wave"):
            EmoteLibrary(self.tmp).get("shrug")

    def test_an_empty_directory_says_so(self):
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaisesRegex(EmoteError, "no emotes"):
                EmoteLibrary(empty).get("wave")

    def test_a_name_that_is_not_a_name_never_touches_the_disk(self):
        for bad in ("../machines/striker", "wave.toml", ""):
            with self.subTest(name=bad):
                with self.assertRaisesRegex(EmoteError, "not an emote name"):
                    EmoteLibrary(self.tmp).get(bad)

    def test_an_edit_is_picked_up_without_a_reload_verb(self):
        lib = EmoteLibrary(self.tmp)
        first = lib.get("wave")
        self.assertIs(lib.get("wave"), first)      # cached: a stat, not a parse
        self.write(self.tmp, "wave", """
            [emote]
            name = "wave"
            [[key]]
            t = 0.0
            [[key]]
            t = 1.5
            head_yaw = 0.4
            """)
        os.utime(os.path.join(self.tmp, "wave.toml"), (0, 0))  # a changed mtime
        self.assertEqual(lib.get("wave").duration, 1.5)

    def test_a_broken_file_is_remembered_as_its_error(self):
        self.write(self.tmp, "broken", """
            [emote]
            name = "broken"
            [[key]]
            t = 0.0
            [[key]]
            t = 0.5
            head_ptich = 0.3
            """)
        lib = EmoteLibrary(self.tmp)
        with self.assertRaisesRegex(EmoteError, "head_ptich"):
            lib.get("broken")
        listing = {row["name"]: row for row in lib.listing()}
        self.assertEqual(sorted(listing), ["broken", "wave"])
        self.assertFalse(listing["broken"]["valid"])
        self.assertIn("head_ptich", listing["broken"]["error"])
        self.assertTrue(listing["wave"]["valid"])
        self.assertEqual(listing["wave"]["duration_s"], 0.5)


if __name__ == "__main__":
    unittest.main()
