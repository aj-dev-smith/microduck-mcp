"""Offline tests for the voice: DSP, chirp placement, mouth, and the wiring.

Silent by design — nothing here spawns `say` or `afplay`, and nothing renders
audio out loud: signals are synthesized, and the one ffmpeg round-trip lives
in test_film. What is worth locking down is everything deterministic between
text and performance: the envelope, which syllables get chirps, how the beak
tracks the voice, the graceful chirpless degradation, and the mouth intent's
clamp-and-pose math in the sim server.

    uv run --with pytest pytest tests/ --ignore=tests/live
"""

import io
import json
import math
import os
import tempfile
import unittest
from contextlib import redirect_stderr

import numpy as np

from microduck_mcp import voice
from microduck_mcp.sim_server import (MOUTH_HINGE_POS, MOUTH_MAX_RAD,
                                      mouth_pose)
from microduck_mcp.voice import (ENV_RATE, GRAIN_GAIN, MOODS, MOUTH_RATE_HZ,
                                 SR, VoiceError, blend_chirps, envelope,
                                 load_grain, load_wav48, modulation_chain,
                                 mouth_trajectory, resolve_mood, save_wav48,
                                 syllable_peaks)


def tone_bursts(centers_s, dur_s=2.0, burst_s=0.12, freq=300.0):
    """Silence with short loud tone bursts at the given times — fake syllables."""
    t = np.arange(int(dur_s * SR)) / SR
    x = np.zeros_like(t, dtype=np.float32)
    for c in centers_s:
        on = (t > c) & (t < c + burst_s)
        x[on] = 0.8 * np.sin(2 * math.pi * freq * t[on]).astype(np.float32)
    return x


class Envelope(unittest.TestCase):
    def test_silence_is_flat_zero(self):
        env = envelope(np.zeros(SR, dtype=np.float32))
        self.assertEqual(len(env), ENV_RATE)
        self.assertEqual(float(env.max()), 0.0)

    def test_burst_raises_it_and_it_decays_after(self):
        env = envelope(tone_bursts([0.5], dur_s=1.5))
        peak_i = int(np.argmax(env))
        self.assertAlmostEqual(peak_i / ENV_RATE, 0.6, delta=0.1)
        self.assertLess(env[-1], 0.05 * env[peak_i])


class SyllablePeaks(unittest.TestCase):
    def test_keeps_the_top_half_and_each_lands_on_a_burst(self):
        env = envelope(tone_bursts([0.3, 0.9, 1.5]))
        times = [i / ENV_RATE for i, _, _ in syllable_peaks(env)]
        # three equal syllables -> the top half by attack = 2 kept
        self.assertEqual(len(times), 2)
        for t in times:
            self.assertTrue(any(c <= t <= c + 0.15 for c in (0.3, 0.9, 1.5)),
                            f"peak at {t}s is not on any burst")

    def test_deterministic(self):
        env = envelope(tone_bursts([0.3, 0.9, 1.5]))
        self.assertEqual(syllable_peaks(env), syllable_peaks(env))

    def test_silence_has_no_syllables(self):
        self.assertEqual(syllable_peaks(np.zeros(ENV_RATE)), [])


class ChirpBlend(unittest.TestCase):
    def chirp(self):
        t = np.arange(int(0.15 * SR)) / SR
        return (0.5 * np.sin(2 * math.pi * 900 * t)).astype(np.float32)

    def test_grains_land_on_the_syllables_only(self):
        speech = tone_bursts([0.4, 1.2])
        env = envelope(speech)
        out, hits = blend_chirps(speech, self.chirp(), env)
        self.assertEqual(len(hits), 2)
        delta = np.abs(out - speech)
        # energy was added near the bursts...
        self.assertGreater(delta[int(0.35 * SR):int(0.65 * SR)].max(), 0.01)
        # ...and the silence between them is untouched: chirps in the words,
        # not between them — the whole point of the v3 -> v4 direction.
        self.assertEqual(float(delta[int(0.8 * SR):int(1.1 * SR)].max()), 0.0)

    def test_no_chirp_bank_means_untouched_speech(self):
        speech = tone_bursts([0.4])
        out, hits = blend_chirps(speech, None, envelope(speech))
        self.assertIs(out, speech)
        self.assertEqual(hits, [])

    def test_grain_gain_is_the_ratified_poc_value(self):
        self.assertEqual(GRAIN_GAIN, 0.75)


class MouthTrajectory(unittest.TestCase):
    def test_silence_keeps_the_beak_shut(self):
        traj = mouth_trajectory(envelope(np.zeros(SR, dtype=np.float32)))
        self.assertTrue((traj == 0.0).all())

    def test_speech_opens_it_and_it_closes_after(self):
        traj = mouth_trajectory(envelope(tone_bursts([0.5], dur_s=1.6)))
        self.assertEqual(len(traj), int(1.6 * MOUTH_RATE_HZ))
        self.assertGreater(traj.max(), 0.9)   # normalized: it really opens
        self.assertEqual(float(traj[-1]), 0.0)  # and shuts by the end

    def test_attack_beats_release(self):
        traj = mouth_trajectory(envelope(tone_bursts([0.5], dur_s=1.6)))
        peak = int(np.argmax(traj))
        onset = int(0.5 * MOUTH_RATE_HZ)
        rise = peak - onset
        fall = int(np.argmax(traj[peak:] < 0.1))
        self.assertGreater(fall, rise, "a beak snaps open and eases shut")

    def test_range_is_clamped(self):
        traj = mouth_trajectory(envelope(tone_bursts([0.2, 0.6, 1.0])))
        self.assertGreaterEqual(float(traj.min()), 0.0)
        self.assertLessEqual(float(traj.max()), 1.0)


class VoiceBank(unittest.TestCase):
    def test_missing_dir_and_empty_dir_degrade_with_a_note(self):
        self.assertIsNone(load_grain(None))
        with tempfile.TemporaryDirectory() as d, redirect_stderr(io.StringIO()) as err:
            self.assertIsNone(load_grain(d))
        self.assertIn("without chirps", err.getvalue())

    def test_bank_chirp_round_trips_via_wav(self):
        ramp = np.linspace(-0.5, 0.5, SR // 10, dtype=np.float32)
        with tempfile.TemporaryDirectory() as d:
            save_wav48(os.path.join(d, "chirp.wav"), ramp)
            chirp = load_grain(d)
        self.assertEqual(len(chirp), SR // 10)
        np.testing.assert_allclose(chirp, ramp, atol=2 / 32767)

    def test_too_long_text_is_refused(self):
        with self.assertRaises(VoiceError):
            voice.render_voice("quack " * 200, ffmpeg="ffmpeg")


class ChirpBank(unittest.TestCase):
    """`duck chirp`: resolving a tag to a wav, deterministically.

    The bank is content the duck reacts with, so the same tag must name the
    same call on every host — hence sorted globs and an index, not a shuffle.
    """

    def bank(self, d, *names):
        for name in names:
            save_wav48(os.path.join(d, name),
                       np.linspace(-0.4, 0.4, SR // 20, dtype=np.float32))
        return d

    def test_the_first_sorted_match_wins(self):
        with tempfile.TemporaryDirectory() as d:
            self.bank(d, "greet2.wav", "greet1.wav", "greet3.wav")
            self.assertTrue(voice.bank_wav_path(d, "greet").endswith("greet1.wav"))

    def test_variant_indexes_the_sorted_matches(self):
        with tempfile.TemporaryDirectory() as d:
            self.bank(d, "chirp1.wav", "chirp2.wav")
            self.assertTrue(
                voice.bank_wav_path(d, "chirp", 1).endswith("chirp2.wav"))
            with self.assertRaisesRegex(VoiceError, "no variant 2"):
                voice.bank_wav_path(d, "chirp", 2)

    def test_a_bare_tag_matches_the_unnumbered_wav(self):
        with tempfile.TemporaryDirectory() as d:
            self.bank(d, "wheee.wav")
            self.assertTrue(voice.bank_wav_path(d, "wheee").endswith("wheee.wav"))

    def test_missing_bank_and_missing_tag_are_clean_sentences(self):
        with self.assertRaisesRegex(VoiceError, "no voice bank"):
            voice.bank_wav_path(None, "coo")
        with tempfile.TemporaryDirectory() as d:
            self.bank(d, "coo.wav")
            with self.assertRaisesRegex(VoiceError, r"no alarm\*\.wav"):
                voice.bank_wav_path(d, "alarm")
            with self.assertRaisesRegex(VoiceError, "not a voice-bank tag"):
                voice.bank_wav_path(d, "../etc/passwd")

    def test_the_call_drives_the_beak_from_its_own_envelope(self):
        with tempfile.TemporaryDirectory() as d:
            save_wav48(os.path.join(d, "peck.wav"), tone_bursts([0.2], dur_s=1.0))
            path, traj = voice.chirp_render(d, "peck")
        self.assertTrue(path.endswith("peck.wav"))
        self.assertEqual(len(traj), int(1.0 * MOUTH_RATE_HZ))
        self.assertGreater(traj.max(), 0.9)
        self.assertEqual(float(traj[-1]), 0.0)   # and shuts when the call ends


class Moods(unittest.TestCase):
    """Weather over the climate: a mood moves the recipe's knobs, nothing else.

    The numbers in MOODS are meant to be retuned by ear, so nothing here
    asserts a taste — it asserts that neutral is still literally the ratified
    pipeline, that every mood resolves to a complete and playable parameter
    set, and that the parameter set is what the stages actually read.
    """

    def test_neutral_is_the_ratified_recipe_verbatim(self):
        # The one mood that is not allowed to be a change: `duck say` with no
        # mood must render exactly what it rendered before moods existed.
        legacy = (f"aresample={SR},asetrate={voice.PITCH_ASETRATE},"
                  f"aresample={SR},atempo={voice.PITCH_TEMPO},"
                  f"vibrato=f={voice.VIBRATO_HZ}:d={voice.VIBRATO_DEPTH},"
                  f"tremolo=f={voice.TREMOLO_HZ}:d={voice.TREMOLO_DEPTH},"
                  f"aformat=sample_rates={SR}:channel_layouts=mono")
        self.assertEqual(modulation_chain(MOODS["neutral"]), legacy)
        self.assertEqual(modulation_chain(), legacy)
        neutral = MOODS["neutral"]
        self.assertEqual(neutral.grain_gain, GRAIN_GAIN)
        self.assertEqual(neutral.grain_tag, "chirp")
        self.assertEqual(neutral.mouth_release_s, voice.MOUTH_RELEASE_S)
        self.assertEqual(neutral.syllable_share, 0.5)

    def test_neutral_keeps_the_old_top_half_syllable_count(self):
        for centers in ([0.3, 0.9, 1.5], [0.3, 0.7, 1.1, 1.5, 1.9]):
            with self.subTest(syllables=len(centers)):
                env = envelope(tone_bursts(centers, dur_s=2.2))
                self.assertEqual(len(syllable_peaks(env)),
                                 max(2, len(syllable_peaks(
                                     env, share=1.0)) // 2))

    def test_every_mood_resolves_to_a_playable_parameter_set(self):
        for name, m in MOODS.items():
            with self.subTest(mood=name):
                self.assertIs(resolve_mood(name), m)
                # atempo's own domain: outside it ffmpeg refuses the filter.
                self.assertGreaterEqual(m.tempo, 0.5)
                self.assertLessEqual(m.tempo, 2.0)
                for depth in (m.vibrato_depth, m.tremolo_depth):
                    self.assertGreaterEqual(depth, 0.0)
                    self.assertLessEqual(depth, 1.0)
                self.assertGreater(m.syllable_share, 0.0)
                self.assertLessEqual(m.syllable_share, 1.0)
                self.assertTrue(m.grain_tag and m.grain_tag.isidentifier())
                self.assertGreater(m.grain_gain, 0.0)
                self.assertGreater(m.mouth_release_s, 0.0)
                # Still a duck in every weather: the pitch-up never comes off,
                # so no mood can land the voice back in human range.
                self.assertGreater(m.asetrate, SR)

    def test_the_filter_chain_is_built_from_the_mood(self):
        for name, expected in (("excited", "asetrate=56800"),
                               ("sad", "atempo=0.74"),
                               ("alarmed", "tremolo=f=25.9:d=0.33"),
                               ("smug", "vibrato=f=8.6:d=0.2,")):
            with self.subTest(mood=name):
                self.assertIn(expected, modulation_chain(MOODS[name]))
        # and the rates are the personality's, in every mood
        for m in MOODS.values():
            self.assertIn(f"vibrato=f={voice.VIBRATO_HZ}", modulation_chain(m))

    def test_the_mood_picks_the_grain_tag_out_of_the_one_bank_loader(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ("chirp1.wav", "chirp2.wav", "coo1.wav", "alarm1.wav",
                         "peck1.wav"):
                save_wav48(os.path.join(d, name),
                           np.linspace(-0.4, 0.4, SR // 20, dtype=np.float32))
            for mood, wav in (("neutral", "chirp1.wav"), ("excited", "chirp1.wav"),
                              ("sad", "coo1.wav"), ("alarmed", "alarm1.wav"),
                              ("smug", "peck1.wav")):
                with self.subTest(mood=mood):
                    grain = load_grain(d, MOODS[mood].grain_tag)
                    self.assertIsNotNone(grain)
                    # the same sorted tag lookup `duck chirp` resolves with
                    self.assertTrue(
                        voice.bank_wav_path(d, MOODS[mood].grain_tag)
                        .endswith(wav))

    def test_a_bank_without_this_moods_tag_still_speaks(self):
        # A duck that cannot coo should still be able to be sad out loud.
        with tempfile.TemporaryDirectory() as d:
            save_wav48(os.path.join(d, "chirp.wav"),
                       np.linspace(-0.4, 0.4, SR // 20, dtype=np.float32))
            with redirect_stderr(io.StringIO()) as err:
                self.assertIsNone(load_grain(d, MOODS["sad"].grain_tag))
        self.assertIn("no coo*.wav", err.getvalue())

    def test_the_syllable_share_is_the_moods_and_the_floor_holds(self):
        env = envelope(tone_bursts([0.3, 0.6, 0.9, 1.2, 1.5, 1.8], dur_s=2.2))
        self.assertEqual(len(syllable_peaks(env, share=1.0)), 6)
        counts = {name: len(syllable_peaks(env, share=m.syllable_share))
                  for name, m in MOODS.items()}
        self.assertEqual(counts["neutral"], 3)
        self.assertEqual(counts["excited"], 4)   # two thirds, not 3.99 -> 3
        self.assertEqual(counts["sad"], 2)
        # ...and the floor of two survives any mood, on any short line
        short = envelope(tone_bursts([0.3, 0.9, 1.5]))
        self.assertEqual(len(syllable_peaks(short, share=1 / 3)), 2)

    def test_the_grain_gain_and_the_beak_read_the_mood_too(self):
        speech = tone_bursts([0.4, 1.2])
        env = envelope(speech)
        chirp = (0.5 * np.sin(2 * math.pi * 900 *
                              np.arange(int(0.15 * SR)) / SR)).astype(np.float32)
        loud, _ = blend_chirps(speech, chirp, env, MOODS["excited"])
        quiet, _ = blend_chirps(speech, chirp, env, MOODS["smug"])
        self.assertGreater(np.abs(loud - speech).max(),
                           np.abs(quiet - speech).max())
        # a sad beak droops shut: the long release keeps it open longer
        long_line = envelope(tone_bursts([0.5], dur_s=1.6))
        sad = mouth_trajectory(long_line, mood=MOODS["sad"])
        keen = mouth_trajectory(long_line, mood=MOODS["excited"])
        self.assertGreater(float(sad.sum()), float(keen.sum()))

    def test_an_unknown_mood_is_a_clean_sentence_everywhere(self):
        from unittest import mock
        from microduck_mcp import client
        with self.assertRaisesRegex(VoiceError, "is not a mood"):
            resolve_mood("furious")
        with self.assertRaisesRegex(VoiceError, "is not a mood"):
            voice.render_voice("hello", ffmpeg="ffmpeg", mood="furious")
        # the roster is in the message, so the fix is in the error
        with self.assertRaisesRegex(VoiceError, "smug"):
            resolve_mood("furious")
        # CLI: argparse refuses it before anything renders
        with mock.patch.object(client.sys, "argv",
                               ["duck", "say", "hi", "--mood", "furious"]):
            with redirect_stderr(io.StringIO()) as err:
                with self.assertRaises(SystemExit) as cm:
                    client.main()
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("furious", err.getvalue())

    def test_the_mcp_tool_refuses_it_as_a_tool_error(self):
        from unittest import mock
        from mcp.server.mcpserver.exceptions import ToolError
        from microduck_mcp import mcp_server
        with mock.patch.object(voice, "find_ffmpeg", return_value="ffmpeg"):
            with self.assertRaisesRegex(ToolError, "is not a mood"):
                mcp_server.duck_say("hello", mood="furious")


class TheWheeeIsEarned(unittest.TestCase):
    """The server's half of the chirp contract, on a bare DuckSim.

    Fairness by construction, the same shape as the guard vocabulary: the
    celebration is not something a client can decide it deserves.
    """

    def sim(self, count=None):
        from collections import deque
        from microduck_mcp.sim_server import DuckSim, GoalReferee
        sim = DuckSim.__new__(DuckSim)
        sim.events = deque(maxlen=500)
        sim._event_id = 0
        sim.policy = None     # a chirp is an annotation: no physics in the path
        sim.referee = None
        if count is not None:
            sim.referee = GoalReferee()
            sim.referee.count = count
        return sim

    def test_no_goal_no_wheee(self):
        resp = self.sim(count=0).handle({"cmd": "chirp", "tag": "wheee"})
        self.assertFalse(resp["ok"])
        self.assertIn("actually scored", resp["error"])

    def test_a_scene_without_a_referee_can_never_earn_one(self):
        self.assertFalse(self.sim().handle({"cmd": "chirp", "tag": "wheee"})["ok"])

    def test_a_goal_on_the_board_grants_it(self):
        resp = self.sim(count=1).handle({"cmd": "chirp", "tag": "wheee"})
        self.assertTrue(resp["ok"])
        self.assertEqual(resp["goals"], 1)

    def test_every_other_call_is_the_duck_s_to_make(self):
        for tag in ("alarm", "greet", "inquire", "peck", "chirp", "coo"):
            with self.subTest(tag=tag):
                self.assertTrue(self.sim().handle({"cmd": "chirp",
                                                   "tag": tag})["ok"])


class MouthPose(unittest.TestCase):
    HEAD_POS = np.array([0.1, -0.2, 0.22])
    HEAD_QUAT = np.array([0.92387953, 0.0, 0.0, 0.38268343])  # 45 deg yaw

    def test_closed_coincides_with_the_head(self):
        pos, quat = mouth_pose(self.HEAD_POS, self.HEAD_QUAT, 0.0)
        np.testing.assert_allclose(pos, self.HEAD_POS, atol=1e-12)
        np.testing.assert_allclose(quat, self.HEAD_QUAT, atol=1e-12)

    def test_opening_clamps_both_ways(self):
        for raw, ref in ((-3.0, 0.0), (7.0, 1.0)):
            pos, quat = mouth_pose(self.HEAD_POS, self.HEAD_QUAT, raw)
            rpos, rquat = mouth_pose(self.HEAD_POS, self.HEAD_QUAT, ref)
            np.testing.assert_allclose(pos, rpos)
            np.testing.assert_allclose(quat, rquat)

    def test_hinge_point_never_moves(self):
        """The plate pivots AT the hinge: that head-frame point maps to the
        same world point at any opening, which is what makes it a hinge and
        not a drift."""
        import mujoco
        h = np.asarray(MOUTH_HINGE_POS)
        worlds = []
        for opening in (0.0, 0.5, 1.0):
            pos, quat = mouth_pose(self.HEAD_POS, self.HEAD_QUAT, opening)
            r = np.zeros(3)
            mujoco.mju_rotVecQuat(r, h, quat)
            worlds.append(pos + r)
        np.testing.assert_allclose(worlds[0], worlds[1], atol=1e-12)
        np.testing.assert_allclose(worlds[0], worlds[2], atol=1e-12)

    def test_full_open_is_the_tuned_angle(self):
        _, quat = mouth_pose(self.HEAD_POS, np.array([1.0, 0, 0, 0]), 1.0)
        angle = 2 * math.acos(abs(float(quat[0])))
        self.assertAlmostEqual(angle, MOUTH_MAX_RAD, places=9)


class TheMachinesVoice(unittest.TestCase):
    """SayPlayer: the same pipeline, for lines nobody asked for out loud.

    Still silent here — the player is exercised without ever reaching TTS or
    a speaker. What matters is that it cannot stall or crash the 50 Hz control
    loop that hands it a line.
    """

    def player(self):
        import threading
        p = voice.SayPlayer.__new__(voice.SayPlayer)
        p._busy = threading.Lock()
        p.ffmpeg, p.bank_dir, p.player = "ffmpeg", None, None
        p.tts_voice = voice.DEFAULT_TTS_VOICE
        return p

    def test_the_duck_does_not_talk_over_itself(self):
        p = self.player()
        p._busy.acquire()               # pretend a line is playing
        self.assertFalse(p.speak("second thoughts"))

    def test_a_line_that_cannot_be_rendered_releases_the_lock(self):
        # A duck that failed to say one thing must still be able to say the
        # next one; a stuck lock would mute it for the rest of the session.
        p = self.player()
        p.ffmpeg = "/nonexistent/ffmpeg"
        p._busy.acquire()               # as speak() would, before the worker
        with redirect_stderr(io.StringIO()) as err:
            p._run("hello", None)
        self.assertIn("could not say", err.getvalue())
        self.assertTrue(p._busy.acquire(blocking=False))

    def test_a_host_that_cannot_speak_gets_no_player_and_a_note(self):
        with redirect_stderr(io.StringIO()) as err:
            self.assertIsNone(voice.SayPlayer.available(ffmpeg="/nonexistent"))
        self.assertIn("no voice this session", err.getvalue())

    def test_a_line_carries_its_mood_to_the_render(self):
        from unittest import mock
        p = self.player()
        with mock.patch.object(voice, "render_voice") as render:
            with mock.patch.object(p, "_perform"):
                p._busy.acquire()
                p._run("we lost it", None, "sad")
        self.assertEqual(render.call_args.kwargs["mood"], "sad")

    def test_a_moods_typo_in_a_machine_never_reaches_the_control_loop(self):
        # speak() is called from the 50 Hz sim thread. An author's bad mood
        # name is a note on stderr from the worker, not an exception thrown
        # into the walking.
        p = self.player()
        p._busy.acquire()
        with redirect_stderr(io.StringIO()) as err:
            p._run("hello", None, "furious")
        self.assertIn("could not say", err.getvalue())
        self.assertIn("is not a mood", err.getvalue())
        self.assertTrue(p._busy.acquire(blocking=False))

    def test_the_beak_is_left_shut_when_a_line_ends(self):
        import types
        p = self.player()
        sim = types.SimpleNamespace(mouth_opening=0.7)
        p._perform("unused.wav", np.array([0.4, 0.8], dtype=np.float32), sim,
                   rate_hz=1000)
        self.assertEqual(sim.mouth_opening, 0.0)

    def test_it_drives_the_beak_from_the_trajectory(self):
        class Recorder:
            """A sim that only remembers what its beak was told to do."""
            def __init__(self):
                self.seen = []

            def __setattr__(self, name, value):
                if name == "mouth_opening":
                    self.seen.append(value)
                else:
                    object.__setattr__(self, name, value)

        sim = Recorder()
        self.player()._perform("unused.wav",
                               np.array([0.25, 0.5], dtype=np.float32),
                               sim, rate_hz=1000)
        self.assertEqual(sim.seen, [0.25, 0.5, 0.0])


class CliWiring(unittest.TestCase):
    def test_duck_parser_reaches_say_and_mouth(self):
        import contextlib
        from unittest import mock
        from microduck_mcp import client
        with mock.patch.object(client.sys, "argv",
                               ["duck", "say", "hi", "--audio-only"]):
            with mock.patch.object(voice, "run", return_value=0) as run:
                with self.assertRaises(SystemExit):
                    client.main()
        self.assertEqual(run.call_args[0][0].text, "hi")
        self.assertTrue(run.call_args[0][0].audio_only)
        self.assertEqual(run.call_args[0][0].mood, "neutral")
        with mock.patch.object(client.sys, "argv",
                               ["duck", "say", "oh no", "--mood", "sad"]):
            with mock.patch.object(voice, "run", return_value=0) as run:
                with self.assertRaises(SystemExit):
                    client.main()
        self.assertEqual(run.call_args[0][0].mood, "sad")
        with mock.patch.object(client.sys, "argv",
                               ["duck", "chirp", "inquire", "--variant", "1",
                                "--voice-bank", "bank/"]):
            with mock.patch.object(voice, "run_chirp", return_value=0) as run:
                with self.assertRaises(SystemExit):
                    client.main()
        args = run.call_args[0][0]
        self.assertEqual((args.tag, args.variant, args.voice_bank),
                         ("inquire", 1, "bank/"))
        sent = {}
        with mock.patch.object(client.sys, "argv", ["duck", "mouth", "0.6"]):
            with mock.patch.object(client, "request",
                                   side_effect=lambda req, **kw: sent.update(req)
                                   or {"ok": True}):
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        client.main()
        self.assertEqual(sent, {"cmd": "mouth", "opening": 0.6})


if __name__ == "__main__":
    unittest.main()
