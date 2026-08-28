"""Offline tests for `duck film`: take selection, overlays, the encoder gate.

No sim and no match — filming one takes minutes of MuJoCo. What is testable
without it is everything that decides *which* take becomes the film and *what*
the overlays say, plus the ffmpeg preflight, which exists precisely so a
missing encoder fails in a second instead of after a shoot.

    uv run --with pytest pytest tests/ --ignore=tests/live
"""

import os
import shutil
import tempfile
import unittest
from collections import deque

import numpy as np

from microduck_mcp import film
from microduck_mcp.film import (DISCARD, FALLBACK, FPS, KEEP, MATCH_SPAWNS,
                                FilmError, MatchFilm, Take, classify_take,
                                encode, feed_lines, find_ffmpeg, hud_lines,
                                resolve_machine, select_take, take_path)

HAVE_FFMPEG = shutil.which("ffmpeg") is not None


def take(index=0, scored=True, node="won"):
    return Take(index=index, spawn=MATCH_SPAWNS[0], path=f"take{index}.mp4",
                scored=scored, node=node, duration_s=42.0)


class TakeSelection(unittest.TestCase):
    def test_goal_plus_landed_celebration_is_the_keeper(self):
        self.assertEqual(classify_take(take(node="won")), KEEP)

    def test_goal_that_ends_on_its_side_is_plan_b(self):
        self.assertEqual(classify_take(take(node="down")), FALLBACK)

    def test_no_goal_is_no_take(self):
        for node in ("won", "down", "search", "watch"):
            with self.subTest(node=node):
                self.assertEqual(classify_take(take(scored=False, node=node)),
                                 DISCARD)

    def test_select_goal_takes_any_goal_outright(self):
        self.assertEqual(classify_take(take(node="down"), prefer_won=False),
                         KEEP)
        # ...but a take that never scored is still not a film.
        self.assertEqual(classify_take(take(scored=False), prefer_won=False),
                         DISCARD)

    def test_keeper_wins_over_an_earlier_fallback(self):
        takes = [take(0, node="down"), take(1, node="won")]
        self.assertEqual(select_take(takes).index, 1)

    def test_first_keeper_wins_over_a_later_one(self):
        takes = [take(0, node="won"), take(1, node="won")]
        self.assertEqual(select_take(takes).index, 0)

    def test_fallback_when_nothing_landed(self):
        takes = [take(0, scored=False, node="down"), take(1, node="down"),
                 take(2, node="down")]
        self.assertEqual(select_take(takes).index, 1)

    def test_nothing_scored_is_no_film(self):
        self.assertIsNone(select_take([take(0, scored=False),
                                       take(1, scored=False)]))
        self.assertIsNone(select_take([]))

    def test_take_paths_sit_beside_the_output(self):
        # Same directory as the output so the winner is a rename, not a copy
        # across filesystems; hidden unless the rushes are being kept.
        hidden = take_path("/films/duck_match.mp4", 3, keep=False)
        kept = take_path("/films/duck_match.mp4", 3, keep=True)
        self.assertEqual(hidden, "/films/.duck_match.take3.mp4")
        self.assertEqual(kept, "/films/duck_match.take3.mp4")

    def test_take_paths_are_absolute_even_for_a_bare_filename(self):
        self.assertTrue(os.path.isabs(take_path("out.mp4", 0, keep=False)))


class SensedStateHud(unittest.TestCase):
    """The HUD is the machine's guard vocabulary, rendered. Nulls must read as
    nulls — a blank where a number should be is the honest thing to show."""

    FULL = {"ball_seen.est_forward_m": 0.213, "ball_seen.est_left_m": -0.068,
            "ball_seen.speed_mps": 0.021, "goal_seen.est_bearing_deg": -34.2,
            "goal_seen.est_distance_m": 1.184}
    EMPTY = dict.fromkeys(FULL)

    def test_numbers_are_signed_and_column_aligned(self):
        fwd, spd, goal = hud_lines(self.FULL)
        self.assertEqual(fwd, "ball fwd +0.213 left -0.068")
        self.assertEqual(spd, "     spd 0.021 m/s")
        self.assertEqual(goal, "goal brg -34.2deg dist 1.18")

    def test_unseen_reads_as_a_placeholder_not_a_zero(self):
        for line in hud_lines(self.EMPTY):
            with self.subTest(line=line):
                self.assertIn("--", line)
                self.assertNotIn("0.000", line)

    def test_lines_keep_a_stable_width_as_values_move(self):
        # The HUD sits in a fixed box under the duck cam; a sign flip must not
        # reflow it.
        a = hud_lines(self.FULL)
        b = hud_lines({**self.FULL, "ball_seen.est_forward_m": -0.213,
                       "ball_seen.est_left_m": 0.068})
        self.assertEqual([len(x) for x in a], [len(x) for x in b])


class ControlSurfaceFeed(unittest.TestCase):
    """What scrolls along the bottom of the film is the sim's real event ring:
    the MCP calls, the transitions, and the guard that fired them."""

    MCP = {"client": "mcp", "cmd": "machine",
           "args": {"action": "load", "path": "machines/striker.toml"}}
    ARM = {"client": "mcp", "cmd": "machine", "args": {"action": "arm"}}
    TRANSITION = {"client": "machine", "cmd": "-> kick",
                  "args": {"from": "approach",
                           "when": "ball_seen.est_left_m < -0.056"}}
    GOAL = {"client": "referee", "cmd": "GOAL!", "args": {"count": 1}}

    def test_mcp_lines_read_like_the_command_a_human_would_type(self):
        lines, _ = feed_lines([self.MCP])
        self.assertEqual(lines[0][1], "duck machine load machines/striker.toml")

    def test_an_argless_mcp_call_does_not_trail_whitespace(self):
        lines, _ = feed_lines([self.ARM])
        self.assertEqual(lines[0][1], "duck machine arm")

    def test_a_transition_shows_where_it_came_from(self):
        lines, _ = feed_lines([self.TRANSITION])
        self.assertEqual(lines[0][1], "approach -> kick")

    def test_the_guard_expression_travels_with_the_transition(self):
        _, guard = feed_lines([self.TRANSITION])
        self.assertEqual(guard, "ball_seen.est_left_m < -0.056")

    def test_the_newest_guard_wins(self):
        later = {"client": "machine", "cmd": "-> watch",
                 "args": {"from": "kick", "when": "elapsed_s > 1.2"}}
        _, guard = feed_lines([self.TRANSITION, later])
        self.assertEqual(guard, "elapsed_s > 1.2")

    def test_no_transition_in_view_means_no_guard_line(self):
        _, guard = feed_lines([self.MCP, self.GOAL])
        self.assertIsNone(guard)

    def test_the_referee_calls_the_score(self):
        lines, _ = feed_lines([self.GOAL])
        self.assertEqual(lines[0][1], "GOAL! #1")

    def test_other_clients_do_not_scroll_past(self):
        # The AX page shows cli/web traffic too; the film is about the machine.
        lines, _ = feed_lines([{"client": "web", "cmd": "state", "args": {}},
                               {"client": "cli", "cmd": "state", "args": {}}])
        self.assertEqual(lines, [])

    def test_only_the_tail_of_the_ring_is_read(self):
        events = deque([self.MCP] * 40 + [self.TRANSITION], maxlen=500)
        lines, guard = feed_lines(events)
        self.assertEqual(len(lines), film.FEED_TAIL)
        self.assertEqual(guard, self.TRANSITION["args"]["when"])

    def test_each_client_gets_its_own_colour(self):
        lines, _ = feed_lines([self.MCP, self.TRANSITION, self.GOAL])
        self.assertEqual([c for _, _, c in lines],
                         [film.FEED_COLORS["mcp"], film.FEED_COLORS["machine"],
                          film.FEED_COLORS["referee"]])


class TheCut(unittest.TestCase):
    """The goal moment cold-opens the film so it becomes the thumbnail."""

    def cutter(self, cards=False):
        cut = object.__new__(MatchFilm)   # the cut needs no sim, no renderers
        cut.cards = cards
        if cards:
            cut.fonts = film.Fonts.load()
        return cut

    def test_the_goal_frame_opens_the_film(self):
        frames = list(range(400))
        goal_i = 200
        seq = self.cutter().cut(frames, goal_t=goal_i / FPS, t0=0.0)
        opener = frames[goal_i + int(1.1 * FPS)]  # 1.1 s after the ball crosses
        self.assertEqual(seq[:int(FPS)], [opener] * int(FPS))
        self.assertEqual(seq[int(FPS):], frames)

    def test_a_short_take_clamps_to_the_last_frame(self):
        frames = list(range(10))
        seq = self.cutter().cut(frames, goal_t=9.0, t0=0.0)
        self.assertEqual(seq[0], frames[-1])

    def test_cards_are_appended_not_prepended(self):
        frames = [np.zeros((4, 4, 3), dtype=np.uint8)] * 5
        with_cards = self.cutter(cards=True).cut(frames, goal_t=0.1, t0=0.0)
        without = self.cutter(cards=False).cut(frames, goal_t=0.1, t0=0.0)
        self.assertGreater(len(with_cards), len(without))
        self.assertTrue(np.array_equal(with_cards[len(without) - 1],
                                       without[-1]))


class TheSoundClock(unittest.TestCase):
    """Sim time -> the finished film's clock, across the cold open."""

    def frames(self, n=100):
        return [i / FPS for i in range(n)]

    def test_the_cold_open_pushes_everything_back(self):
        at = film.frame_clock(self.frames())
        self.assertAlmostEqual(at(0.0), film.HOOK_FRAMES / FPS)

    def test_a_beat_lands_on_the_frame_the_viewer_is_watching(self):
        at = film.frame_clock(self.frames())
        # the beat at sim t=3 s is on frame 50; the viewer sees it 1 s later
        self.assertAlmostEqual(at(3.0), (film.HOOK_FRAMES + 50) / FPS)

    def test_a_beat_after_the_last_frame_clamps_into_the_film(self):
        at = film.frame_clock(self.frames(10))
        self.assertAlmostEqual(at(900.0), (film.HOOK_FRAMES + 9) / FPS)

    def test_a_take_with_no_frames_does_not_divide_by_anything(self):
        self.assertAlmostEqual(film.frame_clock([])(7.0),
                               film.HOOK_FRAMES / FPS)

    def test_beats_keep_their_order_on_the_new_clock(self):
        at = film.frame_clock(self.frames())
        times = [at(t) for t in (0.0, 1.5, 3.0, 5.9)]
        self.assertEqual(times, sorted(times))


class TheScript(unittest.TestCase):
    def args(self, **kw):
        import argparse
        p = argparse.ArgumentParser()
        sub = p.add_subparsers(dest="command")
        film.add_arguments(sub.add_parser("film"))
        argv = ["film"]
        for flag, value in kw.items():
            name = f"--{flag.replace('_', '-')}"
            argv += [name] if value is True else [name, value]
        return p.parse_args(argv)

    def test_zero_config_uses_the_shipped_lines(self):
        from microduck_mcp import soundtrack
        self.assertEqual(film.script_lines(self.args()),
                         soundtrack.DEFAULT_LINES)

    def test_a_line_can_be_rewritten(self):
        lines = film.script_lines(self.args(line_goal="get in"))
        self.assertEqual(lines["goal"], "get in")
        self.assertEqual(lines["arm"],
                         film.script_lines(self.args())["arm"])

    def test_an_empty_line_is_kept_as_a_deletion(self):
        # argparse cannot tell "" from "unset" for us; script_lines must.
        self.assertEqual(film.script_lines(self.args(line_arm=""))["arm"], "")

    def test_no_audio_films_silently_without_touching_the_toolchain(self):
        self.assertIsNone(film.build_kit(self.args(no_audio=True),
                                         ffmpeg="/nonexistent",
                                         work_dir="/nonexistent"))

    def test_a_toolchain_that_renders_nothing_is_a_silent_film_not_a_crash(self):
        # No usable ffmpeg and no bank: every stage degrades, and the shoot
        # ends up with no kit rather than with a broken one.
        kit = film.build_kit(self.args(voice_bank="/nonexistent/bank"),
                             ffmpeg="/nonexistent", work_dir="/nonexistent")
        self.assertIsNone(kit)


class ASilentFilm(unittest.TestCase):
    """The soundtrack is an enhancement: without one, nothing changes."""

    def shoot(self, kit=None):
        s = object.__new__(MatchFilm)
        s.kit = kit
        s.quiet = True
        s.ffmpeg = "ffmpeg"
        return s

    def test_dubbing_a_kitless_shoot_does_nothing(self):
        # No kit, no ffmpeg call, no exception, and the mp4 is left alone.
        self.shoot().dub("/nonexistent/film.mp4", [], [], 10)

    def test_a_failed_dub_leaves_the_film_alone(self):
        from microduck_mcp import soundtrack
        kit = object.__new__(soundtrack.SoundKit)
        kit.lines, kit.voicings = dict(soundtrack.DEFAULT_LINES), {}
        kit.wheee, kit.chirp = None, np.ones(10, dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "film.mp4")
            with open(path, "wb") as f:
                f.write(b"not really an mp4")
            shoot = self.shoot(kit)
            shoot.ffmpeg = "/nonexistent/ffmpeg"
            shoot.dub(path, [film._soundtrack().Beat("node", 1.0, "kick")],
                      [0.0, 1.0, 2.0], 30)
            with open(path, "rb") as f:
                self.assertEqual(f.read(), b"not really an mp4")

    def test_the_beak_is_shut_when_nobody_is_speaking(self):
        self.assertIsNone(film._soundtrack().mouth_at([], 1.0))


class FfmpegPreflight(unittest.TestCase):
    def test_a_missing_encoder_says_what_to_install(self):
        with self.assertRaises(FilmError) as caught:
            find_ffmpeg("ffmpeg-that-is-not-installed")
        message = str(caught.exception)
        self.assertIn("ffmpeg-that-is-not-installed", message)
        self.assertIn("not a Python dependency", message)
        self.assertIn("brew install ffmpeg", message)
        self.assertIn("--ffmpeg", message)

    @unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not installed")
    def test_a_present_encoder_resolves_to_a_path(self):
        self.assertTrue(os.path.isabs(find_ffmpeg("ffmpeg")))

    @unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not installed")
    def test_raw_frames_go_in_and_an_mp4_comes_out(self):
        frames = [np.full((32, 64, 3), i * 8, dtype=np.uint8) for i in range(10)]
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "clip.mp4")
            encode(frames, out, "ffmpeg", width=64, height=32, fps=10)
            self.assertGreater(os.path.getsize(out), 0)

    @unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not installed")
    def test_an_encoder_that_fails_is_not_reported_as_a_film(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FilmError):
                # Frames whose bytes do not match the declared geometry: the
                # encoder rejects the stream and the take must not be kept.
                encode([np.zeros((3, 3, 3), dtype=np.uint8)],
                       os.path.join(d, "clip.mp4"), "ffmpeg",
                       width=64, height=32, fps=10)


class MachineResolution(unittest.TestCase):
    def test_the_shipped_striker_is_found_from_anywhere(self):
        # Zero-config: `duck film` in some other directory still finds the
        # machine that ships with the repo.
        cwd = os.getcwd()
        try:
            os.chdir(tempfile.gettempdir())
            path = resolve_machine("machines/striker.toml")
        finally:
            os.chdir(cwd)
        self.assertTrue(os.path.isfile(path))
        self.assertTrue(path.endswith(os.path.join("machines", "striker.toml")))

    def test_an_unknown_machine_says_where_it_looked(self):
        with self.assertRaises(FilmError) as caught:
            resolve_machine("machines/nope.toml")
        self.assertIn("nope.toml", str(caught.exception))


class Wiring(unittest.TestCase):
    def test_film_is_a_duck_subcommand(self):
        # The point of the whole exercise: `duck film` alongside `duck machine`.
        import argparse
        p = argparse.ArgumentParser()
        sub = p.add_subparsers(dest="command")
        film.add_arguments(sub.add_parser("film"))
        args = p.parse_args(["film"])
        self.assertEqual(args.output, film.DEFAULT_OUTPUT)
        self.assertEqual(args.machine, film.DEFAULT_MACHINE)
        self.assertEqual(args.select, "won")
        self.assertEqual(args.scene, "pitch")
        self.assertEqual(args.takes, len(MATCH_SPAWNS))
        # Sound is on by default, with nothing to configure.
        self.assertFalse(args.no_audio)
        self.assertIsNone(args.line_arm)

    def test_the_real_cli_parser_knows_the_subcommand(self):
        # Reaches the film subparser (which rejects the flag) rather than the
        # top-level one (which would reject the command) — no sim, no socket.
        import contextlib
        import io
        import sys as _sys

        from microduck_mcp import client
        argv, stderr = _sys.argv, io.StringIO()
        _sys.argv = ["duck", "film", "--select", "no-such-policy"]
        try:
            with contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as caught:
                    client.main()
        finally:
            _sys.argv = argv
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("duck film", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
