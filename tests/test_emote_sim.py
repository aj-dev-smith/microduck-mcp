"""Offline tests for playing emotes in the sim: arbitration and the clock.

The engine is pure data (test_emote.py); this is the half that decides who
owns the head and who owns the beak. Like test_wake.py, it runs the real
methods on a bare `DuckSim.__new__` instance — no MuJoCo, no policy, just the
plumbing the 50 Hz loop actually calls.
"""

import os
import threading
import types
import unittest
from collections import deque

import numpy as np

from microduck_mcp.emote import EmoteLibrary
from microduck_mcp.machine import Machine
from microduck_mcp.sim_server import CONTROL_HZ, DuckSim

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMOTES = os.path.join(REPO, "emotes")


class _Policy:
    """Just enough policy to be pointed at something."""

    def __init__(self):
        self.head_offset = np.zeros(4, dtype=np.float32)
        self.head_max = 1.4
        self.commands = 0

    def set_vel_cmd(self, vx, vy, wz):
        pass

    def _update_command(self):
        self.commands += 1


def bare_sim(machine_spec=None, armed=True, voice=None):
    sim = DuckSim.__new__(DuckSim)
    sim.policy = _Policy()
    sim.events = deque(maxlen=500)
    sim._event_id = 0
    sim.sim_time = 4.0
    sim.mouth_opening = 0.0
    sim._mouth_intent_t = 0.0
    sim.voice = voice
    sim.voice_bank = None          # no bank on a test host: sounds are notes
    sim.emotes = EmoteLibrary(EMOTES)
    sim._emote = None
    sim.referee = None
    sim.machine = None
    sim._wake_cond = threading.Condition()
    sim._wakes = deque(maxlen=16)
    sim._wake_id = 0
    if machine_spec is not None:
        sim.machine = Machine(machine_spec)
        sim.machine.armed = armed
    return sim


def play(sim, seconds: float):
    """Run the emote clock forward, one control step at a time."""
    for _ in range(int(seconds * CONTROL_HZ)):
        sim.sim_time += 1.0 / CONTROL_HZ
        sim.emote_tick()


def spec(behavior="idle", **node):
    return {"machine": {"name": "t", "initial": "a"},
            "node": [{"name": "a", "behavior": behavior, **node}]}


class Playing(unittest.TestCase):
    def test_a_gesture_drives_the_head_and_hands_it_back(self):
        sim = bare_sim()
        sim.policy.head_offset[:] = [0.0, 0.5, 0.0, 0.0]   # mid-approach tilt
        resp = sim.start_emote("head_tilt")
        self.assertTrue(resp["ok"])
        self.assertEqual(resp["duration_s"], 1.7)
        play(sim, 0.5)
        self.assertGreater(float(sim.policy.head_offset[3]), 0.2)  # rolled
        play(sim, 2.0)
        self.assertIsNone(sim._emote)
        np.testing.assert_allclose(sim.policy.head_offset, [0.0, 0.5, 0.0, 0.0],
                                   atol=1e-6)

    def test_it_goes_through_the_gaze_command_every_tick(self):
        # Not a servo write: the same path `look` uses, so the balance policy
        # is told about the pose rather than surprised by it.
        sim = bare_sim()
        sim.start_emote("nod")
        before = sim.policy.commands
        play(sim, 0.2)
        self.assertEqual(sim.policy.commands - before, 10)

    def test_the_head_is_clamped_however_the_file_reads(self):
        sim = bare_sim()
        sim.policy.head_max = 0.1
        sim.start_emote("nod")
        play(sim, 0.3)
        self.assertAlmostEqual(float(np.abs(sim.policy.head_offset).max()),
                               0.1, places=6)

    def test_the_beak_moves_and_shuts_again(self):
        sim = bare_sim()
        sim.start_emote("droop")
        play(sim, 1.5)
        self.assertGreater(sim.mouth_opening, 0.1)
        play(sim, 2.2)
        self.assertEqual(sim.mouth_opening, 0.0)

    def test_a_gesture_with_no_bank_is_still_a_gesture(self):
        # The sound is best-effort; the body language is the emote.
        resp = bare_sim().start_emote("head_tilt")
        self.assertTrue(resp["ok"])
        self.assertIn("no voice bank", resp["note"])

    def test_an_unknown_emote_is_a_sentence_not_a_stack_trace(self):
        resp = bare_sim().start_emote("moonwalk")
        self.assertFalse(resp["ok"])
        self.assertIn("no emote named 'moonwalk'", resp["error"])

    def test_a_dropped_gesture_stops_writing_the_head(self):
        # What a reset does to a gesture mid-flight (the reset path itself
        # needs MuJoCo): the emote is dropped, and nothing it was going to
        # restore reaches the head afterwards.
        sim = bare_sim()
        sim.start_emote("droop")
        play(sim, 0.5)
        self.assertIsNotNone(sim._emote)
        sim._emote = None
        sim.policy.head_offset[:] = 0.0
        play(sim, 1.0)
        self.assertEqual(float(np.abs(sim.policy.head_offset).max()), 0.0)


class WhoOwnsTheHead(unittest.TestCase):
    def test_refused_while_a_behavior_is_steering_by_the_camera(self):
        for behavior in ("approach_ball", "kick"):
            with self.subTest(behavior=behavior):
                sim = bare_sim(spec(behavior=behavior))
                resp = sim.start_emote("nod")
                self.assertFalse(resp["ok"])
                self.assertIn("the head belongs to 'a'", resp["error"])
                self.assertIsNone(sim._emote)

    def test_the_machine_may_do_what_a_client_may_not(self):
        # The machine author chose it in source; that is their call to make.
        sim = bare_sim(spec(behavior="approach_ball"))
        self.assertTrue(sim.start_emote("nod", machine=True)["ok"])

    def test_a_disarmed_machine_holds_nothing(self):
        sim = bare_sim(spec(behavior="kick"), armed=False)
        self.assertTrue(sim.start_emote("nod")["ok"])

    def test_other_behaviors_lend_the_head_out(self):
        for behavior in ("idle", "search_ball", "celebrate", "drive"):
            with self.subTest(behavior=behavior):
                self.assertTrue(
                    bare_sim(spec(behavior=behavior)).start_emote("nod")["ok"])

    def test_a_gesture_mid_gesture_is_refused_not_restarted(self):
        sim = bare_sim()
        sim.start_emote("nod")
        play(sim, 0.4)
        resp = sim.start_emote("head_tilt")
        self.assertFalse(resp["ok"])
        self.assertIn("mid-emote (nod)", resp["error"])
        self.assertEqual(sim._emote["name"], "nod")
        play(sim, 1.0)                       # ...and once it finishes, fine
        self.assertTrue(sim.start_emote("head_tilt")["ok"])


class WhoOwnsTheBeak(unittest.TestCase):
    """Say > emote for the mouth, and only for the mouth."""

    def talking(self):
        return types.SimpleNamespace(busy=True)

    def test_the_beak_is_left_to_the_words(self):
        sim = bare_sim(voice=self.talking())
        sim.mouth_opening = 0.6          # mid-syllable
        sim.start_emote("droop")
        play(sim, 1.5)
        self.assertEqual(sim.mouth_opening, 0.6)
        self.assertGreater(float(sim.policy.head_offset[1]), 0.3)  # head still droops

    def test_a_streamed_say_from_elsewhere_also_holds_it(self):
        # `duck say` runs in another process: what the sim sees is a stream of
        # `mouth` intents, and a recent one means the beak is spoken for.
        sim = bare_sim()
        sim.handle({"cmd": "mouth", "opening": 0.4})
        sim.start_emote("droop")
        play(sim, 1.5)
        self.assertEqual(sim.mouth_opening, 0.4)

    def test_a_talking_duck_does_not_fire_the_gesture_s_sound(self):
        sim = bare_sim(voice=self.talking())
        resp = sim.start_emote("droop")
        self.assertTrue(resp["ok"])
        self.assertIn("already talking", resp["note"])

    def test_the_beak_comes_back_when_the_stream_stops(self):
        sim = bare_sim()
        sim.handle({"cmd": "mouth", "opening": 0.4})
        sim.start_emote("droop")
        play(sim, 0.5)
        self.assertEqual(sim.mouth_opening, 0.4)   # the words still have it
        sim._mouth_intent_t = 0.0                  # ...and then they stop
        play(sim, 1.0)
        self.assertGreater(sim.mouth_opening, 0.1)


NODE_SPEC = {
    "machine": {"name": "t", "initial": "watch"},
    "node": [{"name": "watch", "behavior": "idle",
              "transition": [{"when": "ball_seen.visible", "to": "spotted"}]},
             {"name": "spotted", "behavior": "idle",
              "say": "oh!", "emote": "perk_up"}],
}


class EmotingNodes(unittest.TestCase):
    """A node's gesture fires on entry, exactly where its line does."""

    def sim(self, emote="perk_up", behavior="idle"):
        spec = {**NODE_SPEC,
                "node": [NODE_SPEC["node"][0],
                         {**NODE_SPEC["node"][1], "emote": emote,
                          "behavior": behavior}]}
        sim = bare_sim(spec)
        sim._machine_digest = lambda: {"sim_time_s": sim.sim_time,
                                       "ball_seen.visible": True}
        return sim

    def cmds(self, sim):
        return [e["cmd"] for e in sim.events]

    def test_entering_the_node_plays_the_gesture(self):
        sim = self.sim()
        sim.machine_tick()
        self.assertEqual(sim.machine.current, "spotted")
        self.assertEqual(sim._emote["name"], "perk_up")

    def test_the_mouth_says_it_and_the_body_plays_it(self):
        sim = self.sim()
        sim.machine_tick()
        self.assertEqual(self.cmds(sim), ["-> spotted", "say", "emote"])
        said = sim.events[-1]
        self.assertEqual(said["args"]["name"], "perk_up")
        self.assertEqual(said["args"]["node"], "spotted")
        self.assertTrue(said["ok"])

    def test_a_sound_that_stayed_home_says_so_on_the_feed(self):
        # There is no voice bank on a test host, so the gesture plays silently
        # — and the feed carries the reason rather than swallowing it.
        sim = self.sim()
        sim.machine_tick()
        self.assertIn("no voice bank", sim.events[-1]["note"])

    def test_the_machine_gets_the_head_a_client_would_be_refused(self):
        # The author wrote the gesture on the node; declining it here would be
        # second-guessing source.
        sim = self.sim(behavior="approach_ball")
        sim.machine_tick()
        self.assertEqual(sim._emote["name"], "perk_up")

    def test_a_gesture_nobody_has_is_a_note_mid_run(self):
        sim = self.sim(emote="moonwalk")
        sim.machine_tick()
        self.assertEqual(sim.machine.current, "spotted")   # the machine runs on
        self.assertIsNone(sim._emote)
        event = sim.events[-1]
        self.assertEqual(event["cmd"], "emote")
        self.assertFalse(event["ok"])
        self.assertIn("moonwalk", event["note"])

    def test_arming_onto_the_node_does_not_fire_it(self):
        # Same as `say`: entry by arm/force is a jump, not a transition.
        sim = self.sim()
        sim._handle_machine({"action": "force", "node": "spotted"})
        self.assertEqual(sim.machine.current, "spotted")
        self.assertIsNone(sim._emote)


class LoadTimeLint(unittest.TestCase):
    """Naming a gesture the server does not have is a warning, never a
    rejection — the grammar knows nothing about emotes, and a machine has to
    survive landing on a server whose directory differs."""

    def machine(self, emote):
        return Machine({"machine": {"name": "t", "initial": "a"},
                        "node": [{"name": "a", "behavior": "idle",
                                  "emote": emote}]})

    def test_a_gesture_the_server_has_is_quiet(self):
        self.assertEqual(bare_sim()._emote_warnings(self.machine("nod")), [])

    def test_a_gesture_it_does_not_have_is_named(self):
        warnings = bare_sim()._emote_warnings(self.machine("moonwalk"))
        self.assertEqual(len(warnings), 1)
        self.assertIn("'a'", warnings[0])
        self.assertIn("moonwalk", warnings[0])

    def test_a_server_with_no_emotes_at_all_says_which_it_cannot_play(self):
        sim = bare_sim()
        sim.emotes = None
        warnings = sim._emote_warnings(self.machine("nod"))
        self.assertIn("no emote directory", warnings[0])

    def test_the_shipped_resident_names_a_gesture_that_ships(self):
        m = Machine.load(os.path.join(REPO, "machines", "resident.toml"))
        self.assertEqual(bare_sim()._emote_warnings(m), [])


class Listing(unittest.TestCase):
    def test_it_reports_the_directory_and_what_is_playing(self):
        sim = bare_sim()
        sim.start_emote("nod")
        resp = sim.handle({"cmd": "emote", "action": "list"})
        self.assertTrue(resp["ok"])
        self.assertEqual(resp["playing"], "nod")
        self.assertEqual(sorted(e["name"] for e in resp["emotes"]),
                         ["double_take", "droop", "head_tilt", "new_brain",
                          "nod", "nuzzle", "perk_up", "shiver", "yawn"])
        self.assertTrue(all(e["valid"] for e in resp["emotes"]))

    def test_a_server_without_an_emote_directory_says_so(self):
        sim = bare_sim()
        sim.emotes = None
        for req in ({"cmd": "emote", "name": "nod"},
                    {"cmd": "emote", "action": "list"}):
            with self.subTest(req=req):
                resp = sim.handle(req)
                self.assertFalse(resp["ok"])
                self.assertIn("no emote directory", resp["error"])


if __name__ == "__main__":
    unittest.main()
