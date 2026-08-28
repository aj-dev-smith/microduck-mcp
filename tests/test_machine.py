"""Offline tests for the behavior machine: guard grammar, validation, executor,
and the annotations a node may carry on the way past."""

import os
import threading
import unittest
from collections import deque

import numpy as np

from microduck_mcp.machine import (
    GUARD_PATHS, MOOD_NAMES, GuardError, Machine, MachineError, compile_guard)
from microduck_mcp.sim_server import DuckSim

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class GuardGrammar(unittest.TestCase):
    def test_comparisons_and_boolops(self):
        g = compile_guard("ball_seen.visible and ball_seen.age_s < 0.4")
        self.assertTrue(g({"ball_seen.visible": True, "ball_seen.age_s": 0.1}))
        self.assertFalse(g({"ball_seen.visible": True, "ball_seen.age_s": 0.9}))
        self.assertFalse(g({"ball_seen.visible": False, "ball_seen.age_s": 0.1}))

    def test_negative_literals(self):
        g = compile_guard("ball_seen.bearing_deg > -40 and ball_seen.bearing_deg < -12")
        self.assertTrue(g({"ball_seen.bearing_deg": -25}))
        self.assertFalse(g({"ball_seen.bearing_deg": -5}))

    def test_null_fields_compare_false(self):
        # distance while not visible is null: any comparison over it is False,
        # negated or chained — 'unknown', never an exception.
        self.assertFalse(compile_guard("ball_seen.distance_m < 0.5")(
            {"ball_seen.distance_m": None}))
        self.assertFalse(compile_guard("ball_seen.distance_m > 0.5")(
            {"ball_seen.distance_m": None}))

    def test_string_equality(self):
        g = compile_guard("node != 'celebrate'")
        self.assertTrue(g({"node": "kick"}))
        self.assertFalse(g({"node": "celebrate"}))

    def test_not(self):
        g = compile_guard("not upright")
        self.assertTrue(g({"upright": False}))

    def test_rejects_everything_else(self):
        for bad in [
            "__import__('os').system('true')",
            "ball_seen.visible + 1",
            "unknown_path > 2",
            "ball_seen.nope > 2",
            "(1).__class__",
            "[1, 2]",
            "f(1)",
            "ball_seen.visible if 1 else 0",
        ]:
            with self.assertRaises(GuardError, msg=bad):
                compile_guard(bad)

    def test_whitelist_paths_all_compile(self):
        for p in GUARD_PATHS:
            compile_guard(f"{p} == {p}")

    def test_speed_guard_null_means_no_kick(self):
        # speed_mps is null until the detector has tracked the ball ~0.25 s;
        # a "speed_mps < x" entry guard must treat unknown as "don't kick".
        g = compile_guard("ball_seen.speed_mps < 0.25")
        self.assertTrue(g({"ball_seen.speed_mps": 0.04}))
        self.assertFalse(g({"ball_seen.speed_mps": 0.4}))
        self.assertFalse(g({"ball_seen.speed_mps": None}))


class MachineValidation(unittest.TestCase):
    BASE = {"machine": {"name": "t", "initial": "a"},
            "node": [{"name": "a", "behavior": "idle"}]}

    def test_shipped_soccer_machine_loads(self):
        m = Machine.load(os.path.join(REPO, "machines", "soccer.toml"))
        self.assertEqual(m.initial, "search")
        self.assertIn("kick", m.nodes)
        self.assertFalse(m.armed)

    def test_shipped_striker_machine_loads(self):
        # The scoreboard variant: same hunt, but it waits for the referee.
        m = Machine.load(os.path.join(REPO, "machines", "striker.toml"))
        self.assertEqual(m.initial, "search")
        self.assertIn("watch", m.nodes)
        self.assertIn("won", m.nodes)

    def test_unknown_behavior_refused(self):
        spec = {"machine": {"name": "t", "initial": "a"},
                "node": [{"name": "a", "behavior": "moonwalk"}]}
        with self.assertRaisesRegex(MachineError, "moonwalk"):
            Machine(spec)

    def test_unknown_transition_target_refused(self):
        spec = {"machine": {"name": "t", "initial": "a"},
                "node": [{"name": "a", "behavior": "idle",
                          "transition": [{"when": "upright", "to": "ghost"}]}]}
        with self.assertRaisesRegex(MachineError, "ghost"):
            Machine(spec)

    def test_bad_guard_refused_at_load(self):
        spec = {"machine": {"name": "t", "initial": "a"},
                "node": [{"name": "a", "behavior": "idle",
                          "transition": [{"when": "import os", "to": "a"}]}]}
        with self.assertRaises(GuardError):
            Machine(spec)

    def test_missing_initial_refused(self):
        spec = {"machine": {"name": "t", "initial": "nope"},
                "node": [{"name": "a", "behavior": "idle"}]}
        with self.assertRaisesRegex(MachineError, "initial"):
            Machine(spec)


class WakeNodes(unittest.TestCase):
    """The wake grammar: `wake = "reason"` on a node, with ocarina's
    mandatory-default discipline — a deadline transition on elapsed_s, or an
    explicit wake_hold declaring that parking forever is the answer."""

    @staticmethod
    def _spec(node):
        return {"machine": {"name": "t", "initial": "a"},
                "node": [{"name": "a", "behavior": "idle",
                          "transition": [{"when": "upright", "to": "w"}]},
                         node]}

    def test_wake_without_deadline_or_hold_refused(self):
        with self.assertRaisesRegex(MachineError, "deadline"):
            Machine(self._spec({"name": "w", "behavior": "idle",
                                "wake": "something happened"}))

    def test_wake_with_deadline_loads(self):
        m = Machine(self._spec({
            "name": "w", "behavior": "idle", "wake": "something happened",
            "transition": [{"when": "elapsed_s > 300.0", "to": "a"}]}))
        self.assertEqual(m.nodes["w"]["wake"], "something happened")
        self.assertIsNone(m.nodes["a"]["wake"])

    def test_wake_with_hold_loads(self):
        m = Machine(self._spec({"name": "w", "behavior": "idle",
                                "wake": "down", "wake_hold": "safe on floor"}))
        self.assertEqual(m.nodes["w"]["wake_hold"], "safe on floor")

    def test_hold_without_wake_refused(self):
        with self.assertRaisesRegex(MachineError, "wake_hold"):
            Machine(self._spec({"name": "w", "behavior": "idle",
                                "wake_hold": "orphaned hold"}))

    def test_non_string_wake_refused(self):
        with self.assertRaisesRegex(MachineError, "string"):
            Machine(self._spec({"name": "w", "behavior": "idle", "wake": 1}))

    def test_status_lists_wake_nodes(self):
        m = Machine(self._spec({"name": "w", "behavior": "idle",
                                "wake": "hey", "wake_hold": "parked"}))
        self.assertEqual(m.status()["wake_nodes"], ["w"])

    def test_shipped_resident_machine_loads(self):
        m = Machine.load(os.path.join(REPO, "machines", "resident.toml"))
        self.assertEqual(m.initial, "rest")
        self.assertEqual(m.status()["wake_nodes"], ["ball_spotted", "fallen"])

    def test_shipped_striker_wake_nodes(self):
        m = Machine.load(os.path.join(REPO, "machines", "striker.toml"))
        self.assertEqual(m.status()["wake_nodes"], ["down", "won"])


class SpeakingNodes(unittest.TestCase):
    """`say = "..."` on a node: an annotation in the same sense `wake` is.

    The machine says what it is doing; whether anything is listening is
    somebody else's problem. Nothing about the behavior, the guards or the
    physics may depend on it.
    """

    @staticmethod
    def _spec(node):
        return {"machine": {"name": "t", "initial": "a"},
                "node": [{"name": "a", "behavior": "idle",
                          "transition": [{"when": "upright", "to": "s"}]},
                         node]}

    def test_a_node_can_carry_a_line(self):
        m = Machine(self._spec({"name": "s", "behavior": "idle",
                                "say": "Goal!"}))
        self.assertEqual(m.nodes["s"]["say"], "Goal!")

    def test_a_silent_node_says_none_not_nothing(self):
        m = Machine(self._spec({"name": "s", "behavior": "idle"}))
        self.assertIsNone(m.nodes["s"]["say"])
        self.assertIsNone(m.nodes["a"]["say"])

    def test_non_string_refused(self):
        with self.assertRaisesRegex(MachineError, "string"):
            Machine(self._spec({"name": "s", "behavior": "idle", "say": 3}))

    def test_a_monologue_is_refused(self):
        with self.assertRaisesRegex(MachineError, "monologue"):
            Machine(self._spec({"name": "s", "behavior": "idle",
                                "say": "la " * 300}))

    def test_it_needs_no_wake_and_grants_none(self):
        # Speaking and waking are independent: a node may do either, both or
        # neither, and a `say` must not smuggle in a wake node's obligations.
        m = Machine(self._spec({"name": "s", "behavior": "idle",
                                "say": "just talking"}))
        self.assertEqual(m.status()["wake_nodes"], [])
        self.assertEqual(m.status()["say_nodes"], ["s"])

    def test_a_machine_that_talks_still_runs_the_same_behavior(self):
        quiet = Machine(self._spec({"name": "s", "behavior": "idle"}))
        loud = Machine(self._spec({"name": "s", "behavior": "idle",
                                   "say": "hello"}))
        for m in (quiet, loud):
            m.enter("s", 0.0)
        self.assertEqual(quiet.nodes["s"]["behavior"], loud.nodes["s"]["behavior"])
        self.assertEqual(quiet.nodes["s"]["params"], loud.nodes["s"]["params"])
        self.assertEqual(len(quiet.nodes["s"]["transitions"]),
                         len(loud.nodes["s"]["transitions"]))

    def test_an_unknown_node_key_is_ignored(self):
        # Why a machine with `say` is safe to hot-reload onto a server too old
        # to know the key: unknown keys have never been an error here.
        m = Machine(self._spec({"name": "s", "behavior": "idle",
                                "sing": "not a real annotation"}))
        self.assertEqual(m.nodes["s"]["behavior"], "idle")

    def test_the_shipped_striker_speaks_only_where_it_earned_it(self):
        m = Machine.load(os.path.join(REPO, "machines", "striker.toml"))
        self.assertEqual(m.status()["say_nodes"], ["celebrate", "won"])
        # Both routes into `celebrate` are guarded on the referee's call, so
        # the celebration line cannot be said without a goal behind it.
        into = [t["when"] for t in m.global_transitions
                if t["to"] == "celebrate"]
        into += [t["when"] for node in m.nodes.values()
                 for t in node["transitions"] if t["to"] == "celebrate"]
        self.assertTrue(into)
        for when in into:
            self.assertIn("goal.scored", when)

    def test_no_shipped_line_plays_the_wheee(self):
        # The wheee belongs to `duck film`, on the goal moment and nowhere
        # else. The say pipeline speaks words; it does not fire bank sounds.
        for name in ("striker.toml", "soccer.toml", "resident.toml"):
            m = Machine.load(os.path.join(REPO, "machines", name))
            for node, spec in m.nodes.items():
                with self.subTest(machine=name, node=node):
                    self.assertNotIn("wheee", (spec["say"] or "").lower())


class SpokenMoods(unittest.TestCase):
    """`say_mood = "excited"`: the weather on a node's line.

    A separate key rather than a table-valued `say`, and that IS the design —
    a server too old to know the mood speaks the line neutral, where a table
    would have made the line itself unreadable. The roster is closed, though:
    a mood is a fixed vocabulary the renderer implements, not free-form
    content like an emote name.
    """

    @staticmethod
    def _spec(node):
        return {"machine": {"name": "t", "initial": "a"},
                "node": [{"name": "a", "behavior": "idle",
                          "transition": [{"when": "upright", "to": "s"}]},
                         node]}

    def test_a_line_can_carry_a_mood(self):
        m = Machine(self._spec({"name": "s", "behavior": "idle",
                                "say": "Goal!", "say_mood": "excited"}))
        self.assertEqual(m.nodes["s"]["say_mood"], "excited")
        self.assertEqual(m.status()["say_mood_nodes"], ["s"])

    def test_a_plain_line_has_no_mood_and_that_is_neutral(self):
        m = Machine(self._spec({"name": "s", "behavior": "idle",
                                "say": "Goal!"}))
        self.assertIsNone(m.nodes["s"]["say_mood"])
        self.assertEqual(m.status()["say_nodes"], ["s"])
        self.assertEqual(m.status()["say_mood_nodes"], [])

    def test_non_string_refused(self):
        with self.assertRaisesRegex(MachineError, "string"):
            Machine(self._spec({"name": "s", "behavior": "idle",
                                "say": "hi", "say_mood": 2}))

    def test_a_mood_nobody_implements_is_refused_at_load(self):
        # Unlike an emote name, which the server may or may not have: the
        # moods are a closed roster, so a typo is a load error and not a line
        # that quietly comes out flat.
        with self.assertRaisesRegex(MachineError, "unknown say_mood"):
            Machine(self._spec({"name": "s", "behavior": "idle",
                                "say": "hi", "say_mood": "furious"}))

    def test_the_roster_is_the_voices_own(self):
        # Spelled out here (like MAX_SAY_CHARS) so the grammar needs no audio
        # stack to validate a machine — but it must not drift from the voice.
        from microduck_mcp.voice import MOODS
        self.assertEqual(MOOD_NAMES, frozenset(MOODS))

    def test_a_mood_with_nothing_to_say_is_refused(self):
        with self.assertRaisesRegex(MachineError, "nothing to say"):
            Machine(self._spec({"name": "s", "behavior": "idle",
                                "say_mood": "sad"}))

    def test_every_mood_in_the_roster_loads(self):
        for mood in sorted(MOOD_NAMES):
            with self.subTest(mood=mood):
                m = Machine(self._spec({"name": "s", "behavior": "idle",
                                        "say": "hi", "say_mood": mood}))
                self.assertEqual(m.nodes["s"]["say_mood"], mood)

    def test_the_striker_celebrates_in_a_mood(self):
        m = Machine.load(os.path.join(REPO, "machines", "striker.toml"))
        self.assertEqual(m.status()["say_mood_nodes"], ["celebrate"])
        self.assertEqual(m.nodes["celebrate"]["say_mood"], "excited")

    def test_the_other_shipped_machines_speak_neutral(self):
        for name in ("soccer.toml", "resident.toml"):
            with self.subTest(machine=name):
                m = Machine.load(os.path.join(REPO, "machines", name))
                self.assertEqual(m.status()["say_mood_nodes"], [])


class EmotingNodes(unittest.TestCase):
    """`emote = "..."` on a node: `say`'s parallel, and just as ignorant.

    The grammar validates that this is a string and stops there. It does not
    know what emotes exist — a machine naming one the server has never heard
    of must still load, or hot-reloading a machine onto a server would depend
    on the two agreeing about content.
    """

    @staticmethod
    def _spec(node):
        return {"machine": {"name": "t", "initial": "a"},
                "node": [{"name": "a", "behavior": "idle",
                          "transition": [{"when": "upright", "to": "e"}]},
                         node]}

    def test_a_node_can_carry_a_gesture(self):
        m = Machine(self._spec({"name": "e", "behavior": "idle",
                                "emote": "perk_up"}))
        self.assertEqual(m.nodes["e"]["emote"], "perk_up")
        self.assertIsNone(m.nodes["a"]["emote"])

    def test_non_string_refused(self):
        with self.assertRaisesRegex(MachineError, "string"):
            Machine(self._spec({"name": "e", "behavior": "idle", "emote": 7}))

    def test_an_emote_nobody_has_still_loads(self):
        m = Machine(self._spec({"name": "e", "behavior": "idle",
                                "emote": "moonwalk"}))
        self.assertEqual(m.status()["emote_nodes"], ["e"])

    def test_a_node_may_say_and_emote_at_once(self):
        # The whole point: mouth to say, body to emote. They are separate
        # channels, so carrying both is a chord, not a conflict.
        m = Machine(self._spec({"name": "e", "behavior": "idle",
                                "say": "oh!", "emote": "perk_up"}))
        self.assertEqual(m.status()["say_nodes"], ["e"])
        self.assertEqual(m.status()["emote_nodes"], ["e"])

    def test_status_reports_them_alongside_the_speaking_ones(self):
        m = Machine(self._spec({"name": "e", "behavior": "idle"}))
        self.assertEqual(m.status()["emote_nodes"], [])

    def test_the_resident_startles_at_the_ball(self):
        m = Machine.load(os.path.join(REPO, "machines", "resident.toml"))
        self.assertEqual(m.status()["emote_nodes"], ["ball_spotted"])
        self.assertEqual(m.nodes["ball_spotted"]["emote"], "perk_up")

    def test_the_match_machines_are_left_alone(self):
        # striker is the demo machine and approach/kick own the head anyway.
        for name in ("striker.toml", "soccer.toml"):
            with self.subTest(machine=name):
                m = Machine.load(os.path.join(REPO, "machines", name))
                self.assertEqual(m.status()["emote_nodes"], [])


class _StubPolicy:
    def __init__(self):
        self.cmds = []
        self.head_offset = np.zeros(4, dtype=np.float32)
        self.head_max = np.ones(4, dtype=np.float32)

    def set_vel_cmd(self, vx, vy, wz):
        self.cmds.append((vx, vy, wz))

    def _update_command(self):
        pass


class _StubSim:
    def __init__(self):
        self.policy = _StubPolicy()


class Executor(unittest.TestCase):
    def _machine(self):
        return Machine({
            "machine": {"name": "t", "initial": "a",
                        "transition": [{"when": "not upright", "to": "c"}]},
            "node": [
                {"name": "a", "behavior": "idle",
                 "transition": [{"when": "elapsed_s > 1.0", "to": "b"}]},
                {"name": "b", "behavior": "idle"},
                {"name": "c", "behavior": "idle"},
            ]})

    def _digest(self, t, upright=True):
        return {"sim_time_s": t, "upright": upright}

    def test_transition_fires_on_elapsed(self):
        m, sim = self._machine(), _StubSim()
        m.armed = True
        m.enter("a", 0.0)
        self.assertIsNone(m.tick(sim, self._digest(0.5)))
        fired = m.tick(sim, self._digest(1.2))
        self.assertEqual((fired["from"], fired["to"]), ("a", "b"))
        self.assertEqual(m.current, "b")

    def test_elapsed_resets_on_entry(self):
        m, sim = self._machine(), _StubSim()
        m.enter("a", 10.0)
        self.assertIsNone(m.tick(sim, self._digest(10.9)))
        self.assertIsNotNone(m.tick(sim, self._digest(11.1)))

    def test_global_transition_wins(self):
        m, sim = self._machine(), _StubSim()
        m.enter("a", 0.0)
        fired = m.tick(sim, self._digest(5.0, upright=False))
        self.assertEqual(fired["to"], "c")

    def test_no_self_transition_loop(self):
        m, sim = self._machine(), _StubSim()
        m.enter("c", 0.0)
        # global 'not upright -> c' must not re-fire while already in c
        self.assertIsNone(m.tick(sim, self._digest(3.0, upright=False)))

    def test_idle_behavior_zeroes_once(self):
        m, sim = self._machine(), _StubSim()
        m.enter("b", 0.0)
        m.tick(sim, self._digest(0.1))
        m.tick(sim, self._digest(0.2))
        self.assertEqual(sim.policy.cmds, [(0.0, 0.0, 0.0)])


class _Ear:
    """A voice that only remembers what it was asked to say."""

    def __init__(self, boom=False):
        self.heard = []
        self.moods = []
        self.boom = boom

    def speak(self, text, sim=None, mood="neutral"):
        self.heard.append(text)
        self.moods.append(mood)
        if self.boom:
            raise RuntimeError("no speaker on this host")
        return True


SPEAKING_SPEC = {
    "machine": {"name": "t", "initial": "play"},
    "node": [{"name": "play", "behavior": "idle",
              "transition": [{"when": "goal.scored", "to": "won"}]},
             {"name": "won", "behavior": "idle", "say": "Goal!",
              "say_mood": "excited",
              "wake": "a goal deserves company",
              "wake_hold": "parked and safe"}],
}


def speaking_sim(voice=None):
    """A bare DuckSim: no MuJoCo, real event ring, real annotation plumbing."""
    sim = DuckSim.__new__(DuckSim)
    sim.policy = _StubPolicy()
    sim.events = deque(maxlen=500)
    sim._event_id = 0
    sim.sim_time = 12.5
    sim.voice = voice
    sim.machine = Machine(SPEAKING_SPEC)
    sim.machine.armed = True
    sim._wake_cond = threading.Condition()
    sim._wakes = deque(maxlen=16)
    sim._wake_id = 0
    return sim


class SpokenOnEntry(unittest.TestCase):
    """Entering a speaking node forwards the line through the `say` verb.

    The same verb `duck say` uses, so a line the machine decided to say and
    one a person asked for are indistinguishable on the event feed — which is
    what puts machine speech on the film's control-surface feed for free.
    """

    def says(self, sim):
        return [e for e in sim.events if e["cmd"] == "say"]

    def test_the_line_lands_on_the_control_surface(self):
        sim = speaking_sim()
        sim._say_line("won", "Goal!")
        said = self.says(sim)
        self.assertEqual(len(said), 1)
        self.assertEqual(said[0]["client"], "machine")
        self.assertEqual(said[0]["args"]["text"], "Goal!")
        self.assertEqual(said[0]["args"]["node"], "won")
        self.assertTrue(said[0]["ok"])

    def test_a_voiceless_session_still_logs_the_line(self):
        # The robot has a mouth servo, not a speaker: speech is host-side and
        # optional, the annotation is not.
        sim = speaking_sim(voice=None)
        sim._say_line("won", "Goal!")
        self.assertEqual(len(self.says(sim)), 1)

    def test_a_voice_gets_the_line(self):
        ear = _Ear()
        sim = speaking_sim(voice=ear)
        sim._say_line("won", "Goal!")
        self.assertEqual(ear.heard, ["Goal!"])
        self.assertEqual(ear.moods, ["neutral"])

    def test_the_mood_rides_the_line_to_the_voice(self):
        ear = _Ear()
        sim = speaking_sim(voice=ear)
        sim._say_line("won", "Goal!", "excited")
        self.assertEqual(ear.moods, ["excited"])

    def test_the_feed_notes_a_mood_only_when_there_is_one(self):
        # A feed that annotates the ordinary case stops being readable.
        sim = speaking_sim()
        sim._say_line("won", "Goal!")
        sim._say_line("won", "Goal!", "excited")
        self.assertNotIn("mood", self.says(sim)[0]["args"])
        self.assertEqual(self.says(sim)[1]["args"]["mood"], "excited")

    def test_a_voice_that_throws_does_not_reach_the_control_loop(self):
        # _say_line runs in the 50 Hz sim thread. Nothing about talking is
        # allowed to stall the walking.
        ear = _Ear(boom=True)
        sim = speaking_sim(voice=ear)
        sim._say_line("won", "Goal!")
        self.assertEqual(len(self.says(sim)), 1)   # still annotated
        self.assertIsNone(sim.voice)               # and struck off for the session

    def test_a_transition_into_a_speaking_node_speaks(self):
        ear = _Ear()
        sim = speaking_sim(voice=ear)
        sim._machine_digest = lambda: {"sim_time_s": sim.sim_time,
                                       "goal.scored": True}
        sim.machine_tick()
        self.assertEqual(sim.machine.current, "won")
        self.assertEqual(ear.heard, ["Goal!"])
        self.assertEqual(ear.moods, ["excited"])   # the node's say_mood
        # ...and the wake it also carries is untouched by the speaking
        self.assertEqual([e["cmd"] for e in sim.events],
                         ["-> won", "say", "wake"])

    def test_a_silent_node_says_nothing_at_all(self):
        ear = _Ear()
        sim = speaking_sim(voice=ear)
        sim._machine_digest = lambda: {"sim_time_s": sim.sim_time,
                                       "goal.scored": False}
        sim.machine_tick()
        self.assertEqual(sim.machine.current, "play")
        self.assertEqual(ear.heard, [])
        self.assertEqual(self.says(sim), [])


if __name__ == "__main__":
    unittest.main()
