"""Offline tests for the behavior machine: guard grammar, validation, executor."""

import os
import unittest

import numpy as np

from microduck_mcp.machine import (
    GUARD_PATHS, GuardError, Machine, MachineError, compile_guard)

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


class MachineValidation(unittest.TestCase):
    BASE = {"machine": {"name": "t", "initial": "a"},
            "node": [{"name": "a", "behavior": "idle"}]}

    def test_shipped_soccer_machine_loads(self):
        m = Machine.load(os.path.join(REPO, "machines", "soccer.toml"))
        self.assertEqual(m.initial, "search")
        self.assertIn("kick", m.nodes)
        self.assertFalse(m.armed)

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


if __name__ == "__main__":
    unittest.main()
