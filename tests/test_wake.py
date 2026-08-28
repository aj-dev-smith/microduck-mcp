"""Offline tests for the wake latch: park, long-poll, deadline resolution.

The latch is plain threading state on DuckSim (no MuJoCo in the path), so a
bare instance exercises the real methods: _latch_wake from a "sim thread",
machine_wait from a "connection thread"."""

import threading
import time
import types
import unittest
from collections import deque

from microduck_mcp.sim_server import DuckSim


def bare_sim(armed=True, machine=True):
    sim = DuckSim.__new__(DuckSim)
    sim._wake_cond = threading.Condition()
    sim._wakes = deque(maxlen=16)
    sim._wake_id = 0
    sim.events = deque(maxlen=500)
    sim._event_id = 0
    sim.sim_time = 7.5
    sim.machine = types.SimpleNamespace(armed=armed) if machine else None
    return sim


DIGEST = {"upright": True, "sim_time_s": 7.5, "elapsed_s": 0.0}


class WakeLatch(unittest.TestCase):
    def test_parked_wake_returned_immediately(self):
        sim = bare_sim()
        sim._latch_wake("won", "GOAL", {"from": "celebrate", "when": "upright"},
                        dict(DIGEST))
        resp = sim.machine_wait(block_s=0.0)
        self.assertTrue(resp["ok"])
        w = resp["wake"]
        self.assertEqual((w["node"], w["reason"]), ("won", "GOAL"))
        self.assertEqual(w["via"]["from"], "celebrate")
        self.assertEqual(w["digest"]["upright"], True)
        self.assertIsNone(w["resolved"])
        self.assertEqual(w["events"][-1]["cmd"], "wake")  # the tail rides along
        # consumed: the latch is empty again
        self.assertTrue(sim.machine_wait(block_s=0.0).get("no_wake"))

    def test_fifo_order(self):
        sim = bare_sim()
        sim._latch_wake("a", "first", {}, dict(DIGEST))
        sim._latch_wake("b", "second", {}, dict(DIGEST))
        self.assertEqual(sim.machine_wait(0.0)["wake"]["reason"], "first")
        self.assertEqual(sim.machine_wait(0.0)["wake"]["reason"], "second")

    def test_no_machine_is_an_error(self):
        resp = bare_sim(machine=False).machine_wait(block_s=0.0)
        self.assertFalse(resp["ok"])
        self.assertIn("no machine", resp["error"])

    def test_disarmed_is_an_error(self):
        resp = bare_sim(armed=False).machine_wait(block_s=0.0)
        self.assertFalse(resp["ok"])
        self.assertIn("disarmed", resp["error"])

    def test_parked_wake_beats_disarm(self):
        # A wake that fired before a disarm is still delivered — the mind
        # hears what happened, then learns the machine is off.
        sim = bare_sim(armed=False)
        sim._latch_wake("down", "fell", {}, dict(DIGEST))
        self.assertTrue(sim.machine_wait(0.0)["ok"])

    def test_timeout_returns_honest_no_wake(self):
        resp = bare_sim().machine_wait(block_s=0.2)
        self.assertTrue(resp["ok"])
        self.assertTrue(resp["no_wake"])
        self.assertGreaterEqual(resp["waited_s"], 0.2)

    def test_blocked_waiter_released_by_latch(self):
        sim = bare_sim()
        out = {}

        def waiter():
            out["resp"] = sim.machine_wait(block_s=10.0)

        t = threading.Thread(target=waiter)
        t.start()
        time.sleep(0.15)  # let it block
        t0 = time.monotonic()
        sim._latch_wake("ball_spotted", "ball sighted", {}, dict(DIGEST))
        t.join(timeout=5.0)
        self.assertFalse(t.is_alive())
        self.assertLess(time.monotonic() - t0, 2.0)  # released, not timed out
        self.assertEqual(out["resp"]["wake"]["node"], "ball_spotted")

    def test_deadline_resolution_marks_parked_pack(self):
        # Nobody listened; the machine's deadline default ran. The late
        # listener still gets the pack — with the machine's own answer in it.
        sim = bare_sim()
        sim._latch_wake("ball_spotted", "ball sighted", {}, dict(DIGEST))
        sim._resolve_wakes("ball_spotted",
                           {"to": "rest", "when": "elapsed_s > 300.0"})
        w = sim.machine_wait(0.0)["wake"]
        self.assertEqual(w["resolved"]["to"], "rest")


if __name__ == "__main__":
    unittest.main()
