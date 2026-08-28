"""Live end-to-end test of the wake loop against a real headless sim.

Boots its own `duck-sim` (own socket, so it can run alongside or without
test_live_sim.py) and walks the whole residency contract with a tiny
purpose-built machine whose transitions are timed, not sensed — every wake
here is deterministic, no ball required:

    quiet --1s--> notice (wake, 2s deadline) --> parked (wake + wake_hold)

Covered live: the blocking arm, a second wake picked up by a later wait, the
honest no_wake on a drained latch, the entry wake from `force`, a wake slept
through until its deadline default ran (delivered late, with `resolved`
carrying the machine's own answer), and the disarmed error.
"""

import os
import subprocess
import tempfile
import time
import unittest

from microduck_mcp.client import request

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOCK = os.path.join(os.environ.get("TMPDIR", "/tmp"), "livetest-wake.sock")
BOOT_TIMEOUT_S = 30.0

MACHINE_TOML = """\
[machine]
name = "waketest"
initial = "quiet"

[[node]]
name = "quiet"
behavior = "idle"
[[node.transition]]
when = "elapsed_s > 1.0"
to = "notice"

[[node]]
name = "notice"
behavior = "idle"
wake = "noticed something"
[[node.transition]]
when = "elapsed_s > 2.0"
to = "parked"

[[node]]
name = "parked"
behavior = "idle"
wake = "parked for good"
wake_hold = "held for the mind"
"""


def rq(req, timeout=12.0):
    return request(req, sock_path=SOCK, timeout=timeout)


def machine(action, **kw):
    timeout = kw.get("block_s", 0.0) + 10.0
    return rq({"cmd": "machine", "action": action, **kw}, timeout=timeout)


class TestLiveWake(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.path.exists(SOCK):
            os.unlink(SOCK)
        cls.tmp = tempfile.mkdtemp(prefix="livetest-wake-")
        cls.machine_path = os.path.join(cls.tmp, "waketest.toml")
        with open(cls.machine_path, "w") as f:
            f.write(MACHINE_TOML)
        env = dict(os.environ, DUCK_SIM_SOCKET=SOCK)
        cls.log = open(os.path.join(cls.tmp, "sim.log"), "w+")
        cls.proc = subprocess.Popen(
            ["uv", "run", "duck-sim", "--web", "0",
             "--frames-dir", os.path.join(cls.tmp, "frames")],
            cwd=REPO, env=env, stdout=cls.log, stderr=subprocess.STDOUT)
        deadline = time.time() + BOOT_TIMEOUT_S
        while time.time() < deadline:
            if cls.proc.poll() is not None:
                cls.log.seek(0)
                raise RuntimeError(f"duck-sim exited early:\n{cls.log.read()}")
            try:
                if rq({"cmd": "ping"}, timeout=2.0).get("ok"):
                    return
            except (ConnectionRefusedError, FileNotFoundError, OSError):
                time.sleep(0.5)
        raise RuntimeError("duck-sim did not come up in time")

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        cls.proc.wait(timeout=10)
        cls.log.close()

    def test_reset_reenters_machine_node(self):
        """A reset rewinds the sim clock and the world; the machine must
        re-enter its current node (fresh per-node memory, fresh entry clock).
        Regression: before the fix, entered_at outlived the rewound clock, so
        elapsed_s went negative and every timed guard froze — observed live
        as post-reset ghost detours to stale world-frame targets."""
        self.assertTrue(machine("load", path=self.machine_path)["ok"])
        self.assertTrue(machine("arm")["ok"])
        time.sleep(4.0)  # machine settles: quiet -> notice -> parked
        # Jump back into notice, giving it a LARGE entered_at, then drain
        # every parked pack (notice's, parked's, and this entry wake).
        self.assertTrue(machine("force", node="notice")["ok"])
        for _ in range(10):
            if machine("wait", block_s=0.0).get("no_wake"):
                break
        else:
            self.fail("wake latch would not drain")
        # Reset within notice's 2 s deadline window: the sim clock rewinds
        # to ~0 while entered_at reads several seconds.
        self.assertTrue(rq({"cmd": "reset"})["ok"])
        status = machine("status")
        self.assertTrue(status["armed"])
        self.assertEqual(status["node"], "notice")  # same node, fresh entry
        # Fixed: the re-entered deadline fires ~2 s after the reset and
        # parks. Broken: elapsed_s is negative and 6 s of listening times out.
        resp = machine("wait", block_s=6.0)
        self.assertIsNotNone(resp.get("wake"),
                             f"deadline never fired after reset: {resp}")
        self.assertEqual(resp["wake"]["reason"], "parked for good")
        self.assertTrue(machine("disarm")["ok"])

    def test_wake_loop(self):
        with self.subTest("load reports wake nodes"):
            resp = machine("load", path=self.machine_path)
            self.assertTrue(resp["ok"], resp)
            self.assertEqual(resp["wake_nodes"], ["notice", "parked"])

        with self.subTest("blocking arm returns the first wake"):
            t0 = time.monotonic()
            resp = machine("arm", block_s=10.0)
            self.assertTrue(resp["ok"], resp)
            w = resp["wake"]
            self.assertEqual(w["reason"], "noticed something")
            self.assertEqual(w["via"], {"from": "quiet",
                                        "when": "elapsed_s > 1.0"})
            self.assertIsNone(w["resolved"])  # consumed before the deadline
            self.assertTrue(w["digest"]["upright"])
            self.assertEqual(w["events"][-1]["cmd"], "wake")
            self.assertLess(time.monotonic() - t0, 8.0)  # woken, not timed out

        with self.subTest("a later wait picks up the next wake"):
            w = machine("wait", block_s=6.0)["wake"]
            self.assertEqual(w["reason"], "parked for good")
            self.assertEqual(w["via"]["from"], "notice")
            self.assertIsNone(w["resolved"])  # wake_hold: parked never exits

        with self.subTest("drained latch answers no_wake honestly"):
            resp = machine("wait", block_s=0.0)
            self.assertTrue(resp["ok"], resp)
            self.assertTrue(resp.get("no_wake"))

        with self.subTest("force into a wake node latches an entry wake"):
            self.assertTrue(machine("force", node="notice")["ok"])
            w = machine("wait", block_s=5.0)["wake"]
            self.assertEqual(w["reason"], "noticed something")
            self.assertEqual(w["via"], {"action": "force"})
            # ...and 2 s later the deadline marches it on to parked again
            self.assertEqual(machine("wait", block_s=5.0)["wake"]["reason"],
                             "parked for good")

        with self.subTest("a wake slept through carries the machine's answer"):
            self.assertTrue(machine("arm")["ok"])  # restart at quiet, no block
            time.sleep(4.0)  # notice wakes at ~1 s, its deadline runs at ~3 s
            w = machine("wait", block_s=5.0)["wake"]
            self.assertEqual(w["reason"], "noticed something")
            self.assertEqual(w["resolved"]["to"], "parked")
            self.assertEqual(w["resolved"]["when"], "elapsed_s > 2.0")
            # the follow-on parked wake is in the queue behind it
            self.assertEqual(machine("wait", block_s=5.0)["wake"]["reason"],
                             "parked for good")

        with self.subTest("waiting on a disarmed machine is a loud error"):
            self.assertTrue(machine("disarm")["ok"])
            resp = machine("wait", block_s=0.0)
            self.assertFalse(resp["ok"])
            self.assertIn("disarmed", resp["error"])


if __name__ == "__main__":
    unittest.main()
