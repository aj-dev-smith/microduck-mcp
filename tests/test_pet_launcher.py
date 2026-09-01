"""Tests for the pet launcher's judgment, not its spawning.

`duck pet up` and `down` are mostly subprocess plumbing that only a live
desktop can exercise; what CAN go wrong quietly is the judgment around it —
whose pid a stale file names, which squatter a sweep may shoot, where a
../-relative default lands when typed from the wrong directory. Those are
pure functions, so they get pinned here; nothing in this file starts a
process or opens a socket.
"""

import os
import tempfile
import unittest
from unittest import mock

from microduck_mcp import pet_launcher


class PidfilesForgiveGarbage(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        patcher = mock.patch.object(pet_launcher, "STATE_DIR", self.dir.name)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_missing_pidfile_reads_as_absent(self):
        self.assertIsNone(pet_launcher._read_pid("daemon"))

    def test_a_garbage_pidfile_reads_as_absent_not_as_a_crash(self):
        with open(pet_launcher._pidfile("daemon"), "w") as f:
            f.write("not a pid\n")
        self.assertIsNone(pet_launcher._read_pid("daemon"))

    def test_a_written_pid_round_trips(self):
        with open(pet_launcher._pidfile("overlay"), "w") as f:
            f.write("12345")
        self.assertEqual(pet_launcher._read_pid("overlay"), 12345)


class AlivenessIsHonest(unittest.TestCase):
    def test_our_own_pid_is_alive(self):
        self.assertTrue(pet_launcher._alive(os.getpid()))

    def test_none_and_nonsense_are_dead(self):
        self.assertFalse(pet_launcher._alive(None))
        self.assertFalse(pet_launcher._alive(0))
        self.assertFalse(pet_launcher._alive(-4))

    def test_a_long_gone_pid_is_dead(self):
        # pid_max on macOS is 99998; anything above it can't exist.
        self.assertFalse(pet_launcher._alive(999999))


class TheSweepKnowsItsOwn(unittest.TestCase):
    """The guard between "clean up my strays" and "shoot the squatter"."""

    def test_our_processes_are_recognized_however_they_were_launched(self):
        for cmd in (
            "python -m microduck_mcp.sim_server --scene desktop",
            "/x/.venv/bin/python /x/.venv/bin/duck-sim --web 8410",
            "/x/.venv/bin/duck-pet --port 8410",
        ):
            self.assertTrue(pet_launcher._is_ours(cmd), cmd)

    def test_a_stranger_on_the_port_is_spared(self):
        for cmd in (
            "/Applications/Chrome.app/Contents/MacOS/Chrome --serve",
            "node server.js --port 8410",
            "",
        ):
            self.assertFalse(pet_launcher._is_ours(cmd), cmd)


class RelativeDefaultsFindTheCheckout(unittest.TestCase):
    def test_an_absolute_path_passes_through(self):
        self.assertEqual(pet_launcher._resolve("/etc/hosts"), "/etc/hosts")

    def test_a_cwd_relative_hit_wins(self):
        # This file exists relative to the repo root; resolve from there.
        old = os.getcwd()
        self.addCleanup(os.chdir, old)
        os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        got = pet_launcher._resolve("machines/pet.toml")
        self.assertTrue(os.path.exists(got))

    def test_a_miss_falls_back_to_the_checkout_root(self):
        # From a directory where machines/pet.toml does NOT exist, the
        # default must still land inside the checkout — that is the whole
        # point of the fallback: `duck pet up` typed from anywhere.
        old = os.getcwd()
        self.addCleanup(os.chdir, old)
        with tempfile.TemporaryDirectory() as elsewhere:
            os.chdir(elsewhere)
            got = pet_launcher._resolve("machines/pet.toml")
        self.assertTrue(os.path.exists(got), got)

    def test_a_miss_everywhere_reports_the_cwd_guess(self):
        old = os.getcwd()
        self.addCleanup(os.chdir, old)
        with tempfile.TemporaryDirectory() as elsewhere:
            os.chdir(elsewhere)
            got = pet_launcher._resolve("no/such/thing.toml")
            self.assertEqual(
                got, os.path.join(os.path.realpath(elsewhere),
                                  "no/such/thing.toml"))


if __name__ == "__main__":
    unittest.main()
