"""Live end-to-end tests against a real headless sim subprocess.

Deliberately NOT collected by the offline suite (`tests/live` has no
`__init__.py`, so `discover tests` skips it). These boot `duck-sim`, drive the
duck around in real time, and assert on the reply dicts — slow (~50 s) and
physics-dependent, unlike the pure-maths checks in tests/.

    uv run python -m unittest discover -s tests/live -t tests/live   # from the repo

`-t .` cannot be used: unittest refuses a start dir that is not an importable
package, and an `__init__.py` here is exactly what would make `discover tests`
walk in and boot a sim.
"""

import math
import os
import shutil
import subprocess
import tempfile
import time
import unittest

from microduck_mcp.client import request

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOCK = os.path.join(os.environ.get("TMPDIR", "/tmp"), "livetest.sock")
BOOT_TIMEOUT_S = 30.0
SETTLE_S = 1.2
BALL_HOME = (0.3, 0.0)

STATE_KEYS = ["ok", "sim_time_s", "position_m", "rpy_deg", "vel_body_mps",
              "yaw_rate_rps", "trunk_height_mm", "upright", "active_policy",
              "vel_cmd", "sitting", "behavior", "ground_pick", "ball_seen",
              "ball_position_m", "ball_offset_m"]


def rq(req, timeout=12.0):
    return request(req, sock_path=SOCK, timeout=timeout)


class TestLiveSim(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.path.exists(SOCK):
            os.unlink(SOCK)
        cls.frames = tempfile.mkdtemp(prefix="livetest-frames-")
        env = dict(os.environ, DUCK_SIM_SOCKET=SOCK)
        cls.log = open(os.path.join(cls.frames, "sim.log"), "w+")
        cls.proc = subprocess.Popen(
            ["uv", "run", "duck-sim", "--web", "0", "--frames-dir", cls.frames],
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
                time.sleep(0.4)
        cls.tearDownClass()
        raise RuntimeError(f"sim not ready within {BOOT_TIMEOUT_S:.0f}s")

    @classmethod
    def tearDownClass(cls):
        proc = getattr(cls, "proc", None)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        if getattr(cls, "log", None) is not None:
            cls.log.close()
        if os.path.exists(SOCK):
            os.unlink(SOCK)
        shutil.rmtree(getattr(cls, "frames", ""), ignore_errors=True)

    def setUp(self):
        self.assertTrue(rq({"cmd": "reset"})["ok"])
        time.sleep(SETTLE_S)

    def drive(self, vx=0.0, vy=0.0, wz=0.0, secs=0.0):
        rq({"cmd": "set_velocity", "vx": vx, "vy": vy, "wz": wz})
        if secs:
            time.sleep(secs)

    def stop(self, settle=0.5):
        rq({"cmd": "set_velocity", "vx": 0.0, "vy": 0.0, "wz": 0.0})
        time.sleep(settle)

    # ---------- tests (run in name order) ----------

    def test_1_state_contract(self):
        s = rq({"cmd": "state"})
        for k in STATE_KEYS:
            self.assertIn(k, s)
        seen = s["ball_seen"]
        for k in ("visible", "distance_m", "bearing_deg", "elevation_deg", "age_s"):
            self.assertIn(k, seen)
        self.assertTrue(seen["visible"], f"ball not seen at reset: {seen}")
        self.assertGreaterEqual(seen["distance_m"], 0.2)
        self.assertLessEqual(seen["distance_m"], 0.45)
        self.assertLess(abs(seen["bearing_deg"]), 6.0)
        self.assertTrue(s["upright"])
        self.assertEqual(s["active_policy"], "standing")

    def test_2_drive_forward_moves_the_duck(self):
        x0 = rq({"cmd": "state"})["position_m"][0]
        self.drive(vx=0.3, secs=2.0)
        self.stop()
        s = rq({"cmd": "state"})
        self.assertGreater(s["position_m"][0] - x0, 0.05,
                           f"barely moved: {x0} -> {s['position_m']}")
        self.assertTrue(s["upright"], f"fell over: rpy={s['rpy_deg']}")

    def test_3_turning_loses_sight_of_the_ball(self):
        self.assertTrue(rq({"cmd": "state"})["ball_seen"]["visible"])
        self.drive(wz=1.5, secs=2.5)
        self.stop(settle=0.6)  # > 1 detector period (5 Hz) so ball_seen is fresh
        s = rq({"cmd": "state"})
        self.assertGreater(abs(s["rpy_deg"][2]), 60.0, "barely turned")
        self.assertFalse(s["ball_seen"]["visible"],
                         f"ball still in frame after {s['rpy_deg'][2]} deg turn")

    def test_4_honest_kick_whiffs_and_staged_kick_launches(self):
        # Ball sits 0.3 m ahead, well out of kick reach (staging puts it at
        # x=0.09). An honest kick must swing at nothing and leave it there.
        r = rq({"cmd": "trick", "name": "kick_right", "stage_ball": False})
        self.assertTrue(r["ok"] and r["started"], r)
        time.sleep(3.5)
        b = rq({"cmd": "state"})["ball_position_m"]
        moved = math.hypot(b[0] - BALL_HOME[0], b[1] - BALL_HOME[1])
        self.assertLess(moved, 0.08, f"unstaged kick teleported the ball to {b}")

        rq({"cmd": "reset"})
        time.sleep(SETTLE_S)
        r = rq({"cmd": "trick", "name": "kick_right"})  # stage_ball defaults True
        self.assertTrue(r["ok"] and r["started"], r)
        time.sleep(3.5)
        b = rq({"cmd": "state"})["ball_position_m"]
        moved = math.hypot(b[0] - BALL_HOME[0], b[1] - BALL_HOME[1])
        # Roll-out with the ball's rolling friction enabled (sim_server fixes
        # the geom's condim at load): a clean kick runs ~0.9-1.8 m and STOPS,
        # instead of the old frictionless 3-4 m glide to infinity.
        self.assertGreater(moved, 0.5, f"staged kick barely moved the ball: {b}")

    def test_5_reset_restores_duck_and_ball(self):
        self.drive(vx=0.3, wz=0.6, secs=2.0)
        rq({"cmd": "set_velocity", "vx": 0.0, "vy": 0.0, "wz": 0.0})
        rq({"cmd": "trick", "name": "kick_right"})  # staging displaces the ball
        time.sleep(0.8)
        s = rq({"cmd": "state"})
        self.assertGreater(math.hypot(*s["position_m"][:2]), 0.05, "duck never left origin")
        b = s["ball_position_m"]
        self.assertGreater(math.hypot(b[0] - BALL_HOME[0], b[1] - BALL_HOME[1]), 0.05,
                           "ball never moved, so reset proves nothing")

        s = rq({"cmd": "reset"})
        self.assertLess(math.hypot(*s["position_m"][:2]), 0.05, s["position_m"])
        b = s["ball_position_m"]
        self.assertLess(math.hypot(b[0] - BALL_HOME[0], b[1] - BALL_HOME[1]), 0.05, b)
        self.assertIsNone(s["behavior"])
        self.assertFalse(s["sitting"])

    def test_6_runs_in_real_time(self):
        t0 = time.perf_counter()
        sim0 = rq({"cmd": "ping"})["sim_time_s"]
        time.sleep(10.5)
        sim1 = rq({"cmd": "ping"})["sim_time_s"]
        wall = time.perf_counter() - t0
        rtf = (sim1 - sim0) / wall
        self.assertGreaterEqual(wall, 10.0)
        self.assertGreaterEqual(rtf, 0.9, f"real-time factor {rtf:.3f} "
                                f"({sim1 - sim0:.2f} s sim / {wall:.2f} s wall)")


if __name__ == "__main__":
    unittest.main()
