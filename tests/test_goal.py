"""Offline tests for the referee: the goal predicate, the latch, the wiring.

No MuJoCo model — the pitch scene lives in microduck_rl, so everything here
runs against the pure geometry (ball_in_goal), the scoreboard (GoalReferee)
and stub sims standing in for the real one, the same way test_machine.py does.

    uv run --with pytest pytest tests/
"""

import unittest
from collections import deque

import mujoco
import numpy as np

from microduck_mcp.machine import GUARD_PATHS
from microduck_mcp.sim_server import (BALL_RADIUS_M, GOAL_GEOM_NAMES,
                                      GOAL_HALF_WIDTH_Y, GOAL_HEIGHT_Z,
                                      GOAL_LINE_X, GOAL_NET_BACK_X,
                                      GOAL_NOT_SEEN, DuckSim, GoalReferee,
                                      ball_in_goal, find_goal_geom)

IN = GOAL_LINE_X + BALL_RADIUS_M + 0.02  # comfortably over the line


class GoalPredicate(unittest.TestCase):
    def test_fully_across_and_centred_is_a_goal(self):
        self.assertTrue(ball_in_goal(IN, 0.0, BALL_RADIUS_M))

    def test_short_of_the_line_is_not(self):
        self.assertFalse(ball_in_goal(GOAL_LINE_X - 0.05, 0.0, BALL_RADIUS_M))

    def test_sitting_on_the_line_is_not_yet(self):
        # The whole ball has to cross: centre exactly on the line still has
        # half of it in play.
        self.assertFalse(ball_in_goal(GOAL_LINE_X, 0.0, BALL_RADIUS_M))

    def test_last_touch_on_the_line_is_not(self):
        # Centre one radius past the line: the trailing edge is *on* the line,
        # not over it.
        self.assertFalse(ball_in_goal(GOAL_LINE_X + BALL_RADIUS_M, 0.0, 0.03))
        self.assertTrue(ball_in_goal(GOAL_LINE_X + BALL_RADIUS_M + 1e-6, 0.0, 0.03))

    def test_wide_of_the_posts_is_not(self):
        for y in (GOAL_HALF_WIDTH_Y + 0.01, -GOAL_HALF_WIDTH_Y - 0.01, 0.9, -1.2):
            with self.subTest(y=y):
                self.assertFalse(ball_in_goal(IN, y, BALL_RADIUS_M))

    def test_just_inside_the_posts_is(self):
        for y in (GOAL_HALF_WIDTH_Y - 0.005, -GOAL_HALF_WIDTH_Y + 0.005):
            with self.subTest(y=y):
                self.assertTrue(ball_in_goal(IN, y, BALL_RADIUS_M))

    def test_on_the_post_is_not(self):
        self.assertFalse(ball_in_goal(IN, GOAL_HALF_WIDTH_Y, BALL_RADIUS_M))

    def test_over_the_bar_is_not(self):
        for z in (GOAL_HEIGHT_Z, GOAL_HEIGHT_Z + 0.05, 1.0):
            with self.subTest(z=z):
                self.assertFalse(ball_in_goal(IN, 0.0, z))

    def test_under_the_bar_is(self):
        self.assertTrue(ball_in_goal(IN, 0.0, GOAL_HEIGHT_Z - 0.001))

    def test_behind_the_goal_but_wide_is_not(self):
        # A ball that sails past the whole structure is not a goal, however
        # far behind the line it ends up.
        self.assertFalse(ball_in_goal(GOAL_LINE_X + 1.5, 0.5, BALL_RADIUS_M))

    def test_behind_the_net_is_not(self):
        # The one that fooled a filmed match: rolled wide, wrapped around the
        # OUTSIDE of the net, and stopped behind the goal with |y| drifting
        # under the post line. The scoring volume ends at the back of the
        # net — only the mouth leads inside it.
        self.assertFalse(ball_in_goal(1.115, 0.199, BALL_RADIUS_M))
        self.assertFalse(ball_in_goal(GOAL_NET_BACK_X + 0.001, 0.0, BALL_RADIUS_M))
        self.assertTrue(ball_in_goal(GOAL_NET_BACK_X - 0.05, 0.0, BALL_RADIUS_M))

    def test_radius_is_a_parameter(self):
        x = GOAL_LINE_X + 0.05
        self.assertTrue(ball_in_goal(x, 0.0, 0.03, radius_m=0.01))
        self.assertFalse(ball_in_goal(x, 0.0, 0.03, radius_m=0.10))


class Scoreboard(unittest.TestCase):
    def test_rising_edge_scores_once_and_latches(self):
        r = GoalReferee()
        self.assertTrue(r.update(IN, 0.0, BALL_RADIUS_M, 12.34))
        self.assertEqual(r.count, 1)
        self.assertTrue(r.scored)
        self.assertEqual(r.last_goal_sim_time_s, 12.34)
        # Still sitting in the net: no second goal, and the time stands.
        for t in (12.4, 12.5, 20.0):
            self.assertFalse(r.update(IN, 0.01, BALL_RADIUS_M, t))
        self.assertEqual(r.count, 1)
        self.assertEqual(r.last_goal_sim_time_s, 12.34)

    def test_starts_empty(self):
        r = GoalReferee()
        self.assertEqual(r.state(), {"scored": False, "count": 0,
                                     "last_goal_sim_time_s": None})

    def test_rearms_only_once_the_ball_is_back_out(self):
        r = GoalReferee()
        r.update(IN, 0.0, BALL_RADIUS_M, 1.0)
        # Rolled back over the line but not yet in front of it (e.g. wedged
        # against a post): still latched, still one goal.
        self.assertFalse(r.update(GOAL_LINE_X + 0.005, 0.0, BALL_RADIUS_M, 2.0))
        self.assertTrue(r.scored)
        # Out in front of the line: re-armed, but that is not itself a goal.
        self.assertFalse(r.update(0.2, 0.0, BALL_RADIUS_M, 3.0))
        self.assertFalse(r.scored)
        self.assertEqual(r.count, 1)
        # Scores again on the next shot.
        self.assertTrue(r.update(IN, 0.0, BALL_RADIUS_M, 4.0))
        self.assertEqual(r.count, 2)
        self.assertEqual(r.last_goal_sim_time_s, 4.0)

    def test_over_the_bar_and_out_the_back_rearms_without_scoring(self):
        r = GoalReferee()
        self.assertFalse(r.update(IN, 0.0, GOAL_HEIGHT_Z + 0.1, 1.0))
        self.assertEqual(r.count, 0)
        self.assertFalse(r.scored)

    def test_reset_clears_count_and_latch(self):
        r = GoalReferee()
        r.update(IN, 0.0, BALL_RADIUS_M, 5.0)
        r.reset()
        self.assertEqual(r.state(), {"scored": False, "count": 0,
                                     "last_goal_sim_time_s": None})
        # Re-armed by the reset: a ball already in the net scores afresh.
        self.assertTrue(r.update(IN, 0.0, BALL_RADIUS_M, 0.1))


class _StubPolicy:
    ball_qpos_adr = 0
    sit_mode = False
    current_policy = "standing"
    behavior_mode = None

    def get_projected_gravity(self):
        return np.array([0.0, 0.0, -1.0], dtype=np.float32)


class _StubData:
    def __init__(self, ball_xyz):
        self.qpos = np.array(list(ball_xyz) + [1.0, 0.0, 0.0, 0.0])


class _StubSim:
    """Enough of a DuckSim for referee_tick/_machine_digest, no MuJoCo."""

    def __init__(self, ball_xyz=(0.0, 0.0, BALL_RADIUS_M), referee=None):
        self.policy = _StubPolicy()
        self.data = _StubData(ball_xyz)
        self.referee = referee
        self.sim_time = 7.5
        self.events = deque(maxlen=500)
        self._event_id = 0
        self._ball_seen = {"visible": False, "distance_m": None,
                           "bearing_deg": None, "elevation_deg": None}
        self._ball_seen_t = 0.0
        self._goal_seen = dict(GOAL_NOT_SEEN)
        self._goal_seen_t = 0.0
        self.machine = None

    def _goal_estimates(self):
        return None, None  # goal never sighted


class RefereeTick(unittest.TestCase):
    def test_no_referee_is_a_noop(self):
        sim = _StubSim(ball_xyz=(IN, 0.0, BALL_RADIUS_M))
        DuckSim.referee_tick(sim)
        self.assertEqual(list(sim.events), [])

    def test_goal_logs_one_event_in_the_feed_shape(self):
        sim = _StubSim(ball_xyz=(IN, 0.0, BALL_RADIUS_M), referee=GoalReferee())
        DuckSim.referee_tick(sim)
        DuckSim.referee_tick(sim)  # latched: still one event
        self.assertEqual(len(sim.events), 1)
        ev = sim.events[0]
        self.assertEqual(set(ev), {"id", "t", "client", "cmd", "args", "ok", "note"})
        self.assertEqual((ev["id"], ev["client"], ev["cmd"], ev["ok"]),
                         (1, "referee", "GOAL!", True))
        self.assertEqual(ev["args"]["count"], 1)
        self.assertEqual(sim.referee.last_goal_sim_time_s, 7.5)

    def test_miss_logs_nothing(self):
        sim = _StubSim(ball_xyz=(0.3, 0.0, BALL_RADIUS_M), referee=GoalReferee())
        DuckSim.referee_tick(sim)
        self.assertEqual(list(sim.events), [])
        self.assertEqual(sim.referee.count, 0)


def _model(*geom_names):
    """Tiny MuJoCo model with one named geom each — enough for the lookup."""
    geoms = "".join(f'<geom name="{n}" type="box" size=".1 .1 .1" '
                    f'pos="0 0 {i}"/>' for i, n in enumerate(geom_names))
    return mujoco.MjModel.from_xml_string(f"<mujoco><worldbody>{geoms}"
                                          "</worldbody></mujoco>")


class GoalLookup(unittest.TestCase):
    """The pitch scene is authored in microduck_rl, so the probe has to
    survive a name that is close but not exact."""

    def test_finds_the_expected_names(self):
        for name in GOAL_GEOM_NAMES:
            with self.subTest(name=name):
                self.assertEqual(find_goal_geom(_model("floor", name)), name)

    def test_falls_back_to_any_goal_prefix(self):
        self.assertEqual(find_goal_geom(_model("floor", "goal_left_upright")),
                         "goal_left_upright")

    def test_no_goal_in_a_plain_scene(self):
        self.assertIsNone(find_goal_geom(_model("floor", "ball_geom")))


class Digest(unittest.TestCase):
    def test_goal_keys_present_without_a_goal_scene(self):
        d = DuckSim._machine_digest(_StubSim())
        self.assertIs(d["goal.scored"], False)
        self.assertEqual(d["goal.count"], 0)

    def test_goal_keys_track_the_referee(self):
        r = GoalReferee()
        r.update(IN, 0.0, BALL_RADIUS_M, 1.0)
        d = DuckSim._machine_digest(_StubSim(referee=r))
        self.assertIs(d["goal.scored"], True)
        self.assertEqual(d["goal.count"], 1)

    def test_key_set_is_stable_across_scenes(self):
        with_goal = DuckSim._machine_digest(_StubSim(referee=GoalReferee()))
        without = DuckSim._machine_digest(_StubSim())
        self.assertEqual(set(with_goal), set(without))

    def test_goal_paths_are_guardable(self):
        for p in ("goal.scored", "goal.count"):
            self.assertIn(p, GUARD_PATHS)


if __name__ == "__main__":
    unittest.main()
