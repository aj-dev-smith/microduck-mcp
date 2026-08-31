"""Offline tests for the desktop pet: the odometry paths the machine steers
on, the two behaviors it needed, and machines/pet.toml's own graph.

Nothing here touches MuJoCo. The pet is driven the way every other machine
test drives one — a stub sim, a hand-built digest, and Machine.tick — plus one
closed-loop putter that integrates the velocity commands the behaviors issue,
because "walks to the edge and turns itself around" is a claim about the loop,
not about any single transition.
"""

import math
import os
import random
import types
import unittest

import numpy as np

from microduck_mcp.machine import (
    BEHAVIORS, GUARD_PATHS, Machine, MachineError, _pose_digest, compile_guard)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PET = os.path.join(REPO, "machines", "pet.toml")

DT = 0.02  # the executor's tick, 50 Hz


def gait(vx, wz, jitter=1.0):
    """A coarse fit to what the shipped walk policy actually does with a
    velocity command — the numbers pet.toml is tuned against, measured on a
    live headless daemon (plain scene) rather than assumed:

        vx 0.22 -> 0.096 m/s, 0.26 -> 0.114, 0.30 -> 0.131
        vx <= 0.21, and any negative vx, -> the duck does not move
        wz 1.5 in place -> 0.76-0.92 rad/s;  wz >= 1.0 while walking kills it
        at vx 0.30: wz 0.2/0.3/0.4 -> 5/11/15 deg/s and 0.124/0.118/0.105 m/s

    Commanding 0.22 and integrating 0.22 is what made the first pass of this
    file look fine and the real duck shuffle on the spot, so the model lives
    here where the tuning can be held to it.

    `jitter` scales the turn rate, and it is not decoration: the measured
    180 deg turn ran 3.4-4.1 s over repeated trials, and that spread is the
    entire source of randomness in pet.toml. A noiseless model turns every
    coin flip in that file into a constant — which reads as "the machine
    works" right up until the rate it works at is nine wakes an hour.
    """
    fwd = 0.42 * vx if vx >= 0.22 else 0.0
    fwd *= max(0.0, 1.0 - (wz / 1.2) ** 2)          # yaw eats the walk
    turn = 0.55 * wz * jitter if (fwd > 0.0 or abs(wz) >= 1.0) else 0.0
    return fwd, turn


class _PetPolicy:
    """Remembers the standing velocity intent, the way the real one does."""

    def __init__(self):
        self.cmd = (0.0, 0.0, 0.0)
        self.cmds = []
        self.head_offset = np.zeros(4, dtype=np.float32)
        self.head_max = np.ones(4, dtype=np.float32)

    def set_vel_cmd(self, vx, vy, wz):
        self.cmd = (vx, vy, wz)
        self.cmds.append(self.cmd)

    def _update_command(self):
        pass


class _PetSim:
    """A duck-shaped stub: a poseable freejoint and a trick log.

    qpos is laid out like the real trunk_base_freejoint (x y z qw qx qy qz) so
    _own_pose reads it without knowing this is a test.
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0):
        self.policy = _PetPolicy()
        self.qpos_adr = 0
        self.data = types.SimpleNamespace(qpos=np.zeros(7))
        self.tricks = []
        self.sitting = False
        self.set_pose(x, y, yaw)

    def set_pose(self, x, y, yaw):
        q = self.data.qpos
        q[0], q[1], q[2] = x, y, 0.12
        q[3], q[4], q[5], q[6] = math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)

    def _handle_trick(self, name, stage_ball=True):
        self.tricks.append((name, stage_ball))
        if name == "sit":
            self.sitting = True
        elif name == "stand":
            self.sitting = False
        return {"ok": True, "trick": name, "started": True}


def digest(t, node=None, upright=True, sitting=False, **extra):
    """A sensed digest with the pet's scene in it: no ball, no goal, on its
    feet. `node` matters — the fall reflex names nodes, and a null compares
    False, so a digest without it silently disarms the reflex.
    """
    d = {"sim_time_s": t, "node": node, "upright": upright, "sitting": sitting,
         "active_policy": "walking", "behavior": None,
         "ball_seen.visible": False, "ball_seen.age_s": 99.0,
         "goal.scored": False, "goal.count": 0}
    d.update(extra)
    return d


def tick(m, sim, t, **kw):
    """One executor tick, with `node` filled the way the server fills it."""
    return m.tick(sim, digest(t, node=m.current, **kw))


def run_until(m, sim, target, t0=0.0, limit_s=60.0, **kw):
    """Tick the machine (pose held fixed) until it enters `target`.

    Returns the sim time it landed, or None if it never did.
    """
    t = t0
    while t < t0 + limit_s:
        t = round(t + DT, 3)
        tick(m, sim, t, **kw)
        if m.current == target:
            return t
    return None


class OdometryPaths(unittest.TestCase):
    """base.* — the gap that stopped a machine steering by position.

    It is filled by the executor rather than the server's sensed digest, which
    is what lets pet.toml hot-load onto a daemon that has never heard of a
    Dock. The contract that matters: a sim with no pose to give reports None,
    and every comparison over None is False.
    """

    def test_the_paths_are_in_the_grammar(self):
        for p in ("base.x", "base.y", "base.yaw_deg",
                  "base.heading_x", "base.heading_y"):
            self.assertIn(p, GUARD_PATHS)
            compile_guard(f"{p} > 0.5")

    def test_pose_reads_off_the_freejoint(self):
        d = _pose_digest(_PetSim(x=1.25, y=-0.4, yaw=math.radians(90)))
        self.assertAlmostEqual(d["base.x"], 1.25, places=3)
        self.assertAlmostEqual(d["base.y"], -0.4, places=3)
        self.assertAlmostEqual(d["base.yaw_deg"], 90.0, places=1)

    def test_heading_components_are_the_wrap_free_form(self):
        # yaw_deg wraps at ±180, which is exactly where a duck walking -x
        # lives and the grammar has no abs(). cos/sin do not wrap.
        west = _pose_digest(_PetSim(yaw=math.pi))
        self.assertAlmostEqual(west["base.heading_x"], -1.0, places=3)
        just_over = _pose_digest(_PetSim(yaw=math.pi - 0.01))
        just_under = _pose_digest(_PetSim(yaw=-math.pi + 0.01))
        self.assertLess(abs(just_over["base.yaw_deg"]
                            - just_under["base.yaw_deg"]), 361)
        for d in (just_over, just_under):
            self.assertLess(d["base.heading_x"], -0.99)

    def test_a_sim_with_no_pose_says_so(self):
        nothing = _pose_digest(types.SimpleNamespace(policy=None))
        self.assertEqual(set(nothing.values()), {None})
        # ...and the guard over it is False, not an exception.
        self.assertFalse(compile_guard("base.x > 1.05")(nothing))
        self.assertFalse(compile_guard("base.x < -1.05")(nothing))

    def test_the_executor_fills_them_in_before_the_guards(self):
        m = Machine({"machine": {"name": "t", "initial": "a"},
                     "node": [{"name": "a", "behavior": "idle",
                               "transition": [{"when": "base.x > 1.0",
                                               "to": "b"}]},
                              {"name": "b", "behavior": "idle"}]})
        sim = _PetSim(x=0.0)
        d = digest(1.0, node=m.current)
        self.assertIsNone(m.tick(sim, d))
        self.assertEqual(d["base.x"], 0.0)     # mutated in place, so a wake
        sim.set_pose(1.5, 0.0, 0.0)            # pack snapshots the real pose
        self.assertIsNotNone(tick(m, sim, 2.0))
        self.assertEqual(m.current, "b")


class StrollBehavior(unittest.TestCase):
    """`stroll` — drive with a compass. The turn-around at a wall is this
    behavior being told a different heading, so it has to own the point turn.
    """

    def one(self, sim, params):
        mem = {}
        for i in range(3):
            BEHAVIORS["stroll"](sim, params, mem, digest(i * DT))
        return sim.policy.cmd

    def test_on_heading_it_walks(self):
        vx, vy, wz = self.one(_PetSim(yaw=0.0), {"heading_deg": 0.0, "vx": 0.30})
        self.assertAlmostEqual(vx, 0.30)
        self.assertAlmostEqual(vy, 0.0)
        self.assertAlmostEqual(wz, 0.0, places=6)

    def test_badly_off_it_point_turns_at_full_rate(self):
        # Not proportionally: wz below ~1.2 is a gait dead zone, so a
        # proportional turn from 180 deg off would creep instead of turning.
        vx, _, wz = self.one(_PetSim(yaw=0.0), {"heading_deg": 180.0, "vx": 0.30})
        self.assertEqual(vx, 0.0)
        self.assertAlmostEqual(abs(wz), 1.5)

    def test_slightly_off_it_steers_while_walking(self):
        sim = _PetSim(yaw=math.radians(10))
        vx, _, wz = self.one(sim, {"heading_deg": 0.0, "vx": 0.30})
        self.assertAlmostEqual(vx, 0.30)
        self.assertLess(wz, 0.0)                       # turn back clockwise
        self.assertLessEqual(abs(wz), 0.4)             # ...within wz_max

    def test_the_steering_cap_keeps_it_walking(self):
        # The cap is not a comfort setting: at wz 1.2 the measured walk drops
        # to 0.008 m/s. A steering band that winds past ~0.4 stops the duck.
        sim = _PetSim(yaw=math.radians(24))
        vx, _, wz = self.one(sim, {"heading_deg": 0.0, "vx": 0.30,
                                   "gain": 50.0})
        self.assertAlmostEqual(abs(wz), 0.4)
        self.assertGreater(gait(vx, wz)[0], 0.09)

    def test_without_odometry_it_walks_open_loop(self):
        # A stub with no qpos: fall back to the intent rather than steer on a
        # guess. This is the shape a hardware feed has before telemetry warms.
        sim = types.SimpleNamespace(policy=_PetPolicy())
        BEHAVIORS["stroll"](sim, {"vx": 0.26}, {}, digest(0.0))
        self.assertEqual(sim.policy.cmd, (0.26, 0.0, 0.0))


class TrickBehavior(unittest.TestCase):
    """`trick` — celebrate generalised, so a posture can be a node."""

    def test_it_fires_once_on_entry(self):
        sim, mem = _PetSim(), {}
        for i in range(5):
            BEHAVIORS["trick"](sim, {"trick": "sit"}, mem, digest(i * DT))
        self.assertEqual(sim.tricks, [("sit", False)])

    def test_it_does_not_stage_the_ball(self):
        # A posture change has no business teleporting the world.
        sim = _PetSim()
        BEHAVIORS["trick"](sim, {"trick": "stand"}, {}, digest(0.0))
        self.assertEqual(sim.tricks[0][1], False)


class PetMachineLoads(unittest.TestCase):
    def setUp(self):
        self.m = Machine.load(PET)

    def test_it_loads_and_starts_on_its_feet(self):
        self.assertEqual(self.m.name, "pet")
        self.assertEqual(self.m.initial, "settle")
        self.assertFalse(self.m.armed)

    def test_the_wakes_are_the_three_that_earn_one(self):
        self.assertEqual(self.m.status()["wake_nodes"],
                         ["bored", "fallen", "stuck"])

    def test_words_are_spent_only_on_the_wakes(self):
        # It runs for hours beside somebody working: `say` is for the moments
        # that earned it, emotes and chirps carry everything else.
        self.assertEqual(self.m.status()["say_nodes"],
                         self.m.status()["wake_nodes"])

    def test_every_emote_it_names_exists(self):
        # An unknown emote is only a load warning, so the lint lives here.
        have = {f[:-5] for f in os.listdir(os.path.join(REPO, "emotes"))
                if f.endswith(".toml")}
        for node in self.m.status()["emote_nodes"]:
            self.assertIn(self.m.nodes[node]["emote"], have, msg=node)

    def test_it_never_reaches_for_the_wheee(self):
        # The wheee belongs to a goal actually scored, and a Dock has no goal
        # line — the server would refuse it, so the machine must not ask.
        for name, spec in self.m.nodes.items():
            with self.subTest(node=name):
                self.assertNotIn("wheee", (spec["say"] or "").lower())

    def test_nothing_that_happens_every_minute_makes_a_noise(self):
        # A shipped emote can carry a voice-bank tag, and this machine reaches
        # a wall about seventy times an hour. An emote with a sound on one of
        # those is a noise every forty-nine seconds, all day.
        import tomllib
        sounded = set()
        for f in os.listdir(os.path.join(REPO, "emotes")):
            if not f.endswith(".toml"):
                continue
            with open(os.path.join(REPO, "emotes", f), "rb") as fh:
                if tomllib.load(fh).get("sound"):
                    sounded.add(f[:-5])
        for node in ("wall_left", "wall_right", "pause", "peek",
                     "stroll_left", "stroll_right", "amble_left",
                     "amble_right", "settle"):
            with self.subTest(node=node):
                self.assertNotIn(self.m.nodes[node]["emote"], sounded)

    def test_every_node_is_reachable(self):
        reached = {self.m.initial}
        frontier = [self.m.initial]
        edges = {n: [t["to"] for t in v["transitions"]]
                 for n, v in self.m.nodes.items()}
        globals_ = [t["to"] for t in self.m.global_transitions]
        while frontier:
            for dst in edges[frontier.pop()] + globals_:
                if dst not in reached:
                    reached.add(dst)
                    frontier.append(dst)
        self.assertEqual(reached, set(self.m.nodes))

    def test_no_node_is_a_dead_end(self):
        # A pet that can arrive somewhere it can never leave stops being one.
        for name, spec in self.m.nodes.items():
            with self.subTest(node=name):
                self.assertTrue(spec["transitions"], "no way out")


class _NeverRolls:
    """A `random.Random` stand-in whose coin always lands high — see setUp."""

    def random(self):
        return 0.99


class PetTransitions(unittest.TestCase):
    """The transitions the pet's whole day rests on, fired one at a time."""

    def setUp(self):
        # A rigged coin, not a seed: `start()` pins the roll for the node it
        # enters, but every LATER arrival draws a fresh one, and a real
        # generator made these tests flaky at exactly the rate the file's own
        # dice fire (the 8% nap out of `peek` derailed both turn-around tests
        # about one run in six). Rolls that never come up are the ordinary
        # case; the tests that want a low one still pin it explicitly.
        self.m = Machine.load(PET, rng=_NeverRolls())
        self.m.armed = True

    def start(self, node, sim, roll=0.99):
        """Enter a node with the coin pinned. 0.99 is "the roll did not come
        up", which is the ordinary case for every guard in this file."""
        self.m.enter(node, 0.0)
        self.m.roll = roll
        return sim

    def test_it_sets_off_away_from_the_wall_it_woke_next_to(self):
        for x, expect in ((0.9, "stroll_left"), (-0.9, "stroll_right")):
            with self.subTest(x=x):
                m = Machine.load(PET)
                sim = _PetSim(x=x)
                m.enter("settle", 0.0)
                self.assertEqual(run_until(m, sim, expect, limit_s=5) is None,
                                 False)

    def test_it_turns_round_at_the_right_wall(self):
        # ...by way of the peek, which is where the turn-around actually
        # happens now: the wall is a beat and a look, the spin is the turn.
        sim = self.start("stroll_right", _PetSim(x=1.10, yaw=0.0))
        self.assertIsNotNone(run_until(self.m, sim, "wall_right", limit_s=1))
        self.assertIsNotNone(run_until(self.m, sim, "peek", t0=1.0, limit_s=3))
        self.assertIsNotNone(run_until(self.m, sim, "stroll_left",
                                       t0=3.0, limit_s=9))

    def test_it_turns_round_at_the_left_wall(self):
        sim = self.start("stroll_left", _PetSim(x=-1.10, yaw=math.pi))
        self.assertIsNotNone(run_until(self.m, sim, "wall_left", limit_s=1))
        self.assertIsNotNone(run_until(self.m, sim, "stroll_right",
                                       t0=1.0, limit_s=12))

    def test_a_leg_that_never_leaves_a_wall_is_the_stuck_wake(self):
        # Told to walk left, still standing at the right wall six seconds
        # later. That is a claim about time and distance together, which is
        # the only kind of stuck a position guard can honestly make.
        sim = self.start("stroll_left", _PetSim(x=0.90, yaw=math.pi))
        t = run_until(self.m, sim, "stuck", limit_s=12)
        self.assertIsNotNone(t)
        self.assertGreater(t, 6.0)          # not before the window closes
        self.assertIsNotNone(self.m.nodes["stuck"]["wake"])

    def test_the_slow_pace_gets_the_same_test(self):
        sim = self.start("amble_right", _PetSim(x=-0.90, yaw=0.0))
        self.assertIsNotNone(run_until(self.m, sim, "stuck", limit_s=12))

    def test_merely_passing_the_edge_is_not_stuck(self):
        # The trap this file walked into once: a "near the edge for a while"
        # band fires on a perfectly healthy leg, because a healthy leg is in
        # every band eventually and the timer lands inside one sooner or
        # later. Walking left, deep on the left side, is not a jam.
        sim = self.start("stroll_left", _PetSim(x=-0.90, yaw=math.pi))
        t = 0.0
        while t < 13.5:
            t = round(t + DT, 3)
            tick(self.m, sim, t)
        self.assertNotEqual(self.m.current, "stuck")

    def test_a_healthy_leg_is_not_stuck(self):
        sim = self.start("stroll_right", _PetSim(x=0.0, yaw=0.0))
        t = 0.0
        while t < 13.5:
            t = round(t + DT, 3)
            tick(self.m, sim, t)
        self.assertEqual(self.m.current, "stroll_right")

    def test_a_leg_that_times_out_takes_a_breather(self):
        sim = self.start("stroll_right", _PetSim(x=0.0, yaw=0.0))
        self.assertIsNotNone(run_until(self.m, sim, "pause", limit_s=16))

    def test_the_stuck_wake_backs_itself_out_if_nobody_answers(self):
        sim = self.start("stuck", _PetSim(x=1.02, yaw=0.0))
        self.assertIsNotNone(run_until(self.m, sim, "unjam", limit_s=50))
        # It turns rather than reverses — there is no reverse on this policy.
        tick(self.m, sim, 46.02)
        vx, _, wz = sim.policy.cmd
        self.assertEqual(vx, 0.0)
        self.assertAlmostEqual(abs(wz), 1.5)
        # ...and then a leg AWAY from the wall it was leaning on. Jammed at
        # the right edge, the escape walks left.
        t = run_until(self.m, sim, "escape_left", t0=46.02, limit_s=7)
        self.assertIsNotNone(t)
        # The moment the duck is somewhere else, the ordinary day resumes.
        sim.set_pose(0.4, 0.0, math.pi)
        self.assertIsNotNone(run_until(self.m, sim, "settle", t0=t, limit_s=2))

    def test_a_duck_that_never_moves_parks_silently_instead_of_paging(self):
        # The escalation, and the whole reason it exists: `stuck` is a wake
        # node, and with the unjam returning straight to the rotation a duck
        # that genuinely could not move rode a 61 s loop back into it — 59
        # wakes, 59 spoken complaints and 59 chirps an hour, forever. The
        # escape leg is the test, and `wedged` is where a failed one lands.
        sim = self.start("stuck", _PetSim(x=1.02, yaw=0.0))
        t = run_until(self.m, sim, "wedged", limit_s=80)
        self.assertIsNotNone(t)
        self.assertIsNone(self.m.nodes["wedged"]["wake"], "the park must be silent")
        self.assertIsNone(self.m.nodes["wedged"].get("say"))
        self.assertIsNone(self.m.nodes["wedged"].get("emote"))
        # It keeps trying, quietly — but it does not go back through `stuck`.
        self.assertIsNotNone(run_until(self.m, sim, "unjam", t0=t, limit_s=650))
        # ...and a shove that actually moves it rejoins the day.
        sim.set_pose(0.0, 0.0, 0.0)
        self.assertIsNotNone(run_until(self.m, sim, "settle", t0=t + 650,
                                       limit_s=30))

    def test_a_look_around_that_rolls_low_lies_down(self):
        # The nap coin: an honest draw made on arrival, not a threshold on
        # something the walk decides.
        sim = self.start("peek", _PetSim(x=0.0), roll=0.02)
        self.assertIsNotNone(run_until(self.m, sim, "settle_down", limit_s=8))

    def test_the_same_look_around_on_a_high_roll_just_walks_on(self):
        sim = self.start("peek", _PetSim(x=0.0), roll=0.5)
        self.assertIsNotNone(run_until(self.m, sim, "stroll_right", limit_s=8))

    def test_a_pause_carries_on_the_way_it_was_pointing(self):
        for yaw, expect in ((0.0, "amble_right"), (math.pi, "amble_left")):
            with self.subTest(yaw=yaw):
                m = Machine.load(PET)
                m.enter("pause", 0.0)
                m.roll = 0.99          # ...and did not roll a look-around
                sim = _PetSim(x=-0.20, yaw=yaw)
                self.assertIsNotNone(run_until(m, sim, expect, limit_s=4))

    def test_a_pause_near_a_wall_ambles_away_from_it(self):
        for x, expect in ((1.02, "amble_left"), (-1.02, "amble_right")):
            with self.subTest(x=x):
                m = Machine.load(PET)
                m.enter("pause", 0.0)
                m.roll = 0.99
                sim = _PetSim(x=x, yaw=0.0 if x > 0 else math.pi)
                self.assertIsNotNone(run_until(m, sim, expect, limit_s=4))

    def test_the_peek_picks_its_direction_off_the_heading_it_stopped_on(self):
        # Where the pet gets its randomness: a spin in place ends on a heading
        # nothing in the file chose.
        for yaw, expect in ((0.3, "stroll_right"), (math.pi - 0.3, "stroll_left")):
            with self.subTest(yaw=yaw):
                m = Machine.load(PET)
                m.enter("peek", 0.0)
                m.roll = 0.99
                sim = _PetSim(x=0.0, yaw=yaw)
                self.assertIsNotNone(run_until(m, sim, expect, limit_s=8))

    def test_a_look_around_at_a_wall_never_picks_the_wall(self):
        # Without the room check, a spin at the left wall can pick "left",
        # take half a step, re-enter the wall node, and ring-a-ring-o'-roses.
        for x, expect in ((1.02, "stroll_left"), (-1.02, "stroll_right")):
            with self.subTest(x=x):
                m = Machine.load(PET)
                m.enter("peek", 0.0)
                m.roll = 0.99
                sim = _PetSim(x=x, yaw=0.0 if x > 0 else math.pi)
                self.assertIsNotNone(run_until(m, sim, expect, limit_s=8))

    def test_the_nap_waits_for_the_posture_then_dozes(self):
        sim = self.start("settle_down", _PetSim(x=0.7))
        t, sat = 0.0, False
        while t < 2.0 and self.m.current != "doze":
            t = round(t + DT, 3)
            tick(self.m, sim, t, sitting=sim.sitting)
            sat = sat or sim.sitting
        self.assertTrue(sat, "never asked to sit")
        self.assertEqual(sim.tricks, [("sit", False)])
        self.assertEqual(self.m.current, "doze")

    def test_a_refused_sit_still_dozes(self):
        # A stub that says no to everything: the deadline transition is why
        # the machine can wait on `sitting` at all.
        sim = self.start("settle_down", _PetSim(x=0.7))
        sim._handle_trick = lambda name, stage_ball=True: {"ok": False}
        self.assertIsNotNone(run_until(self.m, sim, "doze", limit_s=4))

    def test_a_nap_that_rolls_low_deepens_into_the_boredom_wake(self):
        sim = self.start("doze", _PetSim(x=0.7), roll=0.02)
        self.assertIsNotNone(run_until(self.m, sim, "deep_doze", limit_s=100,
                                       sitting=True))
        self.assertIsNotNone(run_until(self.m, sim, "bored", t0=100,
                                       limit_s=260, sitting=True))
        self.assertIn("bored", self.m.nodes["bored"]["wake"])

    def test_an_ordinary_nap_just_gets_up(self):
        # The second coin, drawn on arrival here and so independent of the one
        # that chose to nap at all. It is what keeps a wake rare enough to run
        # all day: measured at 0.97 an hour over 32 simulated hours.
        sim = self.start("doze", _PetSim(x=0.7), roll=0.5)
        self.assertIsNotNone(run_until(self.m, sim, "get_up", limit_s=100,
                                       sitting=True))

    def test_the_boredom_wake_has_a_deadline_and_takes_it(self):
        sim = self.start("bored", _PetSim(x=0.7))
        self.assertIsNotNone(run_until(self.m, sim, "get_up", limit_s=190,
                                       sitting=True))

    def test_getting_up_waits_for_the_posture_to_go(self):
        sim = self.start("get_up", _PetSim(x=0.7))
        sim.sitting = True
        t = 0.0
        while t < 3.0 and self.m.current != "settle":
            t = round(t + DT, 3)
            tick(self.m, sim, t, sitting=sim.sitting)
        self.assertEqual(sim.tricks, [("stand", False)])
        self.assertEqual(self.m.current, "settle")


class PetFallReflex(unittest.TestCase):
    """The global escape hatch, resident.toml's, with the postures carved out."""

    def setUp(self):
        self.m = Machine.load(PET)
        self.m.armed = True

    def test_a_fall_mid_stroll_wakes_from_anywhere(self):
        self.m.enter("stroll_right", 0.0)
        sim = _PetSim(x=0.0)
        self.assertIsNotNone(run_until(self.m, sim, "fallen", limit_s=2,
                                       upright=False))
        self.assertIn("duck_reset", self.m.nodes["fallen"]["wake"])

    def test_the_wake_cannot_restorm(self):
        self.m.enter("fallen", 0.0)
        sim = _PetSim()
        for i in range(200):
            fired = tick(self.m, sim, i * DT, upright=False)
            self.assertIsNone(fired)
        self.assertEqual(self.m.current, "fallen")

    def test_a_napping_duck_is_not_a_fallen_one(self):
        # The sit pose puts the trunk somewhere the upright test was never
        # asked about, so `not sitting` guards the reflex.
        self.m.enter("doze", 0.0)
        sim = _PetSim(x=0.7)
        for i in range(200):
            tick(self.m, sim, i * DT, upright=False, sitting=True)
        self.assertEqual(self.m.current, "doze")

    def test_the_sitstand_transient_is_carved_out(self):
        for node in ("settle_down", "get_up"):
            with self.subTest(node=node):
                m = Machine.load(PET)
                m.enter(node, 0.0)
                sim = _PetSim(x=0.7)
                for i in range(60):
                    tick(m, sim, i * DT, upright=False, sitting=False)
                self.assertNotEqual(m.current, "fallen")

    def test_it_notices_its_own_recovery(self):
        # `duck reset` stands the duck up and re-enters the current node
        # (reset is deliberately not a wake), so watching `upright` is how the
        # pet gets back to work without anybody forcing a node.
        self.m.enter("fallen", 0.0)
        sim = _PetSim()
        self.assertIsNotNone(run_until(self.m, sim, "settle", limit_s=3))

    def test_the_park_is_reached_but_is_not_terminal(self):
        self.m.enter("fallen", 0.0)
        sim = _PetSim()
        self.assertIsNotNone(run_until(self.m, sim, "down", limit_s=310,
                                       upright=False))
        self.assertIsNone(self.m.nodes["down"]["wake"])
        self.assertIsNotNone(run_until(self.m, sim, "settle", t0=310,
                                       limit_s=5))


class PetPutters(unittest.TestCase):
    """The closed loop: integrate what the behaviors actually command.

    Kinematics only (the real thing is a walk policy over MuJoCo), but it is
    the same commands through the same guards, so it answers the questions no
    single transition can — does it stay on screen, does it turn itself
    around, and does it keep going.
    """

    def putter(self, seconds, x0=0.0, yaw0=0.0, wall=1.30, wedged=False,
               seed=7):
        m = Machine.load(PET, rng=random.Random(seed))
        m.armed = True
        sim = _PetSim(x=x0, yaw=yaw0)
        rng = random.Random(seed)
        x, y, yaw = x0, 0.0, yaw0
        visited, xs, entered = {m.current}, [], []
        t = 0.0
        jitter, node = 1.0, m.current
        while t < seconds:
            t = round(t + DT, 3)
            tick(m, sim, t, sitting=sim.sitting)
            if m.current != node:              # a fresh draw per leg, the way
                node = m.current               # the gait is fresh per attempt
                jitter = rng.uniform(0.87, 1.13)
                entered.append(node)
            visited.add(m.current)
            vx, _, wz = sim.policy.cmd
            fwd, turn = gait(vx, wz, jitter)
            yaw += turn * DT
            x += fwd * math.cos(yaw) * DT
            y += fwd * math.sin(yaw) * DT
            x = max(-wall, min(wall, x))       # the scene's invisible walls
            if wedged:                         # ...or something it cannot pass
                x, y = x0, 0.0
            sim.set_pose(x, y, yaw)
            xs.append(x)
        return m, visited, xs, entered

    def test_it_paces_the_screen_and_never_walks_off_it(self):
        _, visited, xs, _ = self.putter(400.0)
        self.assertIn("wall_right", visited)
        self.assertIn("wall_left", visited)
        # Turn guards at ±1.05, walls at ±1.30: the duck must stop itself well
        # inside them, or the overlay window hangs off the edge of the screen.
        self.assertLess(max(xs), 1.25, "walked past the right wall")
        self.assertGreater(min(xs), -1.25, "walked past the left wall")
        # ...and it must actually use the screen, not shuffle in the middle.
        self.assertGreater(max(xs) - min(xs), 1.5)

    def test_it_varies_its_pace_and_takes_breaks(self):
        _, visited, _, _ = self.putter(600.0)
        for node in ("stroll_right", "stroll_left", "pause", "peek"):
            self.assertIn(node, visited)
        self.assertTrue({"amble_left", "amble_right"} & visited)

    def test_it_never_wedges_or_falls_on_its_own(self):
        # Left alone on flat ground the pet has no business waking anybody.
        _, visited, _, _ = self.putter(600.0)
        self.assertNotIn("stuck", visited)
        self.assertNotIn("fallen", visited)

    def test_a_duck_that_cannot_move_ends_up_asking_for_help(self):
        # Wedged near the right edge: it can still turn, and the walk policy
        # is still being asked for velocity, but the world does not move. The
        # away-from-the-wall guard is what notices, on the first leg that is
        # supposed to cross the screen — and `peek`, spinning onto a heading
        # nothing chose, is what guarantees such a leg gets picked. That is
        # why the peek is load-bearing and not decoration.
        _, visited, _, _ = self.putter(400.0, x0=0.95, wedged=True)
        self.assertIn("stuck", visited)
        self.assertIn("unjam", visited)      # ...and it tries to turn out of it

    def test_it_naps_and_gets_up_again(self):
        m, visited, _, _ = self.putter(1800.0)
        for node in ("settle_down", "doze", "get_up"):
            self.assertIn(node, visited)
        self.assertTrue(m.current in m.nodes)

    def test_it_does_not_wake_a_mind_more_than_about_once_an_hour(self):
        # The budget for a thing that sits beside somebody working all day.
        # An earlier cut of this file managed NINE boredom wakes an hour and a
        # noiseless putter reported it as working perfectly, which is why the
        # rate is asserted here and the coin behind it is a real `roll`.
        wakes = 0
        for seed in (3, 7, 11, 19):
            _, _, _, entered = self.putter(3600.0, seed=seed)
            wakes += sum(1 for n in entered
                         if n in ("bored", "stuck", "fallen"))
        self.assertLessEqual(wakes, 6, f"{wakes} wakes in 4 simulated hours")

    def test_it_still_naps_where_a_person_would_see_it(self):
        # The other side of the budget: rationing the wake must not ration the
        # nap, which is the pet's one visible change of posture.
        naps = 0
        for seed in (3, 7, 11, 19):
            _, _, _, entered = self.putter(3600.0, seed=seed)
            naps += entered.count("settle_down")
        self.assertGreaterEqual(naps, 4, f"{naps} naps in 4 simulated hours")

    def test_it_is_still_going_after_a_long_stretch(self):
        m, visited, xs, _ = self.putter(1800.0)
        self.assertTrue(m.current in m.nodes)
        self.assertGreater(max(xs) - min(xs), 1.5)


class PetHotLoads(unittest.TestCase):
    """The pet must not have broken the machines that were here first."""

    def test_the_older_machines_still_load(self):
        for name in ("resident.toml", "striker.toml", "soccer.toml"):
            with self.subTest(machine=name):
                Machine.load(os.path.join(REPO, "machines", name))

    def test_the_roll_is_drawn_once_per_arrival_not_once_per_tick(self):
        # A per-tick draw would make `roll < 0.15` fire the first tick it felt
        # like rather than at its deadline, which is not what the guards read.
        m = Machine({"machine": {"name": "t", "initial": "a"},
                     "node": [{"name": "a", "behavior": "idle"}]},
                    rng=random.Random(4))
        sim = _PetSim()
        first = m.roll
        for i in range(50):
            tick(m, sim, i * DT)
        self.assertEqual(m.roll, first)
        m.enter("a", 1.0)
        self.assertNotEqual(m.roll, first)

    def test_the_roll_is_seedable(self):
        a = Machine.load(PET, rng=random.Random(5))
        b = Machine.load(PET, rng=random.Random(5))
        self.assertEqual(a.roll, b.roll)
        self.assertIn("roll", GUARD_PATHS)


if __name__ == "__main__":
    unittest.main()
