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
    feet, and nobody in the room. `node` matters — the fall reflex names
    nodes, and a null compares False, so a digest without it silently disarms
    the reflex.

    The `cursor.*` / `touch.*` / `carried` block is the server's half
    (sim_server._machine_digest), and its defaults are the ones a daemon with
    no overlay attached reports: no pointer, never petted, not held. Spelled
    out rather than left missing, because "the key is absent" and "the hand is
    not there" are the same answer only by accident — a null compares False
    and so does a stale 999.0, and a test that relied on the accident would
    stop meaning anything the day the digest grew.
    """
    d = {"sim_time_s": t, "node": node, "upright": upright, "sitting": sitting,
         "active_policy": "walking", "behavior": None,
         # The toy, as a daemon with the ball out of sight reports it. The
         # pocket and speed keys are spelled out for the same reason the
         # cursor's are: "the key is absent" and "the ball is not there" are
         # the same answer only by accident, and the play loop's guards are
         # written against the difference.
         "ball_seen.visible": False, "ball_seen.age_s": 99.0,
         "ball_seen.est_forward_m": None, "ball_seen.est_left_m": None,
         "ball_seen.speed_mps": None, "ball_seen.distance_m": None,
         "goal.scored": False, "goal.count": 0,
         "cursor.present": False, "cursor.x_m": None, "cursor.z_m": None,
         "cursor.dx_m": None, "cursor.dist_m": None, "cursor.age_s": 999.0,
         "cursor.near_floor": False, "cursor.speed_mps": None,
         "touch.petted": False, "touch.age_s": 999.0, "touch.count": 0,
         "carried": False, "carried_s": 0.0}
    d.update(extra)
    return d


def hand(x_m=0.30, z_m=0.10, age_s=0.0, speed_mps=0.15):
    """Digest extras for a pointer `x_m` to the duck's right, in motion.

    Written as a helper because every cursor test needs the same six keys to
    agree with each other — `dist_m` is `abs(dx_m)` in the daemon, and a test
    that let them disagree would be testing a machine that cannot exist.

    Moving by default, because a still pointer is deliberately invisible to
    the notice gate (see `test_a_pointer_left_lying_about_is_furniture`) and a
    helper whose default was inert would make every test below pass for the
    wrong reason.
    """
    return {"cursor.present": True, "cursor.x_m": x_m, "cursor.z_m": z_m,
            "cursor.dx_m": x_m, "cursor.dist_m": abs(x_m),
            "cursor.age_s": age_s, "cursor.near_floor": z_m <= 0.35,
            "cursor.speed_mps": speed_mps}


def toy(speed_mps=0.4, age_s=0.0, forward_m=0.40, left_m=-0.05):
    """Digest extras for a ball in view, `speed_mps` of it rolling.

    MOVING by default, for the exact reason `hand()` is: a ball lying still
    is deliberately invisible to the play gate (pet.toml's PLAY_MPS — a toy
    nobody touched is furniture), so a helper whose default was inert would
    make every test below pass for the wrong reason.

    `forward_m`/`left_m` are the trunk-frame offsets the kick pocket is
    measured in; the defaults are a ball comfortably OUTSIDE it, so a test
    that wants the boot has to ask for it.
    """
    return {"ball_seen.visible": True, "ball_seen.age_s": age_s,
            "ball_seen.speed_mps": speed_mps,
            "ball_seen.est_forward_m": forward_m,
            "ball_seen.est_left_m": left_m,
            "ball_seen.distance_m": math.hypot(forward_m, left_m)}


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
                     "amble_right", "settle",
                     # A pointer crosses the Dock dozens of times an hour, so
                     # `notice` and the approach it leads to are on the same
                     # budget as a wall. `petted` is silent for a different
                     # reason: the DAEMON already answered the hand with a
                     # nuzzle and a coo, and a gesture here would be the duck
                     # thanking you twice for one stroke.
                     "notice", "approach_cursor_left", "approach_cursor_right",
                     "lost_interest", "petted",
                     # `regard` is on this list for the same reason and for a
                     # measurement of its own: it used to carry head_tilt, and
                     # the putter below counted 48-73 of them an hour against
                     # a hand that keeps moving. The inquire lives on
                     # `greet_hand` now, behind a 12% coin.
                     "regard",
                     # The whole play lane, and it is silent for a measured
                     # reason: a duck pacing a 2 m strip walks into its own
                     # toy on most laps, so even with the range and speed
                     # gates a real daemon opened six bouts in ten minutes.
                     # The duck expresses this by walking over and kicking
                     # something, which at 180 px is louder than a chirp.
                     "see_ball", "chase_ball", "find_ball", "boot",
                     "watch_ball", "ball_rest"):
            with self.subTest(node=node):
                self.assertNotIn(self.m.nodes[node]["emote"], sounded)

    # The play loop, currently sealed: its one door in (the global see_ball
    # transition) is commented out in pet.toml — chasing hogged the duck's
    # whole first hands-on test. The nodes stay so the park is one uncomment;
    # this is the list reachability excuses until then.
    PARKED_BALL_NODES = frozenset(
        {"see_ball", "chase_ball", "find_ball", "boot",
         "watch_ball", "ball_rest"})

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
        self.assertEqual(reached | self.PARKED_BALL_NODES, set(self.m.nodes))
        # ...and the park is real: none of the sealed nodes leaked back in.
        self.assertFalse(reached & self.PARKED_BALL_NODES)

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


class _AlwaysRolls:
    """A rigged coin that always lands wherever the test says."""

    def __init__(self, value):
        self.value = value

    def random(self):
        return self.value


class PetNoticesTheHuman(unittest.TestCase):
    """The cursor lane: see the hand, walk to it, look at it, give up.

    The whole loop is ordinary `stroll` legs told a compass heading — "walk
    toward the hand" IS a heading — so what is tested here is the graph, not a
    controller. What a *hand* is, and how a stroke is told from a shove, lives
    on the app's side of the socket (tests/test_pet_app.py).
    """

    def setUp(self):
        self.m = Machine.load(PET, rng=_NeverRolls())     # the high half
        self.m.armed = True

    def enter(self, node, sim, roll=0.99):
        self.m.enter(node, 0.0)
        self.m.roll = roll
        return sim

    def test_the_cursor_paths_are_in_the_grammar(self):
        for p in ("cursor.present", "cursor.x_m", "cursor.z_m", "cursor.dx_m",
                  "cursor.dist_m", "cursor.age_s", "cursor.near_floor",
                  "cursor.speed_mps", "touch.petted", "touch.age_s",
                  "touch.count", "carried", "carried_s"):
            with self.subTest(path=p):
                self.assertIn(p, GUARD_PATHS)
                compile_guard(f"{p} != 0")

    def test_a_hand_near_the_dock_pulls_the_duck_off_its_lap(self):
        sim = self.enter("stroll_right", _PetSim(x=0.0, yaw=0.0))
        t = run_until(self.m, sim, "notice", limit_s=2, **hand(0.30))
        self.assertIsNotNone(t, "a hand beside the duck went unnoticed")
        # ...and it walks the way the hand is, not the way it was already going
        t = run_until(self.m, sim, "approach_cursor_right", t0=t, limit_s=3,
                      **hand(0.30))
        self.assertIsNotNone(t)
        # The hand held still and the duck arrived: personal space, and it
        # stands there and looks. Silently — the rig's coin lands high, and
        # the inquire is behind the low 12% (`greet_hand`, below).
        self.assertIsNotNone(run_until(self.m, sim, "regard", t0=t, limit_s=3,
                                       **hand(0.15)))
        self.assertIsNone(self.m.nodes["regard"]["emote"])

    def test_about_one_arrival_in_eight_says_hello_out_loud(self):
        # The ration. `regard` used to carry head_tilt outright, justified as
        # "reaching personal space rations the sound by itself" — measured on
        # this file's own putter, that was 48-73 inquires an HOUR. The coin is
        # the fix: a low roll greets, a high one just looks, and the greeting
        # node has one way out (into `regard`) so one approach cannot chirp
        # twice.
        m = Machine.load(PET, rng=_AlwaysRolls(0.05))
        m.armed = True
        m.enter("approach_cursor_right", 0.0)
        sim = _PetSim(x=0.0, yaw=0.0)
        t = run_until(m, sim, "greet_hand", limit_s=3, **hand(0.15))
        self.assertIsNotNone(t, "the low half of the coin never greeted")
        self.assertEqual(m.nodes["greet_hand"]["emote"], "head_tilt")
        # ...and it hands straight over to the silent node it split off from.
        self.assertIsNotNone(run_until(m, sim, "regard", t0=t, limit_s=3,
                                       **hand(0.15)))
        self.assertEqual([tr["to"] for tr in m.nodes["regard"]["transitions"]
                          ].count("greet_hand"), 0,
                         "regard can hand the chirp back to itself")

    def test_a_hand_on_the_other_side_is_walked_towards_the_other_way(self):
        sim = self.enter("amble_right", _PetSim(x=0.0, yaw=0.0))
        t = run_until(self.m, sim, "notice", limit_s=2, **hand(-0.30))
        self.assertIsNotNone(t)
        self.assertIsNotNone(run_until(self.m, sim, "approach_cursor_left",
                                       t0=t, limit_s=3, **hand(-0.30)))

    def test_the_duck_ignores_the_cursor_on_the_low_half_of_the_coin(self):
        # `peek` spends roll < 0.08 on the nap and `pause` roll < 0.30 on the
        # extra look-around. A cursor gate in that band, placed above them in
        # file order, would quietly eat both — so the gate is the HIGH half,
        # and a low roll means the duck simply carries on with its leg.
        m = Machine.load(PET, rng=_AlwaysRolls(0.10))
        m.armed = True
        m.enter("stroll_right", 0.0)
        sim = _PetSim(x=0.0, yaw=0.0)
        t = 0.0
        while t < 8.0:
            t = round(t + DT, 3)
            tick(m, sim, t, **hand(0.30))
        self.assertEqual(m.current, "stroll_right")

    def test_a_pointer_left_lying_about_is_furniture(self):
        # The term the spec for this lane did not have, and the measurement
        # that put it there. Without `cursor.speed_mps > 0.05` the gate reads
        # "a pointer is near me", which stays true for as long as somebody
        # leaves the mouse over the Dock and goes back to reading — and the
        # loop that follows (notice -> approach -> regard -> lost_interest ->
        # settle -> a leg -> notice) closes in about sixteen seconds. Measured
        # on the putter below with the cursor pinned at three different x:
        # 54-60 `regard` entries an hour, i.e. an inquiring chirp a minute,
        # all day, from a mouse nobody was touching. With the term: zero.
        sim = self.enter("stroll_right", _PetSim(x=0.0, yaw=0.0))
        t = 0.0
        while t < 10.0:
            t = round(t + DT, 3)
            tick(self.m, sim, t, **hand(0.30, speed_mps=0.0))
        self.assertEqual(self.m.current, "stroll_right")
        # ...and the same hand, moving, is noticed at once.
        self.assertIsNotNone(run_until(self.m, sim, "notice", t0=t, limit_s=2,
                                       **hand(0.30)))

    def test_a_brand_new_pointer_has_no_speed_yet_and_that_is_not_a_crash(self):
        # `cursor.speed_mps` is None until two samples exist (0.2 s at 5 Hz).
        # A comparison over a null is False in the grammar, so the duck simply
        # waits one more sample rather than the guard raising.
        sim = self.enter("stroll_right", _PetSim(x=0.0, yaw=0.0))
        t = 0.0
        while t < 3.0:
            t = round(t + DT, 3)
            tick(self.m, sim, t, **hand(0.30, speed_mps=None))
        self.assertEqual(self.m.current, "stroll_right")

    def test_a_pointer_up_in_an_editor_is_not_a_visit(self):
        # `near_floor` is the other half of the gate: a cursor 0.6 m up the
        # screen is somebody working, not somebody visiting.
        sim = self.enter("stroll_right", _PetSim(x=0.0, yaw=0.0))
        t = 0.0
        while t < 8.0:
            t = round(t + DT, 3)
            tick(self.m, sim, t, **hand(0.30, z_m=0.60))
        self.assertEqual(self.m.current, "stroll_right")

    def test_a_pointer_already_on_the_duck_is_a_click_coming_not_an_invitation(self):
        sim = self.enter("stroll_right", _PetSim(x=0.0, yaw=0.0))
        t = 0.0
        while t < 8.0:
            t = round(t + DT, 3)
            tick(self.m, sim, t, **hand(0.04))
        self.assertEqual(self.m.current, "stroll_right")

    def test_a_hand_that_left_before_the_duck_got_going_is_let_go(self):
        # Most pointers that come near are on their way somewhere else. The
        # beat before `notice` commits is what stops a wasted leg after each.
        sim = self.enter("notice", _PetSim(x=0.0, yaw=0.0))
        self.assertIsNotNone(run_until(self.m, sim, "settle", limit_s=3,
                                       **hand(0.30, age_s=2.0)))

    def test_a_hand_that_leaves_mid_approach_is_given_up_on(self):
        sim = self.enter("approach_cursor_right", _PetSim(x=0.0, yaw=0.0))
        t = run_until(self.m, sim, "lost_interest", limit_s=2,
                      **hand(0.30, age_s=3.0))
        self.assertIsNotNone(t)
        self.assertIsNone(self.m.nodes["lost_interest"]["wake"])
        self.assertIsNone(self.m.nodes["lost_interest"]["say"])
        # ...and then straight back to work. The duck was not waiting on you.
        self.assertIsNotNone(run_until(self.m, sim, "settle", t0=t, limit_s=3))

    def test_a_hand_that_goes_past_behind_turns_the_duck_round(self):
        # The 0.15 m band is hysteresis, not slop: a sign test alone flaps at
        # arm's length and the duck alternates left-right-left forever. Note
        # what the band actually costs, because it is the interesting half —
        # a pointer within 0.15 m BEHIND the duck is also within personal
        # space, so `regard` (listed first) claims it and the turn never
        # happens. Turning round is reserved for a hand that genuinely went
        # somewhere else, which is the only case worth two point turns.
        sim = self.enter("approach_cursor_right", _PetSim(x=0.0, yaw=0.0))
        self.assertIsNotNone(run_until(self.m, sim, "regard", limit_s=1,
                                       **hand(-0.10)))
        sim = self.enter("approach_cursor_right", _PetSim(x=0.0, yaw=0.0))
        self.assertIsNotNone(run_until(self.m, sim, "approach_cursor_left",
                                       limit_s=2, **hand(-0.30)))

    def test_the_wall_still_outranks_the_human(self):
        # A pointer parked off the edge of the screen must not walk the duck
        # into the bezel. The wall guard is the first one in the node.
        sim = self.enter("approach_cursor_right", _PetSim(x=1.10, yaw=0.0))
        self.assertIsNotNone(run_until(self.m, sim, "wall_right", limit_s=2,
                                       **hand(0.30)))

    def test_a_parked_cursor_is_regarded_and_then_let_be(self):
        sim = self.enter("regard", _PetSim(x=0.0, yaw=0.0))
        self.assertIsNotNone(run_until(self.m, sim, "lost_interest",
                                       limit_s=14, **hand(0.10)))

    def test_a_hand_that_wanders_off_is_followed_again(self):
        sim = self.enter("regard", _PetSim(x=0.0, yaw=0.0))
        self.assertIsNotNone(run_until(self.m, sim, "approach_cursor_right",
                                       limit_s=2, **hand(0.70)))


class PetIsPetted(unittest.TestCase):
    """The gesture that is not a shove — from the machine's side of it.

    The DAEMON answers the hand (a nuzzle and a coo on POST /pet/touch, one
    every 2.5 s). All the machine contributes is the one thing a daemon
    cannot: it stops walking.
    """

    def setUp(self):
        self.m = Machine.load(PET, rng=_NeverRolls())
        self.m.armed = True

    def test_a_petted_duck_stops_walking_and_then_gets_back_to_work(self):
        self.m.enter("stroll_right", 0.0)
        sim = _PetSim(x=0.0, yaw=0.0)
        t = run_until(self.m, sim, "petted", limit_s=2,
                      **{"touch.petted": True, "touch.age_s": 0.0})
        self.assertIsNotNone(t)
        self.assertEqual(sim.policy.cmd, (0.0, 0.0, 0.0), "it kept walking")
        # ...and answers the hand ITSELF: the daemon's touch-time nuzzle is
        # refused whenever the head belongs to a steering node — a duck
        # mid-stroll, i.e. nearly every duck anyone actually strokes; on the
        # first hands-on test every pet landed in the tally and looked like
        # nothing. Machine-priority, so it plays over a walk; if the daemon's
        # own nuzzle is already running (the duck was idle when touched), the
        # engine's mid-emote refusal drops this one — one nuzzle either way.
        self.assertEqual(self.m.nodes["petted"]["emote"], "nuzzle")
        self.assertIsNotNone(run_until(self.m, sim, "settle", t0=t, limit_s=4))

    def test_one_stroke_is_one_visit_to_petted_and_not_a_flap(self):
        # `touch.petted` is `age_s < PET_TOUCH_RECENT_S` (sim_server: 3.0 s)
        # and the global petting transition excludes only `petted` itself — so
        # an exit at 2.5 s handed the body back while the reflex was still
        # armed and the global fired again at once. Measured on the real
        # machine with one stroke at t=1.0: petted(1.00) -> regard(3.52) ->
        # petted(3.54) -> regard(6.06), i.e. the head_tilt `regard` used to
        # carry fired TWICE for one stroke — the exact double thank-you the
        # `petted` node's own comment says it exists to prevent.
        self.m.enter("stroll_right", 0.0)
        sim = _PetSim(x=0.0, yaw=0.0)
        entries, node = 0, self.m.current
        t = 0.0
        while t < 8.0:
            t = round(t + DT, 3)
            # One stroke at t = 1.0, and then a hand that stays beside the
            # duck without touching it again — which is what a person does.
            age = 999.0 if t < 1.0 else t - 1.0
            tick(self.m, sim, t, **{"touch.petted": age < 3.0,
                                    "touch.age_s": age, **hand(0.15)})
            if self.m.current != node:
                node = self.m.current
                if node == "petted":
                    entries += 1
        self.assertEqual(entries, 1, "one stroke, more than one `petted`")

    def test_a_petted_duck_that_still_has_a_hand_beside_it_looks_at_it(self):
        self.m.enter("petted", 0.0)
        sim = _PetSim(x=0.0, yaw=0.0)
        self.assertIsNotNone(run_until(self.m, sim, "regard", limit_s=4,
                                       **hand(0.20)))

    def test_a_petted_nap_gets_up_properly_instead_of_strolling_seated(self):
        # `settle` strolls, and `bhv_stroll` on a seated duck commands a
        # velocity the gait cannot deliver — measured, that is a duck
        # shuffling on its haunches until some later deadline notices.
        self.m.enter("doze", 0.0)
        sim = _PetSim(x=0.0, yaw=0.0)
        sim.sitting = True
        t = run_until(self.m, sim, "petted", limit_s=2, sitting=True,
                      **{"touch.petted": True, "touch.age_s": 0.0})
        self.assertIsNotNone(t, "a nap cannot be interrupted by a hand")
        t = run_until(self.m, sim, "get_up", t0=t, limit_s=4, sitting=True)
        self.assertIsNotNone(t)
        tick(self.m, sim, t + DT, sitting=True)   # the entry tick runs the node
        self.assertEqual(sim.tricks[-1], ("stand", False))

    def test_a_fallen_duck_is_not_talked_out_of_being_fallen(self):
        # `fallen` has already latched a wake and asked for help; a stroke is
        # not an answer to it, and a detour would drop the wake on the floor.
        self.m.enter("fallen", 0.0)
        sim = _PetSim(x=0.0)
        for i in range(200):
            tick(self.m, sim, i * DT, upright=False,
                 **{"touch.petted": True, "touch.age_s": 0.0})
        self.assertEqual(self.m.current, "fallen")

    def test_the_sitstand_transient_is_carved_out_of_the_petting_reflex_too(self):
        for node in ("settle_down", "get_up"):
            with self.subTest(node=node):
                m = Machine.load(PET, rng=_NeverRolls())
                m.armed = True
                m.enter(node, 0.0)
                sim = _PetSim(x=0.0)
                for i in range(40):
                    tick(m, sim, i * DT,
                         **{"touch.petted": True, "touch.age_s": 0.0})
                self.assertNotEqual(m.current, "petted")

    def test_being_held_outranks_being_petted(self):
        # Global order is priority, and `carried` is written first: a hand
        # stroking a duck that is dangling from the other hand is a duck being
        # HELD, and `petted` (which is a node about standing still on the
        # floor) is the wrong answer to it.
        self.m.enter("stroll_right", 0.0)
        sim = _PetSim(x=0.0)
        for i in range(200):
            tick(self.m, sim, i * DT, carried=True,
                 **{"touch.petted": True, "touch.age_s": 0.0})
        self.assertEqual(self.m.current, "carried")


class PetIsPickedUp(unittest.TestCase):
    """The pick-up: the one gesture that is a constraint and not a force.

    The machine's whole contribution is `carried` — stop, be surprised, and
    hand back to `settle` when the hand opens — plus two carve-outs on the
    fall reflex that between them are the difference between a feature and a
    bug. Everything else about the lift is the weld and the standing policy.
    """

    def setUp(self):
        self.m = Machine.load(PET, rng=_NeverRolls())
        self.m.armed = True

    def test_the_carry_paths_are_in_the_grammar(self):
        for p in ("carried", "carried_s"):
            self.assertIn(p, GUARD_PATHS)
        compile_guard("carried and carried_s > 1.0")

    def test_a_lift_interrupts_whatever_the_duck_was_doing(self):
        self.m.enter("stroll_right", 0.0)
        sim = _PetSim(x=0.0)
        self.assertIsNotNone(run_until(self.m, sim, "carried", limit_s=1,
                                       carried=True, upright=False))
        self.assertEqual(sim.policy.cmd, (0.0, 0.0, 0.0), "it kept walking")

    def test_the_fall_reflex_does_not_fire_on_a_duck_that_is_being_held(self):
        # THE regression this whole exclusion clause exists for. A dangling
        # duck reads `not upright` within about a second of the lift; without
        # `not carried` the machine storms `fallen`, latches a wake, plays
        # `droop` and says "I have gone over" at somebody who is holding it.
        self.m.enter("stroll_right", 0.0)
        sim = _PetSim(x=0.0)
        for i in range(250):                       # five seconds of dangling
            tick(self.m, sim, i * DT, carried=True, upright=False)
        self.assertEqual(self.m.current, "carried")
        self.assertIsNone(self.m.nodes["carried"]["wake"])
        self.assertIsNone(self.m.nodes["carried"]["say"])

    def test_a_gentle_put_down_is_not_a_fall(self):
        # Let go after five seconds, upright and on its feet: `carried` waits
        # its LAND_S and hands back to the day with nothing said.
        self.m.enter("carried", 0.0)
        sim = _PetSim(x=0.0)
        for i in range(250):
            tick(self.m, sim, i * DT, carried=True, upright=False)
        t = 250 * DT
        self.assertIsNotNone(run_until(self.m, sim, "settle", t0=t, limit_s=2))
        self.assertNotEqual(self.m.current, "fallen")

    def test_a_bad_landing_still_falls(self):
        # ...and the other half: dropped on its side, the ladder that already
        # exists catches it. `carried` -> `settle` -> `fallen`, no landing
        # node of its own, no new wake.
        self.m.enter("carried", 0.0)
        sim = _PetSim(x=0.0)
        for i in range(250):
            tick(self.m, sim, i * DT, carried=True, upright=False)
        t = round(250 * DT, 3)
        t = run_until(self.m, sim, "settle", t0=t, limit_s=2, upright=False)
        self.assertIsNotNone(t)
        self.assertIsNotNone(run_until(self.m, sim, "fallen", t0=t, limit_s=2,
                                       upright=False))

    def test_a_long_hold_does_not_fire_the_reflex_the_moment_it_ends(self):
        # `elapsed_s` in `carried` is however long the human held on, so on
        # the release tick it is already well past the reflex's 0.5 s. Without
        # `node != 'carried'` on the fall reflex, every put-down — however
        # gentle — would be a wake before the feet touched anything.
        self.m.enter("carried", 0.0)
        sim = _PetSim(x=0.0)
        for i in range(500):                       # ten seconds of holding
            tick(self.m, sim, i * DT, carried=True, upright=False)
        # The very next tick is the release. `carried`'s own deadline is
        # already long past, so it hands straight back to `settle` — and what
        # matters is that it is NOT `fallen`: the reflex got no say, and the
        # duck gets `settle`'s half-second on its feet before anything decides
        # it went over.
        tick(self.m, sim, round(500 * DT, 3), carried=False, upright=False)
        self.assertEqual(self.m.current, "settle")

    def test_carry_outranks_petting_and_the_ball(self):
        # All three globals true at once. File order is priority, and being
        # held is the most a thing can be done to a duck.
        self.m.enter("stroll_right", 0.0)
        sim = _PetSim(x=0.0)
        for i in range(50):
            tick(self.m, sim, i * DT, carried=True, upright=True,
                 **{**toy(), "touch.petted": True, "touch.age_s": 0.0})
        self.assertEqual(self.m.current, "carried")

    def test_a_duck_in_a_hand_is_not_chasing_anything(self):
        # The ball gate carries `not carried` of its own, so a toy rolling
        # past underneath a dangling duck is not an invitation.
        self.m.enter("carried", 0.0)
        sim = _PetSim(x=0.0)
        for i in range(200):
            tick(self.m, sim, i * DT, carried=True, upright=True, **toy())
        self.assertEqual(self.m.current, "carried")


class PetPlaysWithTheBall(unittest.TestCase):
    """The toy lane: notice it moving, walk it down, boot it, watch it go.

    Not one new behavior in it — `approach_ball` and `kick` are the striker's
    own controllers with the goal-bearing term switched off, `search_ball` is
    the glance `peek` already uses. So what is tested here is the graph and
    the gate, and the gate is the interesting half.
    """

    def setUp(self):
        self.m = Machine.load(PET, rng=_NeverRolls())     # the high half
        self.m.armed = True

    def enter(self, node, sim, roll=0.99):
        self.m.enter(node, 0.0)
        self.m.roll = roll
        return sim

    def test_a_rolling_ball_does_not_interrupt_the_lap_while_parked(self):
        # The inversion of the lane's own headline test, on purpose: the
        # see_ball door is commented out in pet.toml (chasing monopolised the
        # first live test), so today a rolling ball is furniture like a still
        # one. When the park is lifted, flip this back to
        # `run_until(... "see_ball" ...) is not None`.
        sim = self.enter("stroll_right", _PetSim(x=0.0, yaw=0.0))
        self.assertIsNone(run_until(self.m, sim, "see_ball", limit_s=2,
                                    **toy()))

    def test_a_ball_lying_still_is_furniture(self):
        # The deviation this lane is built on, and the reason for it: a guard
        # that reads "there is a ball in view" is true for as long as the ball
        # EXISTS, and the loop it opens (see_ball -> chase -> boot -> watch ->
        # settle -> a leg -> see_ball) closes in about thirty seconds. That is
        # a greeting chirp twice a minute from a toy nobody touched — the
        # exact failure `notice`'s MOVING_MPS term was added for.
        sim = self.enter("stroll_right", _PetSim(x=0.0, yaw=0.0))
        t = 0.0
        while t < 20.0:
            t = round(t + DT, 3)
            tick(self.m, sim, t, **toy(speed_mps=0.0))
        self.assertNotIn(self.m.current,
                         ("see_ball", "chase_ball", "boot", "watch_ball"))

    def test_a_ball_whose_speed_is_not_known_yet_is_not_chased(self):
        # `ball_seen.speed_mps` is null until the detector's track spans
        # 0.25 s, and a null compares False. That half-second of confirmation
        # is the same one `notice` spends on a pointer.
        sim = self.enter("stroll_right", _PetSim(x=0.0, yaw=0.0))
        for i in range(200):
            tick(self.m, sim, i * DT, **toy(speed_mps=None))
        self.assertEqual(self.m.current, "stroll_right")

    def test_a_stale_sighting_is_a_memory_not_an_invitation(self):
        sim = self.enter("stroll_right", _PetSim(x=0.0, yaw=0.0))
        for i in range(200):
            tick(self.m, sim, i * DT, **toy(age_s=1.2))
        self.assertEqual(self.m.current, "stroll_right")

    def test_the_ball_does_not_interrupt_the_nap(self):
        # `not sitting` is what keeps a nap a nap. A duck that got up to chase
        # a ball every time one rolled past would never finish a sleep.
        sim = self.enter("doze", _PetSim(x=0.0, yaw=0.0))
        sim.sitting = True
        for i in range(400):
            tick(self.m, sim, i * DT, sitting=True, **toy())
        self.assertEqual(self.m.current, "doze")

    def test_the_ball_does_not_talk_a_jammed_duck_out_of_escaping(self):
        # The jam ladder has already asked for help and is measuring whether
        # anything moved. A toy in view is not an answer to that question.
        # Jammed at the wall the node is escaping FROM: an escape leg that has
        # already succeeded is allowed to end, and what it ends into is the
        # ordinary day, where a toy is fair game again.
        for node, x in (("stuck", 1.02), ("unjam", 1.02), ("escape_left", 1.02),
                        ("escape_right", -1.02), ("wedged", 1.02)):
            with self.subTest(node=node):
                m = Machine.load(PET, rng=_NeverRolls())
                m.armed = True
                m.enter(node, 0.0)
                sim = _PetSim(x=x, yaw=0.0)
                for i in range(40):
                    tick(m, sim, i * DT, **toy())
                self.assertNotEqual(m.current, "see_ball")

    def test_a_hand_on_the_duck_outranks_the_toy(self):
        # Global order is priority: the more a thing is being DONE TO the
        # duck, the higher it goes.
        sim = self.enter("stroll_right", _PetSim(x=0.0, yaw=0.0))
        t = run_until(self.m, sim, "petted", limit_s=2,
                      **{**toy(), "touch.petted": True, "touch.age_s": 0.0})
        self.assertIsNotNone(t)
        self.assertEqual(self.m.current, "petted")

    def test_it_walks_the_ball_down_and_boots_it(self):
        sim = self.enter("see_ball", _PetSim(x=0.0, yaw=0.0))
        t = run_until(self.m, sim, "chase_ball", limit_s=3, **toy())
        self.assertIsNotNone(t)
        # ...and it does not swing until the ball is IN the pocket and has
        # stopped rolling: kicking a moving ball is how you whiff.
        rolling = {**toy(forward_m=0.112, left_m=-0.068, speed_mps=0.4)}
        for i in range(200):
            tick(self.m, sim, t + i * DT, **rolling)
        self.assertEqual(self.m.current, "chase_ball")
        settled = {**toy(forward_m=0.112, left_m=-0.068, speed_mps=0.0)}
        t2 = run_until(self.m, sim, "boot", t0=t + 4.0, limit_s=3, **settled)
        self.assertIsNotNone(t2)
        # An honest kick: the ball is hit where it lies, never staged onto the
        # foot first. On a Dock, staging would be a magic trick.
        tick(self.m, sim, t2 + 1.0, **settled)
        self.assertEqual(sim.tricks[-1], ("kick_right", False))

    def test_the_fine_range_servo_settles_where_the_kick_guard_can_fire(self):
        # CLOSED LOOP, from est_left ~= 0, because the guard-only test above
        # injects `left_m = -0.068` straight into the digest and never asks
        # the controller to produce it.
        #
        # `bhv_approach_ball`'s fine-range servo stops turning inside
        # [pocket_left_min, pocket_left_max], and this node's entry guard
        # wants (-0.086, -0.052) — striker.toml's deep window. With the servo
        # left on machine.py's DEFAULTS (-0.020 / -0.080) it is satisfied at
        # about -0.02 to -0.04, vx is already in its own dead band, and
        # nothing moves again: the controller is happy, the guard never fires,
        # `boot` is unreachable and the whole play lane ends silently at the
        # 25 s deadline. "Controller satisfied" has to IMPLY "guard fires",
        # which means the servo window must sit inside the guard's.
        #
        # The loop below is the same coarse kinematics as `PetPutters`: the
        # ball's trunk-frame bearing rotates against the duck's yaw at the
        # measured turn rate, and the sensed pair refreshes at the detector's
        # own 5 Hz — which is where the overshoot per correction comes from.
        m = Machine.load(PET, rng=_NeverRolls())
        m.armed = True
        m.enter("chase_ball", 0.0)
        sim = _PetSim(x=0.0, yaw=0.0)
        fwd, left = 0.125, 0.0          # arrived, lined up on nothing yet
        sensed, t, i = (fwd, left), 0.0, 0
        while t < 12.0 and m.current == "chase_ball":
            t, i = round(t + DT, 3), i + 1
            tick(m, sim, t, **toy(speed_mps=0.0, age_s=0.0,
                                  forward_m=sensed[0], left_m=sensed[1]))
            _walk, turn = gait(*sim.policy.cmd[::2])
            b = math.atan2(left, fwd) - turn * DT
            d = math.hypot(fwd, left)
            fwd, left = d * math.cos(b), d * math.sin(b)
            if i % 10 == 0:             # the detector's own 5 Hz
                sensed = (fwd, left)
        self.assertEqual(m.current, "boot",
                         f"the servo parked at est_left={left:.4f}, "
                         f"est_forward={fwd:.4f} — outside the guard, so the "
                         "duck stands over its toy until the deadline")

    def test_the_play_loop_comes_home(self):
        # The ball is never seen again: search, then back to the ordinary day.
        sim = self.enter("chase_ball", _PetSim(x=0.0, yaw=0.0))
        t = run_until(self.m, sim, "find_ball", limit_s=3)
        self.assertIsNotNone(t)
        self.assertIsNotNone(run_until(self.m, sim, "settle", t0=t, limit_s=8))

    def test_a_chase_that_never_closes_gives_up(self):
        sim = self.enter("chase_ball", _PetSim(x=0.0, yaw=0.0))
        # Visible the whole time, never in the pocket: the deadline is what
        # stops the duck standing on the Dock trying forever.
        t = run_until(self.m, sim, "settle", limit_s=30, **toy())
        self.assertIsNotNone(t)
        self.assertGreater(t, 25.0)

    def test_the_rally_is_a_coin_and_not_a_treadmill(self):
        # `watch_ball` is where a bout ends. High coin: one more chase. Low
        # coin: back to work. Everything a machine can sense after a kick is
        # downstream of that same kick, so an honest coin is the only
        # rationing device here that is not a feedback loop with itself.
        for roll, expect in ((0.99, "chase_ball"), (0.10, "ball_rest")):
            with self.subTest(roll=roll):
                m = Machine.load(PET, rng=_AlwaysRolls(roll))
                m.armed = True
                m.enter("watch_ball", 0.0)
                m.roll = roll
                sim = _PetSim(x=0.0, yaw=0.0)
                # `speed_mps` low: the booted ball has finished rolling, which
                # is exactly what three seconds of watching is for.
                self.assertIsNotNone(
                    run_until(m, sim, expect, limit_s=5,
                              **toy(speed_mps=0.0)))

    def test_the_whole_lane_is_silent(self):
        # `see_ball` wanted a greet and measured its way out of one: a duck
        # pacing a 2 m strip walks into its own toy on most laps, so even
        # behind the range and speed gates a real daemon opened six bouts in
        # ten minutes — a chirp every hundred seconds, all day. The duck says
        # this by walking over and booting something instead.
        for node in ("see_ball", "chase_ball", "find_ball", "boot",
                     "watch_ball", "ball_rest"):
            with self.subTest(node=node):
                self.assertIsNone(self.m.nodes[node]["emote"])
                self.assertIsNone(self.m.nodes[node]["say"])
                self.assertIsNone(self.m.nodes[node]["wake"])

    def test_a_ball_too_far_away_to_be_believed_is_not_an_invitation(self):
        # The detector's range comes from inverting the solid angle the blob
        # covers, so it falls apart with distance: measured live, a stationary
        # ball at 1.95 m read 6.4 m/s. PLAY_M keeps the gate inside the range
        # where a sighting means something — which is also the only range at
        # which a human could have caused it, since the toy is only clickable
        # while it is inside the duck's own 0.39 m window.
        sim = self.enter("stroll_right", _PetSim(x=0.0, yaw=0.0))
        for i in range(200):
            tick(self.m, sim, i * DT, **toy(forward_m=1.90, left_m=0.1))
        self.assertEqual(self.m.current, "stroll_right")

    def test_a_speed_no_ball_on_this_dock_could_have_is_a_broken_sighting(self):
        # PET_PUSH_MAX is 2.0 m/s and nothing here hits harder than a shove.
        sim = self.enter("stroll_right", _PetSim(x=0.0, yaw=0.0))
        for i in range(200):
            tick(self.m, sim, i * DT, **toy(speed_mps=6.4, forward_m=0.30))
        self.assertEqual(self.m.current, "stroll_right")

    def test_a_bout_does_not_end_until_the_toy_has_stopped(self):
        # The node that was missing, and the one the offline putter could not
        # have found: without it the duck's own boot sets the ball rolling,
        # the rolling ball satisfies the play gate, and the loop feeds itself.
        sim = self.enter("watch_ball", _PetSim(x=0.0, yaw=0.0), roll=0.10)
        t = run_until(self.m, sim, "ball_rest", limit_s=5,
                      **toy(speed_mps=0.0))
        self.assertIsNotNone(t)
        # Still rolling: the duck waits, and — the point — cannot re-enter
        # `see_ball` from here, because the gate excludes this node.
        for i in range(300):
            tick(self.m, sim, t + i * DT, **toy(speed_mps=0.8, forward_m=0.30))
        self.assertEqual(self.m.current, "ball_rest")
        # Stopped: the day resumes.
        self.assertIsNotNone(run_until(self.m, sim, "settle", t0=t + 6.0,
                                       limit_s=4, **toy(speed_mps=0.0)))


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
               seed=7, cursor_x=None, cursor_speed=0.0):
        """`cursor_x` parks a mouse pointer at that sim x for the whole run,
        at Dock height, moving at `cursor_speed`. The distance is recomputed
        every tick against the duck's live position, the way the daemon does
        it — a fixed `dist_m` would be a cursor that follows the duck."""
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
            seen = {} if cursor_x is None else {
                "cursor.present": True, "cursor.x_m": cursor_x,
                "cursor.z_m": 0.10, "cursor.dx_m": cursor_x - x,
                "cursor.dist_m": abs(cursor_x - x), "cursor.age_s": 0.0,
                "cursor.near_floor": True, "cursor.speed_mps": cursor_speed}
            tick(m, sim, t, sitting=sim.sitting, **seen)
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

    def test_a_mouse_left_over_the_dock_is_not_an_hour_of_chirping(self):
        # The measurement that added `cursor.speed_mps > 0.05` to the notice
        # gate, kept here so it cannot quietly come back out. `regard` carries
        # head_tilt, and head_tilt carries the inquire — so an entry here is a
        # NOISE, and the budget for one is the same budget the walls are on.
        # Without the movement term this ran 54-60 an hour at every x tried.
        for cursor_x in (0.30, 0.90, -0.60):
            with self.subTest(cursor_x=cursor_x):
                _, _, _, entered = self.putter(3600.0, cursor_x=cursor_x)
                self.assertEqual(entered.count("regard"), 0,
                                 "a parked mouse pointer made the duck chirp")

    def test_a_hand_that_is_actually_moving_does_get_noticed(self):
        # The other side of that ration: it must not have silenced the
        # feature. A hand waving about near the duck over a simulated hour is
        # a hand the duck goes and looks at, repeatedly.
        seen = 0
        for seed in (3, 7, 11, 19):
            _, _, _, entered = self.putter(3600.0, seed=seed, cursor_x=0.30,
                                           cursor_speed=0.20)
            seen += entered.count("regard")
        self.assertGreater(seen, 20, f"{seen} regards in 4 simulated hours")

    def test_a_hand_in_the_room_is_not_an_inquire_a_minute(self):
        # THE MISSING UPPER BOUND, and the one the suite was short of: the
        # test above asserts only `> 20` over four hours, so the 290 `regard`
        # entries this loop actually produces passed it happily — 48-73 an
        # hour, a chirp every fifty seconds all day, nine times the sound
        # budget at the top of pet.toml. Looking is free and stays unbounded;
        # the SOUND is what is counted here, and it lives on `greet_hand`
        # behind a 12% coin.
        #
        # The budget: the two ends of a nap are about sixteen an hour between
        # them, so the hand may have about the same again. Measured with the
        # coin in place: 9.8-11.0 an hour, against 74.2 `regard` entries in
        # the same runs. Four seeds, four simulated hours, five cursor
        # positions across the strip.
        for cursor_x in (0.30, 0.60, 0.90, -0.60, 1.10):
            greets = 0
            for seed in (3, 7, 11, 19):
                _, _, _, entered = self.putter(3600.0, seed=seed,
                                               cursor_x=cursor_x,
                                               cursor_speed=0.20)
                greets += entered.count("greet_hand")
            with self.subTest(cursor_x=cursor_x):
                self.assertLessEqual(greets / 4.0, 16.0,
                                     f"{greets / 4.0:.1f} inquires an hour at "
                                     f"cursor_x={cursor_x}")

    def test_a_hand_in_the_room_does_not_move_the_wake_budget(self):
        # The whole cursor lane adds no wake and no `say`. It must not add one
        # sideways either, by walking the duck into a wall it then reports.
        wakes = 0
        for seed in (3, 7, 11, 19):
            _, _, _, entered = self.putter(3600.0, seed=seed, cursor_x=0.90,
                                           cursor_speed=0.20)
            wakes += sum(1 for n in entered
                         if n in ("bored", "stuck", "fallen"))
        self.assertLessEqual(wakes, 6, f"{wakes} wakes in 4 simulated hours")

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
