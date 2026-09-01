"""Tests for the desktop pet's sim half: the scene, the frame, the mapping.

Two halves, like the rest of tests/:

  * a real MuJoCo model, compiled from scenes/scene_desktop.xml bound to the
    microduck_rl clone next door, for everything that is only true if the
    physics agrees — that a wall stops a duck, that a repositioned ledge
    catches one, that the alpha channel really is the duck and nothing else.
    Skipped (not failed) when the clone is not there, the way a machine
    without the robot repo should be told rather than shouted at.
  * a bare `DuckSim.__new__` instance, no MuJoCo, for the arithmetic that
    decides where the window goes — the screen mapping, the wall margin, the
    inhabited latch, the premultiplied downsample.

    uv run --with pytest pytest tests/
"""

import json
import math
import os
import time
import unittest
from collections import deque

import numpy as np

from microduck_mcp.sim_server import (DUCK_HEIGHT_M, LOCAL_SCENES,
                                      PET_CARRY_EQ, PET_CARRY_MIN_Z_M,
                                      PET_CARRY_TIMEOUT_S, PET_FRAME_PX,
                                      PET_HAND_BODY, PET_INHABITED_S,
                                      PET_LIFT_TRIGGER_M, PET_PARK_POS,
                                      PET_PLATFORM_GEOMS, PET_PROP_GEOMS,
                                      PET_PX_PER_METER, PET_RAIL_GEOMS,
                                      MOUTH_BODY, PET_SOLID_HALF_H,
                                      PET_SOLID_SINK, PET_WALL_GEOMS, DuckSim,
                                      local_scene_xml, local_scenes_dir)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RL_REPO = os.environ.get("MICRODUCK_RL_REPO",
                         os.path.join(REPO, os.pardir, "microduck_rl"))


# ---------------------------------------------------------------- the model

def _desktop_model():
    """Compile scene_desktop.xml, or None if microduck_rl is not next door."""
    try:
        xml_text, path = local_scene_xml("desktop", RL_REPO)
    except (FileNotFoundError, KeyError):
        return None, None
    import mujoco
    from microduck_mcp.sim_server import load_model_with_mouth
    model, _ = load_model_with_mouth(path, xml_text)
    return model, mujoco.MjData(model)


_MODEL, _DATA = _desktop_model()
needs_model = unittest.skipIf(
    _MODEL is None, f"no microduck_rl clone at {RL_REPO} — scene tests skipped")


class _Policy:
    """Just enough policy for pet_state to describe the duck (test_emote_sim's
    trick): the pet never asks the brain anything it could not read off the
    body.

    `ball_qpos_adr`/`ball_qvel_adr` are the two the real PolicyInference
    resolves from the `ball_free` joint, and they are here because the toy's
    position is read through them everywhere — `_pet_ball_state`, the
    `push {target: "ball"}` branch, the detector's own gate. None on a scene
    with no ball, which is exactly what the real one reports too.
    """

    def __init__(self, ball_qpos_adr=None, ball_qvel_adr=None):
        self.sit_mode = False
        self.current_policy = "standing"
        self.behavior_mode = None
        self.vel_cmd = np.zeros(3, dtype=np.float32)
        self.ball_qpos_adr = ball_qpos_adr
        self.ball_qvel_adr = ball_qvel_adr

    def get_projected_gravity(self):
        return np.array([0.0, 0.0, -1.0], dtype=np.float32)


def _pet_sim(model, data):
    """A DuckSim with only the pet's parts wired up.

    Everything the pet touches is model/data plus a config dict, so the real
    handlers can run without ONNX sessions or a 50 Hz loop.
    """
    import mujoco
    sim = DuckSim.__new__(DuckSim)
    sim.model, sim.data = model, data
    bj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
    sim.policy = _Policy(
        ball_qpos_adr=int(model.jnt_qposadr[bj]) if bj >= 0 else None,
        ball_qvel_adr=int(model.jnt_dofadr[bj]) if bj >= 0 else None)
    sim.scene, sim.scene_key = "desktop", "desktop"
    sim.sim_time = 0.0
    sim.machine = None
    sim.pet = sim._pet_default_config()
    sim._pet_geoms, sim._pet_mocap = {}, {}
    for name in (PET_WALL_GEOMS + PET_RAIL_GEOMS + PET_PLATFORM_GEOMS
                 + PET_PROP_GEOMS):
        sim._pet_geoms[name] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        sim._pet_mocap[name] = int(model.body_mocapid[bid]) if bid >= 0 else -1
    sim.pet_scene = True
    # The real method, not a copy of it: the duck/ball split is the whole
    # subject of half the tests below, and a fixture that hand-rolled its own
    # partition would be testing the fixture.
    sim._pet_resolve_masks()
    sim._pet_renderers = {}
    sim._pet_renderer = None
    sim._pet_renderer_px = 0
    sim._pet_option = None
    sim._pet_wall_pinned = False
    sim._pet_seg_ok = True
    sim._pet_cache = None
    sim._pet_cursor = None
    sim._pet_touch = {"t": None, "count": 0, "ack_t": None}
    # The pick-up's three ids, resolved by name exactly the way __init__ does
    # them — the weld is the whole subject of the carry tests below and a
    # fixture that hand-rolled its own would be testing the fixture.
    sim._pet_carry = None
    sim._pet_carry_eq = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY,
                                          PET_CARRY_EQ)
    sim._pet_hand_mocap = sim._pet_mocap.get(PET_HAND_BODY, -1)
    sim._pet_trunk_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                            "trunk_base")
    sim._mcp_intent_t = 0.0
    sim._mcp_intent_cmd = None
    sim.qpos_adr = int(model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")])
    sim.qvel_adr = int(model.jnt_dofadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")])
    # The beak is a mocap body of its own, and an unposed one sits at the
    # world origin — where it is still a duck geom and still lands in the
    # cutout. DuckSim.__init__ poses it before the first render; so does this.
    sim.mouth_opening = 0.0
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mouth_plate")
    sim._mouth_mocap_id = int(model.body_mocapid[bid]) if bid >= 0 else -1
    sim._mouth_head_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                           MOUTH_BODY)
    return sim


def _reset_pose(sim, x=0.0, y=0.0, z=None):
    """Put the duck in the scene's STAND keyframe at (x, y, z), at rest.

    Copies the keyframe rather than calling mj_resetDataKeyframe, which would
    also reset mocap_pos to the compile-time parking spots and quietly undo
    every wall and ledge the test just placed. (The daemon's own `reset` only
    rewinds qpos/qvel, for the same reason.)
    """
    import mujoco
    sim.data.qpos[:] = sim.model.key_qpos[0]
    sim.data.ctrl[:] = sim.model.key_ctrl[0]
    sim.data.qpos[sim.qpos_adr] = x
    sim.data.qpos[sim.qpos_adr + 1] = y
    if z is not None:
        sim.data.qpos[sim.qpos_adr + 2] = z
    sim.data.qvel[:] = 0.0
    mujoco.mj_forward(sim.model, sim.data)
    sim.mouth_tick()          # park the beak on the head before any render
    mujoco.mj_forward(sim.model, sim.data)


def _stub_for_reset(sim):
    """Fill in the half of `_handle_reset` a policy-less fixture has no answer
    for, so the REAL handler can be run.

    Everything here is brain: the action vector it zeroes, the head and body
    commands it clears, the detector it re-runs. None of it is what a reset
    test is about — the point is to reach the real function rather than
    re-implement three of its lines in the test and assert on those, which is
    a test that passes when the handler stops doing them at all.
    """
    sim._qpos0 = getattr(sim, "_qpos0", sim.data.qpos.copy())
    sim.policy.n_joints = int(sim.model.nu)
    sim.policy.default_pose = sim.data.ctrl.copy()
    sim.policy.body_cmd = np.zeros(3, dtype=np.float32)
    sim.policy.ground_pick_mode = False
    sim.policy.set_vel_cmd = lambda *a: None
    sim.policy.head_offset = np.zeros(4, dtype=np.float32)
    sim.mouth_opening, sim._emote = 0.0, None
    sim._det_step = 0
    sim._ball_seen, sim._ball_seen_t = {}, 0.0
    sim._ball_track = deque()
    sim._goal_seen, sim._goal_seen_t = {}, 0.0
    sim._goal_fix = sim._goal_azimuth_w = None
    sim.referee = None
    sim._detect_ball = lambda: None
    sim.get_state = lambda: {"ok": True}
    return sim


def _settle(sim, steps, watch_geom=None):
    """Step the passive model; report whether `watch_geom` ever made contact."""
    import mujoco
    touched = False
    for _ in range(steps):
        mujoco.mj_step(sim.model, sim.data)
        if watch_geom is not None:
            for c in range(sim.data.ncon):
                if watch_geom in (sim.data.contact[c].geom1,
                                  sim.data.contact[c].geom2):
                    touched = True
    return touched


# ---------------------------------------------------------------- the scene

class SceneTemplate(unittest.TestCase):
    def test_template_ships_with_the_repo(self):
        path = os.path.join(local_scenes_dir(), LOCAL_SCENES["desktop"])
        self.assertTrue(os.path.isfile(path), path)

    def test_template_is_bound_to_the_given_rl_repo(self):
        if _MODEL is None:
            self.skipTest("no microduck_rl clone")
        text, _ = local_scene_xml("desktop", RL_REPO)
        # Both placeholders gone, both replaced with paths that exist: the
        # include and the meshdir need different roots, so a template that
        # substituted only one would compile on this machine and nowhere else.
        self.assertNotIn("@ROBOT_XML@", text)
        self.assertNotIn("@MESHDIR@", text)
        self.assertIn("robot_allcollisions.xml", text)
        self.assertIn(os.path.join("microduck", "assets"), text)

    def test_a_missing_rl_repo_says_so(self):
        with self.assertRaises(FileNotFoundError) as e:
            local_scene_xml("desktop", "/nonexistent/microduck_rl")
        self.assertIn("--rl-repo", str(e.exception))


@needs_model
class SceneGeometry(unittest.TestCase):
    def test_every_movable_part_is_a_mocap_body(self):
        # The whole reason the pet's world can be reshaped without a
        # recompile. A plain worldbody geom would move on screen and stop
        # colliding, which is the worst of both.
        import mujoco
        for name in PET_WALL_GEOMS + PET_RAIL_GEOMS + PET_PLATFORM_GEOMS:
            bid = mujoco.mj_name2id(_MODEL, mujoco.mjtObj.mjOBJ_BODY, name)
            self.assertGreaterEqual(bid, 0, f"{name} has no body")
            self.assertGreaterEqual(int(_MODEL.body_mocapid[bid]), 0,
                                    f"{name} is not a mocap body")

    def test_twelve_platforms_are_pre_allocated_and_parked(self):
        import mujoco
        sim = _pet_sim(_MODEL, mujoco.MjData(_MODEL))
        self.assertEqual(len(PET_PLATFORM_GEOMS), 12)
        for name in PET_PLATFORM_GEOMS:
            pos = sim.data.mocap_pos[sim._pet_mocap[name]]
            np.testing.assert_allclose(pos, PET_PARK_POS)

    def test_the_invisible_world_is_hidden_and_the_duck_is_not(self):
        import mujoco
        for name in ("floor",) + PET_WALL_GEOMS + PET_PLATFORM_GEOMS:
            gid = mujoco.mj_name2id(_MODEL, mujoco.mjtObj.mjOBJ_GEOM, name)
            self.assertEqual(int(_MODEL.geom_group[gid]), 4,
                             f"{name} must be in the culled group")
            self.assertEqual(int(_MODEL.geom_contype[gid]), 1,
                             f"{name} must still collide")

    def test_the_extent_is_pinned_so_parked_boxes_cannot_clip_the_duck(self):
        # Left to itself the extent would be computed from a box out at
        # y=+10 and the near/far planes would swallow the duck.
        self.assertAlmostEqual(float(_MODEL.stat.extent), 0.6, places=6)


# ---------------------------------------------------------------- the frame

@needs_model
class Frame(unittest.TestCase):
    def setUp(self):
        import mujoco
        self.sim = _pet_sim(_MODEL, mujoco.MjData(_MODEL))
        _reset_pose(self.sim)

    def tearDown(self):
        for r in self.sim._pet_renderers.values():
            r.close()
        self.sim._pet_renderers.clear()
        self.sim._pet_renderer = None

    def _frame(self, **req):
        resp = self.sim._handle_pet_frame(req)
        self.assertTrue(resp.get("ok"), resp.get("error"))
        from PIL import Image
        import io
        return resp, np.array(Image.open(io.BytesIO(resp["png"])))

    def test_shape_and_channels(self):
        resp, px = self._frame()
        self.assertEqual(px.shape, (PET_FRAME_PX, PET_FRAME_PX, 4))
        self.assertEqual((resp["width"], resp["height"]),
                         (PET_FRAME_PX, PET_FRAME_PX))
        self.assertEqual(resp["alpha"], "segmentation")

    def test_alpha_is_a_duck_shaped_hole_in_a_transparent_frame(self):
        _, px = self._frame()
        alpha = px[:, :, 3]
        self.assertEqual(int(alpha[0, 0]), 0, "the corners must be see-through")
        covered = float((alpha > 0).mean())
        self.assertTrue(0.02 < covered < 0.35,
                        f"{covered:.3f} of the frame is opaque — that is not a duck")

    def test_supersampling_gives_real_antialiasing(self):
        # The segmentation mask is perfectly binary; the 2x box filter is what
        # turns it into an edge. ss=1 has nothing to average, so it must not.
        _, aa = self._frame(supersample=2)
        _, hard = self._frame(supersample=1)
        self.assertGreater(len(np.unique(aa[:, :, 3])), 2)
        self.assertEqual(sorted(np.unique(hard[:, :, 3]).tolist()), [0, 255])

    def test_no_background_bleeds_into_the_edge_pixels(self):
        # Premultiplied downsampling, checked where it actually shows: the
        # backdrop is magenta and the duck has no magenta on it anywhere.
        _, px = self._frame()
        opaque = px[:, :, 3] > 100
        rgb = px[:, :, :3][opaque].astype(int)
        magenta = (rgb[:, 0] > 180) & (rgb[:, 1] < 90) & (rgb[:, 2] > 180)
        self.assertEqual(int(magenta.sum()), 0)

    def test_the_chroma_fallback_finds_the_same_duck(self):
        _, seg = self._frame()
        self.sim._pet_seg_ok = False
        self.sim._pet_cache = None
        resp, chroma = self._frame()
        self.assertEqual(resp["alpha"], "chroma")
        a, b = seg[:, :, 3] > 128, chroma[:, :, 3] > 128
        agree = float((a == b).mean())
        self.assertGreater(agree, 0.98, "the fallback should be a near miss, "
                                        "not a different picture")

    def test_the_duck_is_the_size_the_mapping_promises(self):
        # 180 px of duck at 656 px/m, measured on the silhouette. The live
        # standing pose is a little more crouched than the STAND keyframe the
        # constant comes from, hence the tolerance.
        _, px = self._frame()
        ys, _ = np.nonzero(px[:, :, 3] > 0)
        want = DUCK_HEIGHT_M * self.sim.pet["px_per_meter"]
        self.assertAlmostEqual((ys.max() - ys.min() + 1) / want, 1.0, delta=0.12)

    def test_the_floor_lands_where_the_window_expects_it(self):
        # The whole screen mapping in one assertion: the duck's lowest pixel
        # is the sim floor, and the sim floor is floor_pad_px above the bottom
        # of the frame — which is what the app pins to the Dock's top edge.
        _reset_pose(self.sim)
        _settle(self.sim, 200)
        self.sim._pet_cache = None
        _, px = self._frame()
        ys, _ = np.nonzero(px[:, :, 3] > 0)
        floor_row = PET_FRAME_PX - self.sim.pet["floor_pad_px"]
        self.assertLessEqual(abs(int(ys.max()) - floor_row), 3)

    def test_a_frame_of_another_size_still_stands_on_the_floor_line(self):
        # `size_px` overrides the configured frame, and the camera has to be
        # framed for the size actually rendered: framing off the config would
        # leave the floor somewhere other than floor_pad_px up from the
        # bottom, while pet_state.screen went on claiming otherwise — and the
        # window is hung off pet_state.screen.
        _reset_pose(self.sim)
        _settle(self.sim, 200)
        self.sim._pet_cache = None
        big = PET_FRAME_PX + 128
        resp, px = self._frame(size_px=big)
        self.assertEqual(px.shape[0], big)
        ys, _ = np.nonzero(px[:, :, 3] > 0)
        floor_row = big - resp["screen"]["floor_px_from_bottom"]
        self.assertLessEqual(abs(int(ys.max()) - floor_row), 3)

    def test_a_frame_reports_the_box_its_duck_is_in(self):
        # The overlay decides whether to swallow a click or let it fall
        # through to a Dock icon by asking "is the cursor on the duck", and
        # `bbox` is what makes that the real silhouette rather than a nominal
        # standing rectangle. Frame pixels, top-left origin, half-open.
        _reset_pose(self.sim)
        _settle(self.sim, 200)
        self.sim._pet_cache = None
        resp, px = self._frame()
        x0, y0, x1, y1 = resp["bbox"]
        ys, xs = np.nonzero(px[:, :, 3] > 0)
        self.assertEqual([x0, y0, x1, y1],
                         [int(xs.min()), int(ys.min()),
                          int(xs.max()) + 1, int(ys.max()) + 1])
        self.assertTrue(0 <= x0 < x1 <= PET_FRAME_PX)
        self.assertTrue(0 <= y0 < y1 <= PET_FRAME_PX)
        # ...and the cache must not hand back a frame with no box on it.
        self.assertEqual(self.sim._handle_pet_frame({})["bbox"], resp["bbox"])

    def test_scale_is_orthographic_and_exact(self):
        # Doubling px_per_meter must double the duck, to within a pixel — that
        # only holds under an orthographic camera, and it is what lets a ledge
        # in the sim line up with a window's edge on screen. Measured from
        # half scale up, so the big one still fits inside the frame.
        _, big = self._frame()
        b = np.nonzero(big[:, :, 3] > 0)[0]
        self.sim.pet["px_per_meter"] /= 2
        self.sim._pet_cache = None
        _, small = self._frame()
        s = np.nonzero(small[:, :, 3] > 0)[0]
        ratio = (b.max() - b.min() + 1) / (s.max() - s.min() + 1)
        self.assertAlmostEqual(ratio, 2.0, delta=0.05)

    def test_the_camera_is_put_back_the_way_it_was_found(self):
        # fovy is degrees for every other renderer on this model and metres
        # for this one. Leaving it borrowed turns the AX debug page into a
        # telephoto sliver (it did, once).
        before = (int(_MODEL.vis.global_.orthographic),
                  float(_MODEL.vis.global_.fovy))
        self._frame()
        self.assertEqual((int(_MODEL.vis.global_.orthographic),
                          float(_MODEL.vis.global_.fovy)), before)

    def test_a_frame_carries_the_pose_it_was_taken_at(self):
        _reset_pose(self.sim, x=0.42)
        resp, _ = self._frame()
        self.assertAlmostEqual(resp["base_x_m"], 0.42, places=6)
        self.assertIn("inhabited", resp)
        self.assertIn("config", resp)

    def test_a_second_request_in_the_same_tick_reuses_the_render(self):
        first, _ = self._frame()
        second = self.sim._handle_pet_frame({})
        self.assertFalse(first["cached"])
        self.assertTrue(second["cached"])
        self.sim.sim_time += 0.02
        self.assertFalse(self.sim._handle_pet_frame({})["cached"])


# ------------------------------------------------------------------- the toy

@needs_model
class BallInTheScene(unittest.TestCase):
    """The 70 mm floorball scene_desktop.xml grew, and the two ways it could
    have gone wrong quietly: the wrong size (which the detector would report
    as the wrong distance) and a keyframe that does not know about it."""

    def test_the_scene_has_a_ball_the_detector_can_actually_size(self):
        # BALL_RADIUS_M is not a preference, it is an input to the head
        # camera's range solve: distance comes out of the solid angle the blob
        # covers, so a ball of another size reads at a proportionally wrong
        # distance and the approach stops short of a pocket it thinks it is in.
        import mujoco
        from microduck_mcp.sim_server import BALL_RADIUS_M
        gid = mujoco.mj_name2id(_MODEL, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
        self.assertGreaterEqual(gid, 0, "no ball_geom in the desktop scene")
        self.assertAlmostEqual(float(_MODEL.geom_size[gid][0]), BALL_RADIUS_M)
        # ...and it is DRAWN. Group 4 is the invisible world; the toy is not
        # furniture, it is the one thing in this scene besides the duck that
        # the overlay and the head camera both have to be able to see.
        self.assertEqual(int(_MODEL.geom_group[gid]), 0)

    def test_the_keyframe_covers_the_ball(self):
        # A keyframe shorter than nq is zero-padded, and a zero quaternion is
        # not "no rotation" — it is not a rotation. tests/_reset_pose copies
        # this keyframe wholesale, so a short one would put the ball nowhere
        # in an orientation that is nothing, in every physics test below.
        self.assertEqual(_MODEL.key_qpos.shape[1], _MODEL.nq)
        np.testing.assert_allclose(_MODEL.key_qpos[0][21:24], [0.75, 0.0, 0.035])
        self.assertAlmostEqual(float(np.linalg.norm(_MODEL.key_qpos[0][24:28])),
                               1.0, places=6)

    def test_the_ball_is_its_own_thing_in_the_masks(self):
        import mujoco
        sim = _pet_sim(_MODEL, mujoco.MjData(_MODEL))
        gid = mujoco.mj_name2id(_MODEL, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
        self.assertTrue(bool(sim._pet_ball_geom[gid]))
        self.assertFalse(bool(sim._pet_duck_geom[gid]),
                         "the ball got folded into the duck — that is the bug "
                         "the mask split exists for")
        # The two never overlap, and between them they are everything drawn.
        self.assertFalse(bool((sim._pet_duck_geom & sim._pet_ball_geom).any()))

    def test_the_state_block_says_where_the_toy_is(self):
        import mujoco
        sim = _pet_sim(_MODEL, mujoco.MjData(_MODEL))
        _reset_pose(sim)
        ball = sim._handle_pet_state()["ball"]
        self.assertTrue(ball["present"])
        self.assertAlmostEqual(ball["x_m"], 0.75, places=6)
        self.assertAlmostEqual(ball["dx_m"], 0.75, places=6)
        self.assertAlmostEqual(ball["radius_m"], 0.035, places=6)
        # 0.75 m away with a 0.39 m window: not in the picture, and the app
        # has to be told so rather than left to guess from a null bbox.
        self.assertFalse(ball["in_frame"])
        # ...and it is, once the duck is standing next to it.
        _reset_pose(sim, x=0.70)
        near = sim._handle_pet_state()["ball"]
        self.assertTrue(near["in_frame"])
        self.assertAlmostEqual(near["dx_m"], 0.05, places=6)

    def test_the_ball_is_in_the_picture_but_not_in_the_duck(self):
        import io

        from PIL import Image
        import mujoco
        sim = _pet_sim(_MODEL, mujoco.MjData(_MODEL))
        _reset_pose(sim)
        _settle(sim, 200)
        # Park the toy beside the duck, inside the frame, clear of the beak.
        sim.data.qpos[21:24] = [0.18, 0.0, 0.035]
        sim.data.qpos[24:28] = [1, 0, 0, 0]
        mujoco.mj_forward(sim.model, sim.data)
        sim._pet_cache = None
        try:
            resp = sim._handle_pet_frame({})
            self.assertTrue(resp.get("ok"), resp.get("error"))
            px = np.array(Image.open(io.BytesIO(resp["png"])))
        finally:
            for r in sim._pet_renderers.values():
                r.close()
        duck, ball = resp["bbox"], resp["ball_bbox"]
        self.assertIsNotNone(ball, "the toy was not drawn")
        # The alpha is BOTH of them — one picture, one window.
        alpha = px[:, :, 3]
        for box in (duck, ball):
            x0, y0, x1, y1 = box
            self.assertTrue((alpha[y0:y1, x0:x1] > 0).any())
        # ...and the two boxes are separate, which is the whole reason the
        # masks were split: `bbox` is what the overlay swallows a click
        # inside, and one box around both would be a duck you could grab from
        # wherever the ball happens to be lying.
        self.assertGreaterEqual(ball[0], duck[2] - 1,
                                f"duck {duck} and ball {ball} overlap")
        # The composed alpha's own extent covers the pair, so the split boxes
        # agree with the picture rather than describing a different one.
        cols = np.flatnonzero(alpha.any(axis=0))
        self.assertEqual(int(cols[0]), min(duck[0], ball[0]))
        self.assertEqual(int(cols[-1]) + 1, max(duck[2], ball[2]))

    def test_the_chroma_fallback_admits_it_cannot_tell_them_apart(self):
        # The fallback is "not the backdrop", which is a claim about the
        # background and not about the duck. It cannot say which half of a
        # silhouette is toy, so it says nothing rather than guessing.
        import mujoco
        sim = _pet_sim(_MODEL, mujoco.MjData(_MODEL))
        _reset_pose(sim)
        sim.data.qpos[21] = 0.14
        mujoco.mj_forward(sim.model, sim.data)
        sim._pet_seg_ok = False
        try:
            resp = sim._handle_pet_frame({})
        finally:
            for r in sim._pet_renderers.values():
                r.close()
        self.assertEqual(resp["alpha"], "chroma")
        self.assertIsNone(resp["ball_bbox"])
        self.assertIsNotNone(resp["bbox"])

    def test_the_desktop_scene_now_passes_the_kick_policy_gate(self):
        # The gate used to be "is this scene in LOCAL_SCENES", which was a
        # name standing in for a fact. It is the fact now — and it has to be,
        # or `boot` would be a node with no policy behind it.
        import mujoco
        from microduck_mcp.sim_server import PET_BALL_JOINT, kick_policies_apply
        self.assertGreaterEqual(
            mujoco.mj_name2id(_MODEL, mujoco.mjtObj.mjOBJ_JOINT,
                              PET_BALL_JOINT), 0)
        # ...and the scene passing it is only half the claim. The gate itself
        # is run here, because every test in this file builds `DuckSim.__new__`
        # and never reaches `__init__` — so a regression that put the old
        # `scene_key in LOCAL_SCENES` term back was invisible to the whole
        # suite (verified: it was, on 381 passing tests).
        self.assertTrue(kick_policies_apply("desktop", True))


    def test_a_flicked_ball_stays_between_the_walls(self):
        # The same invisible walls that keep the duck on screen keep its toy
        # there. A ball that could be thrown off the display would be a ball
        # that has to be reset by hand.
        import mujoco
        sim = _pet_sim(_MODEL, mujoco.MjData(_MODEL))
        sim._handle_pet_config({"screen_width_m": 2.634,
                                "wall_margin_m": 0.1227})
        wall = sim._pet_wall_x()
        _reset_pose(sim)
        sim.data.qpos[21:24] = [0.0, 0.0, 0.035]
        sim.data.qvel[20] = 2.0          # PET_PUSH_MAX, straight at the wall
        mujoco.mj_forward(sim.model, sim.data)
        _settle(sim, 600)
        x = float(sim.data.qpos[21])
        self.assertLess(abs(x), wall,
                        f"the ball ended up at x={x:.3f}, past ±{wall:.3f}")

    def test_a_narrow_screen_does_not_spawn_the_toy_inside_a_wall(self):
        # The spawn is baked into the SCENE (x = 0.75) and the walls are
        # placed at RUNTIME from the app's screen, so on any display narrower
        # than ~1.75 m of sim floor the two disagree. A 1512 pt MacBook Pro
        # 14" asked for a 240 pt duck is exactly such a display: 875 px/m,
        # a 1.729 m band, walls at ±0.7417 — and the wall box is 0.02
        # half-thick, so the ball's centre starts INSIDE it. Measured before
        # the clamp: ejected at 1.06 m/s, settled at x = 3.22 m, out of play
        # for the rest of the session and unreachable (the head camera cannot
        # see the wall, so `ball_seen` keeps inviting a chase into it).
        import mujoco
        from microduck_mcp.sim_server import PET_BALL_WALL_PAD_M
        sim = _pet_sim(_MODEL, mujoco.MjData(_MODEL))
        _reset_pose(sim)
        self.assertAlmostEqual(float(sim.data.qpos[21]), 0.75, places=6)
        sim._handle_pet_config({"screen_width_m": 1512 / 875.0,
                                "wall_margin_m": 0.1227})
        wall = sim._pet_wall_x()
        self.assertLess(wall, 0.75, "this screen is not the narrow one")
        x = float(sim.data.qpos[21])
        self.assertLessEqual(abs(x), wall - 0.02 - 0.035 - PET_BALL_WALL_PAD_M
                             + 1e-9, f"the toy is still in the wall at {x:.4f}")
        # ...and it stays there rather than being flung out of the world.
        _settle(sim, 400)
        self.assertLess(abs(float(sim.data.qpos[21])), wall)

    def test_the_toy_is_put_back_in_play_on_a_reset_too(self):
        # `_handle_reset` restores `_qpos0`, which is the SCENE's number
        # again — so without the same clamp every reset re-ejects the ball
        # off the far side of the world, on a screen the daemon has already
        # been told about. (The pet's world is deliberately NOT reset: the
        # walls belong to the screen, not the episode.)
        import mujoco
        sim = _pet_sim(_MODEL, mujoco.MjData(_MODEL))
        _reset_pose(sim)
        sim._qpos0 = sim.model.key_qpos[0].copy()
        sim._handle_pet_config({"screen_width_m": 1512 / 875.0,
                                "wall_margin_m": 0.1227})
        wall = sim._pet_wall_x()
        _stub_for_reset(sim)
        sim._handle_reset()
        self.assertLess(abs(float(sim.data.qpos[21])), wall,
                        "the reset put the toy back inside the wall")

    def test_a_toy_already_in_play_is_left_exactly_where_it_lies(self):
        # The clamp is a trim, not a re-spawn: a ball that is inside the walls
        # is a ball somebody may have just kicked, and moving it would be the
        # daemon rearranging a game in progress.
        import mujoco
        sim = _pet_sim(_MODEL, mujoco.MjData(_MODEL))
        _reset_pose(sim)
        sim.data.qpos[21:24] = [0.30, 0.0, 0.035]
        sim.data.qvel[20] = 0.7
        self.assertFalse(sim._pet_ball_into_play())
        self.assertAlmostEqual(float(sim.data.qpos[21]), 0.30, places=6)
        self.assertAlmostEqual(float(sim.data.qvel[20]), 0.7, places=6)

    def test_a_toy_that_had_to_be_moved_is_not_still_rolling(self):
        import mujoco
        sim = _pet_sim(_MODEL, mujoco.MjData(_MODEL))
        _reset_pose(sim)
        sim.data.qpos[21:24] = [3.2, 0.0, 0.035]
        sim.data.qvel[20:23] = [1.5, 0.0, 0.0]
        self.assertTrue(sim._pet_ball_into_play())
        self.assertLess(abs(float(sim.data.qpos[21])), sim._pet_wall_x())
        self.assertTrue(np.allclose(sim.data.qvel[20:26], 0.0))


class KickPolicyGate(unittest.TestCase):
    """The load-time decision that gives `machines/pet.toml`'s `boot` a policy.

    No model needed, which is the entire reason it was lifted out of
    `DuckSim.__init__`: every test in this file builds `DuckSim.__new__` and
    never runs `__init__`, so the gate itself was unreachable from the suite —
    and a regression in it is invisible from outside as well, because a
    missing kick policy is SAFE. `trigger_behavior` prints and returns, the
    pocket guard is never satisfied, `boot` becomes a 4.4 s no-op, and the
    play loop goes on looking like it works. All four cells are pinned.
    """

    def test_a_scene_with_a_ball_gets_the_kick_policies(self):
        from microduck_mcp.sim_server import kick_policies_apply
        self.assertTrue(kick_policies_apply("desktop", True))
        self.assertTrue(kick_policies_apply("ball", True))

    def test_a_scene_with_no_ball_does_not(self):
        from microduck_mcp.sim_server import kick_policies_apply
        self.assertFalse(kick_policies_apply("desktop", False))

    def test_plain_never_does_however_the_model_answers(self):
        # `plain` is the scene people start for a walk test; not paying for
        # two ONNX loads is the whole reason it exists.
        from microduck_mcp.sim_server import kick_policies_apply
        self.assertFalse(kick_policies_apply("plain", True))
        self.assertFalse(kick_policies_apply("plain", False))


# ------------------------------------------------------------ the pick-up

def _carry_steps(sim, n):
    """Run the sim loop's carry half: advance the hand, then step.

    The order matters and is the run_loop's own: `pet_carry_tick` writes
    mocap_pos BEFORE the physics steps, so the pose the solver sees this tick
    is the pose the human asked for this tick.
    """
    import mujoco
    from microduck_mcp.sim_server import DECIMATION, TIMESTEP
    for _ in range(n):
        sim.pet_carry_tick()
        for _ in range(DECIMATION):
            mujoco.mj_step(sim.model, sim.data)
        sim.sim_time += DECIMATION * TIMESTEP


@needs_model
class Carry(unittest.TestCase):
    """The weld, against the real compiled scene.

    Everything here is physics rather than arithmetic, and it is the one part
    of this lane that could not be settled by reading: `eq_data`'s compile-time
    contents, whether a welded body actually follows a mocap hand, and whether
    the duck comes down again when the hand opens.
    """

    def setUp(self):
        import mujoco
        self.sim = _pet_sim(_MODEL, mujoco.MjData(_MODEL))
        _reset_pose(self.sim)

    def trunk_z(self):
        return float(self.sim.data.qpos[self.sim.qpos_adr + 2])

    def hand(self):
        return np.array(self.sim.data.mocap_pos[self.sim._pet_hand_mocap])

    def test_the_compiled_weld_starts_inactive_and_wrong(self):
        # This test exists to document WHY `_pet_carry_grab` rewrites
        # eq_data. MuJoCo bakes a weld's relpose from qpos0 at compile time,
        # and qpos0 here is a hand parked at z=3 against a duck at z=0.12 — so
        # activating the shipped weld unedited would drive the duck 2.88 m
        # straight down, through the Dock, on the first step of every lift.
        # A FRESH compile, deliberately: `_MODEL` is shared across this file
        # and every grab above rewrites its eq_data, which is the whole point.
        # What is being pinned here is what the SCENE ships.
        model, _ = _desktop_model()
        eid = self.sim._pet_carry_eq
        self.assertGreaterEqual(eid, 0, "the scene has no pet_carry weld")
        self.assertFalse(bool(model.eq_active0[eid]))
        self.assertAlmostEqual(float(model.eq_data[eid][5]), -2.88, places=2)

    def test_the_hand_is_invisible_and_collides_with_nothing(self):
        # It is not a thing in the world, it is where the fingers are. Group 4
        # keeps it out of the picture; contype/conaffinity 0 keeps the duck
        # from ever bumping into the hand that is holding it.
        import mujoco
        gid = mujoco.mj_name2id(_MODEL, mujoco.mjtObj.mjOBJ_GEOM,
                                PET_HAND_BODY)
        self.assertGreaterEqual(gid, 0)
        self.assertEqual(int(_MODEL.geom_group[gid]), 4)
        self.assertEqual(int(_MODEL.geom_contype[gid]), 0)
        self.assertEqual(int(_MODEL.geom_conaffinity[gid]), 0)

    def test_the_hand_is_not_part_of_the_duck(self):
        # `_pet_bbox` over the duck mask is what the overlay hit-tests, so a
        # hand folded into it would be a duck you could grab from the parking
        # spot at z=3. PET_PROP_GEOMS is what keeps it out.
        import mujoco
        gid = mujoco.mj_name2id(_MODEL, mujoco.mjtObj.mjOBJ_GEOM,
                                PET_HAND_BODY)
        self.assertFalse(bool(self.sim._pet_duck_geom[gid]))
        self.assertFalse(bool(self.sim._pet_ball_geom[gid]))

    def test_a_grab_lifts_the_duck_and_a_release_drops_it(self):
        start = self.sim.handle({"cmd": "pet_carry", "action": "start"})
        self.assertTrue(start["ok"])
        self.assertTrue(start["carried"])
        token = start["token"]
        # ...and the hand landed on the trunk rather than 2.88 m away from it.
        self.assertAlmostEqual(self.hand()[2], self.trunk_z(), places=3)
        # Raise it half a metre and keep the hand alive while it climbs.
        for _ in range(20):
            self.sim.handle({"cmd": "pet_carry", "action": "move",
                             "token": token, "x_m": 0.0, "z_m": 0.50})
            _carry_steps(self.sim, 10)
        self.assertGreater(self.hand()[2], 0.45, "the hand never got there")
        self.assertAlmostEqual(self.trunk_z(), self.hand()[2], delta=0.08,
                               msg="the duck did not come with it")
        # Let go, and gravity does the rest — no animation anywhere in this.
        end = self.sim.handle({"cmd": "pet_carry", "action": "end",
                               "token": token})
        self.assertTrue(end["ok"])
        self.assertFalse(end["carried"])
        self.assertEqual(len(end["released_vel_mps"]), 3)
        _carry_steps(self.sim, 300)
        self.assertLess(self.trunk_z(), 0.20, "it stayed up in the air")

    def test_the_frame_follows_the_duck_up_but_not_off_a_stumble(self):
        # The whole illusion is that the Dock's edge is pinned. A camera that
        # followed every wobble would turn a stumble into a camera move.
        self.sim.data.qpos[self.sim.qpos_adr + 2] = 0.15
        self.assertEqual(self.sim._pet_lift_m(), 0.0)
        self.sim.data.qpos[self.sim.qpos_adr + 2] = 0.45
        self.assertAlmostEqual(self.sim._pet_lift_m(),
                               0.45 - PET_LIFT_TRIGGER_M, places=6)
        # ...and it is reported where the app reads it.
        state = self.sim._handle_pet_state()
        self.assertAlmostEqual(state["screen"]["frame_floor_z_m"],
                               0.45 - PET_LIFT_TRIGGER_M, places=6)

    def test_the_frame_edges_travel_with_the_lift(self):
        # `ball.in_frame` is decided against these, so a lifted frame that
        # forgot to move them would say a ball at the duck's feet was still
        # in the picture while the duck dangled half a metre above it.
        flat = self.sim._pet_frame_z_span()
        self.sim.data.qpos[self.sim.qpos_adr + 2] = 0.50
        lifted = self.sim._pet_frame_z_span()
        rise = 0.50 - PET_LIFT_TRIGGER_M
        self.assertAlmostEqual(lifted[0] - flat[0], rise, places=6)
        self.assertAlmostEqual(lifted[1] - flat[1], rise, places=6)

    def test_the_point_you_grabbed_is_the_point_that_follows_the_pointer(self):
        # The weld holds the duck by its TRUNK ORIGIN (zero relpos, the hand
        # IS the trunk) but the press lands wherever on the animal the human
        # aimed — `pet_app.hit_rect_pt`'s box is the whole silhouette, and
        # roughly half of all presses are below trunk height. Driving the hand
        # at the raw cursor therefore SNAPPED the trunk under the pointer:
        # measured, a press on the head (z 0.25 against a trunk at 0.12)
        # yanked the duck 0.13 m upwards on the spot.
        trunk0 = self.trunk_z()
        start = self.sim.handle({"cmd": "pet_carry", "action": "start",
                                 "x_m": 0.0, "z_m": 0.25})
        self.assertTrue(start["ok"])
        # A grab moves NOTHING — it closes a hand where the duck already is.
        self.assertAlmostEqual(start["target_m"][1], trunk0, places=3)
        _carry_steps(self.sim, 5)
        self.assertAlmostEqual(self.trunk_z(), trunk0, delta=0.02,
                               msg="the grab itself moved the duck")
        # Now lift the POINTER by 0.20 m: the head, not the trunk, is what
        # rides with it, so the trunk goes up by the same 0.20 and no more.
        self.sim.handle({"cmd": "pet_carry", "action": "move",
                         "token": start["token"], "x_m": 0.0, "z_m": 0.45})
        self.assertAlmostEqual(self.sim._pet_carry["target"][2],
                               trunk0 + 0.20, places=3)

    def test_a_duck_grabbed_by_the_feet_is_not_pressed_into_the_dock(self):
        # The other half of the same bug, and the nastier one: a press near
        # the floor line clamps to PET_CARRY_MIN_Z_M and drove the trunk 6.6 cm
        # BELOW standing, penetrating the floor plane against the weld and
        # scraping the duck along inside the Dock when the hand moved sideways.
        trunk0 = self.trunk_z()
        start = self.sim.handle({"cmd": "pet_carry", "action": "start",
                                 "x_m": 0.0, "z_m": 0.02})
        self.sim.handle({"cmd": "pet_carry", "action": "move",
                         "token": start["token"], "x_m": 0.0, "z_m": 0.02})
        _carry_steps(self.sim, 10)
        self.assertGreater(self.trunk_z(), trunk0 - 0.01,
                           "the duck was pressed down through the Dock")

    def test_a_grab_that_named_no_point_still_behaves_the_way_it_always_did(self):
        # The CLI and every older caller send `start` with no coordinates at
        # all. There is no grab point to speak of then, so the offset is zero
        # and the hand is the trunk — byte-identical to before the offset
        # existed.
        start = self.sim.handle({"cmd": "pet_carry", "action": "start"})
        self.assertEqual(self.sim._pet_carry["grab"], (0.0, 0.0))
        self.sim.handle({"cmd": "pet_carry", "action": "move",
                         "token": start["token"], "x_m": 0.3, "z_m": 0.40})
        self.assertAlmostEqual(self.sim._pet_carry["target"][0], 0.3, places=6)
        self.assertAlmostEqual(self.sim._pet_carry["target"][2], 0.40, places=6)

    def test_the_hand_cannot_carry_the_duck_through_a_wall(self):
        # The walls stop a duck that walks into one; nothing stops a hand, so
        # the hand is clamped instead — and a little inside, because the duck
        # hangs below it and swings.
        start = self.sim.handle({"cmd": "pet_carry", "action": "start"})
        wall = self.sim._pet_wall_x()
        for _ in range(30):
            self.sim.handle({"cmd": "pet_carry", "action": "move",
                             "token": start["token"], "x_m": 9.0, "z_m": 0.30})
            _carry_steps(self.sim, 10)
        self.assertLess(self.hand()[0], wall)
        self.assertGreater(self.hand()[0], 0.5, "it never went anywhere")

    def test_the_ceiling_and_the_floor_are_clamped_too(self):
        start = self.sim.handle({"cmd": "pet_carry", "action": "start"})
        high = self.sim.handle({"cmd": "pet_carry", "action": "move",
                                "token": start["token"],
                                "x_m": 0.0, "z_m": 99.0})
        self.assertAlmostEqual(high["target_m"][1],
                               self.sim.pet["carry_max_z_m"], places=6)
        low = self.sim.handle({"cmd": "pet_carry", "action": "move",
                               "token": start["token"],
                               "x_m": 0.0, "z_m": -5.0})
        self.assertAlmostEqual(low["target_m"][1], PET_CARRY_MIN_Z_M, places=6)

    def test_the_hand_never_teleports_however_far_the_cursor_jumped(self):
        # A cursor can cross a screen between two samples. A welded body that
        # went with it in one step is a solver explosion, not a pick-up.
        start = self.sim.handle({"cmd": "pet_carry", "action": "start"})
        self.sim.handle({"cmd": "pet_carry", "action": "move",
                         "token": start["token"], "x_m": 1.0, "z_m": 0.55})
        before = self.hand().copy()
        _carry_steps(self.sim, 1)
        moved = float(np.linalg.norm(self.hand() - before))
        from microduck_mcp.sim_server import DECIMATION, TIMESTEP
        cap = self.sim.pet["carry_hand_speed_mps"] * DECIMATION * TIMESTEP
        self.assertLessEqual(moved, cap + 1e-9)

    def test_the_hand_lets_go_when_nobody_is_holding_it(self):
        # The deadman. An overlay that crashed mid-lift must not leave the
        # duck hanging in the air for the rest of the session.
        start = self.sim.handle({"cmd": "pet_carry", "action": "start"})
        self.assertTrue(start["carried"])
        self.sim.sim_time += PET_CARRY_TIMEOUT_S + 0.1
        self.sim.pet_carry_tick()
        self.assertIsNone(self.sim._pet_carry)
        self.assertFalse(self.sim._handle_pet_state()["carry"]["carried"])
        self.assertEqual(bool(self.sim.data.eq_active[self.sim._pet_carry_eq]),
                         False)

    def test_a_stale_token_cannot_release_a_fresh_grab(self):
        # The reason the DAEMON mints the token: a `move` or an `end` from a
        # gesture the window server abandoned (a Space switch, Mission
        # Control) must not release the grab the human started afterwards.
        first = self.sim.handle({"cmd": "pet_carry", "action": "start"})
        self.sim.handle({"cmd": "pet_carry", "action": "end",
                         "token": first["token"]})
        second = self.sim.handle({"cmd": "pet_carry", "action": "start"})
        stale = self.sim.handle({"cmd": "pet_carry", "action": "end",
                                 "token": first["token"]})
        self.assertFalse(stale["ok"])
        self.assertTrue(stale["conflict"])
        self.assertIn("token expired", stale["error"])
        self.assertTrue(self.sim._handle_pet_state()["carry"]["carried"])
        self.assertEqual(self.sim._pet_carry["token"], second["token"])

    def test_a_second_hand_is_refused_rather_than_sharing(self):
        self.sim.handle({"cmd": "pet_carry", "action": "start"})
        again = self.sim.handle({"cmd": "pet_carry", "action": "start"})
        self.assertFalse(again["ok"])
        self.assertTrue(again["conflict"])

    def test_a_verb_nobody_has_heard_of_names_the_roster(self):
        resp = self.sim.handle({"cmd": "pet_carry", "action": "juggle"})
        self.assertFalse(resp["ok"])
        for verb in ("start", "move", "end"):
            self.assertIn(verb, resp["error"])

    def test_a_malformed_position_never_moves_the_hand(self):
        start = self.sim.handle({"cmd": "pet_carry", "action": "start"})
        before = self.hand().copy()
        bad = self.sim.handle({"cmd": "pet_carry", "action": "move",
                               "token": start["token"],
                               "x_m": "left", "z_m": 0.3})
        self.assertFalse(bad["ok"])
        self.assertIn("x_m and z_m", bad["error"])
        _carry_steps(self.sim, 5)
        self.assertTrue(np.allclose(self.hand(), before, atol=1e-9))

    def test_the_release_hands_over_the_hands_own_velocity(self):
        # A duck let go mid-swing flies; a duck set down gently does not. The
        # velocity is the HAND's, measured off its own track, and it lands in
        # qvel like every other gesture in this app.
        start = self.sim.handle({"cmd": "pet_carry", "action": "start"})
        self.sim.handle({"cmd": "pet_carry", "action": "move",
                         "token": start["token"], "x_m": 1.0, "z_m": 0.40})
        # Released MID-SWEEP, not after the hand arrived: a hand that reached
        # where it was going and sat there is standing still, and the test
        # below (`..._puts_the_duck_down_rather_than_throwing_it`) is the one
        # about that.
        _carry_steps(self.sim, 8)
        self.assertLess(self.hand()[0], 0.9, "the hand had already arrived")
        end = self.sim.handle({"cmd": "pet_carry", "action": "end",
                               "token": start["token"]})
        vx = end["released_vel_mps"][0]
        self.assertGreater(vx, 0.1, "a hand sweeping +x let go with nothing")
        self.assertLessEqual(abs(vx), 2.0)   # PET_PUSH_MAX, like every shove
        self.assertAlmostEqual(
            float(self.sim.data.qvel[self.sim.qvel_adr]), vx, places=6)

    def test_a_still_hand_puts_the_duck_down_rather_than_throwing_it(self):
        start = self.sim.handle({"cmd": "pet_carry", "action": "start"})
        for _ in range(10):
            self.sim.handle({"cmd": "pet_carry", "action": "move",
                             "token": start["token"], "x_m": 0.0, "z_m": 0.12})
            _carry_steps(self.sim, 5)
        end = self.sim.handle({"cmd": "pet_carry", "action": "end",
                               "token": start["token"]})
        self.assertLess(max(abs(v) for v in end["released_vel_mps"]), 0.05)

    def test_the_digest_says_it_is_being_held_and_for_how_long(self):
        self.sim.machine = None
        self.sim.policy.head_offset = np.zeros(4, dtype=np.float32)
        self.sim._ball_seen = {"visible": False, "distance_m": None,
                               "ground_distance_m": None, "bearing_deg": None,
                               "elevation_deg": None, "est_forward_m": None,
                               "est_left_m": None, "speed_mps": None}
        self.sim._ball_seen_t = 0.0
        from microduck_mcp.sim_server import GOAL_NOT_SEEN
        self.sim._goal_seen = dict(GOAL_NOT_SEEN)
        self.sim._goal_seen_t = 0.0
        self.sim._goal_fix = None
        self.sim._goal_azimuth_w = None
        self.sim.referee = None
        self.assertFalse(self.sim._machine_digest()["carried"])
        start = self.sim.handle({"cmd": "pet_carry", "action": "start"})
        self.sim.sim_time += 2.0
        d = self.sim._machine_digest()
        self.assertTrue(d["carried"])
        self.assertAlmostEqual(d["carried_s"], 2.0, places=3)
        self.sim.handle({"cmd": "pet_carry", "action": "end",
                         "token": start["token"]})
        self.assertFalse(self.sim._machine_digest()["carried"])
        self.assertEqual(self.sim._machine_digest()["carried_s"], 0.0)

    def test_a_reset_puts_the_duck_down(self):
        # Without the release, `reset` rewinds qpos while the weld cheerfully
        # drags the duck straight back to wherever the hand was left — a reset
        # that looks like it failed. Everything stubbed below is the half of
        # `_handle_reset` this fixture has no policy for; the weld is the half
        # under test.
        sim = self.sim
        start = sim.handle({"cmd": "pet_carry", "action": "start"})
        _carry_steps(sim, 5)
        self.assertTrue(sim._handle_pet_state()["carry"]["carried"])
        sim._qpos0 = sim.data.qpos.copy()
        _stub_for_reset(sim)
        sim._handle_reset()
        self.assertIsNone(sim._pet_carry)
        self.assertFalse(bool(sim.data.eq_active[sim._pet_carry_eq]))
        self.assertFalse(sim._handle_pet_state()["carry"]["carried"])
        # ...and the release velocity did not survive into the new episode.
        self.assertTrue(np.allclose(sim.data.qvel, 0.0))
        self.assertIsNotNone(start["token"])

    def test_a_reset_drops_the_pointer_but_keeps_the_tally(self):
        # The REAL `_handle_reset`, which is the whole point of this test
        # living here beside a compiled model rather than beside the cursor
        # arithmetic: the clock rewinds on reset, so a cursor sample stamped
        # at t=41 is future-dated against a sim back at 0, reads as age -41 s,
        # and every "the hand has gone" guard in machines/pet.toml stops being
        # able to fire. `count` is a session tally of how often this duck has
        # been petted, not episode state, and survives on purpose.
        #
        # It is asserted against the handler and not against three lines
        # copied out of it: with those lines deleted from `_handle_reset` the
        # whole suite passed, which is what a re-implemented reset buys you.
        sim = self.sim
        sim.emotes = None       # the touch reply needs an engine or a refusal
        sim.handle({"cmd": "pet_sense", "x_m": 0.4, "z_m": 0.1})
        sim.handle({"cmd": "pet_touch"})
        self.assertEqual(sim._pet_touch_state()["count"], 1)
        sim.sim_time = 41.0
        _stub_for_reset(sim)
        sim._handle_reset()
        self.assertEqual(sim.sim_time, 0.0)
        self.assertIsNone(sim._pet_cursor)
        self.assertEqual(sim._pet_cursor_state()["age_s"], 999.0)
        self.assertFalse(sim._pet_touch_state()["petted"])
        self.assertEqual(sim._pet_touch_state()["age_s"], 999.0)
        self.assertEqual(sim._pet_touch_state()["count"], 1)

    def test_a_seated_duck_is_stood_up_before_it_is_picked_up(self):
        # Welding a sitter and putting it down leaves `sit_mode` set with the
        # duck in the air, and nothing in the machine or the policy recovers
        # from that: the sit is a transition that has already been consumed.
        toggled = []
        self.sim.policy.sit_mode = True
        self.sim.policy.toggle_sit = lambda: (toggled.append(1),
                                              setattr(self.sim.policy,
                                                      "sit_mode", False))
        self.sim.policy.set_vel_cmd = lambda *a: None
        self.sim.get_state = lambda: {"ok": True}
        resp = self.sim.handle({"cmd": "pet_carry", "action": "start"})
        self.assertTrue(resp["ok"])
        self.assertFalse(self.sim.policy.sit_mode)
        self.assertEqual(len(toggled), 1)
        self.assertIn("stood it up", resp["note"])

    def test_the_move_channel_never_reaches_the_event_feed(self):
        # Twenty a second would flush the 500-entry feed in twenty-five
        # seconds — the same G18 the frame stream and the cursor already hit.
        # `start` and `end` are acts and DO belong there.
        sim = DuckSim.__new__(DuckSim)
        sim.events, sim._event_id = deque(maxlen=500), 0
        ok = {"ok": True, "note": ""}
        for _ in range(300):
            sim._log_event("web", {"cmd": "pet_carry", "action": "move"}, ok)
        self.assertEqual(len(sim.events), 0)
        sim._log_event("web", {"cmd": "pet_carry", "action": "start"}, ok)
        sim._log_event("web", {"cmd": "pet_carry", "action": "end"}, ok)
        self.assertEqual([e["cmd"] for e in sim.events],
                         ["pet_carry", "pet_carry"])


# ------------------------------------------------------------ walls & world

@needs_model
class Walls(unittest.TestCase):
    def setUp(self):
        import mujoco
        self.sim = _pet_sim(_MODEL, mujoco.MjData(_MODEL))

    def test_config_puts_the_walls_at_the_mapped_screen_edges(self):
        resp = self.sim._handle_pet_config({"screen_width_m": 2.0,
                                            "wall_margin_m": 0.25})
        self.assertTrue(resp["ok"])
        self.assertEqual(resp["config"]["walls_m"], [-0.75, 0.75])
        left = self.sim.data.mocap_pos[self.sim._pet_mocap["pet_wall_left"]]
        right = self.sim.data.mocap_pos[self.sim._pet_mocap["pet_wall_right"]]
        self.assertAlmostEqual(float(left[0]), -0.75)
        self.assertAlmostEqual(float(right[0]), 0.75)
        self.assertAlmostEqual(float(left[2]), PET_SOLID_HALF_H - PET_SOLID_SINK)

    def test_a_screen_width_in_points_maps_through_px_per_meter(self):
        resp = self.sim._handle_pet_config({"screen_width_px": 1728,
                                            "px_per_meter": 656})
        self.assertAlmostEqual(resp["config"]["screen_width_m"], 1728 / 656)

    def test_the_default_margin_keeps_the_whole_window_on_screen(self):
        resp = self.sim._handle_pet_config({"frame_px": 400, "px_per_meter": 800})
        self.assertAlmostEqual(resp["config"]["wall_margin_m"], 400 / (2 * 800))

    def test_the_walls_move_again_when_the_screen_changes(self):
        self.sim._handle_pet_config({"screen_width_m": 2.0, "wall_margin_m": 0.0})
        self.sim._handle_pet_config({"screen_width_m": 1.0, "wall_margin_m": 0.0})
        right = self.sim.data.mocap_pos[self.sim._pet_mocap["pet_wall_right"]]
        self.assertAlmostEqual(float(right[0]), 0.5)

    def test_nonsense_is_refused_with_the_range_that_was_missed(self):
        for bad in ({"px_per_meter": 0}, {"supersample": 9}, {"frame_px": 4},
                    {"screen_width_m": "wide"}):
            resp = self.sim._handle_pet_config(dict(bad))
            self.assertFalse(resp["ok"], bad)

    def test_a_pinned_margin_is_not_quietly_unpinned_by_the_next_config(self):
        # The app pins wall_margin_m to one duck-depth precisely because the
        # daemon's own default — half a window — puts the walls inside the arc
        # machines/pet.toml turns around in. A later partial config (the
        # screen changed; `duck pet config` with one flag; `duck pet config`
        # with none, which client.py documents as a read) must not walk them
        # back in.
        first = self.sim._handle_pet_config({"screen_width_px": 3456,
                                             "px_per_meter": 1312,
                                             "frame_px": 512, "supersample": 2,
                                             "wall_margin_m": 0.1227})
        walls = first["config"]["walls_m"]
        for later in ({}, {"corridor_m": 0.4}, {"screen_width_px": 3456}):
            with self.subTest(config=later):
                resp = self.sim._handle_pet_config(dict(later))
                self.assertTrue(resp["ok"], resp)
                self.assertAlmostEqual(resp["config"]["wall_margin_m"], 0.1227)
                self.assertEqual(resp["config"]["walls_m"], walls)

    def test_a_config_with_no_keys_is_a_read(self):
        # `duck pet config` on its own sends only {"cmd": "pet_config"}. It
        # must not move a wall, and it must not spend an mj_forward.
        before = np.array(self.sim.data.mocap_pos[
            self.sim._pet_mocap["pet_wall_right"]])
        resp = self.sim._handle_pet_config({})
        self.assertTrue(resp["ok"])
        self.assertIn("read only", resp["note"])
        self.assertEqual(resp["config"], self.sim._pet_config_state())
        np.testing.assert_allclose(
            self.sim.data.mocap_pos[self.sim._pet_mocap["pet_wall_right"]],
            before)

    def test_two_frame_sizes_keep_two_renderers_instead_of_thrashing(self):
        # A Renderer is a GL context: cheap to hold, tens of milliseconds to
        # build, and built ON THE SIM THREAD. Keyed on size alone, two
        # consumers asking for two sizes (the overlay, and one
        # `duck pet frame --size-px 300` beside it) tore each other's down
        # every frame — measured at 2.5x the frame cost.
        self.sim._handle_pet_frame({"size_px": 128, "supersample": 1})
        self.sim.sim_time += 0.02
        first = self.sim._pet_renderers[128]
        self.sim._handle_pet_frame({"size_px": 160, "supersample": 1})
        self.sim.sim_time += 0.02
        self.sim._handle_pet_frame({"size_px": 128, "supersample": 1})
        self.assertIn(160, self.sim._pet_renderers)
        self.assertIs(self.sim._pet_renderers[128], first,
                      "the 128 px renderer was rebuilt")
        for r in self.sim._pet_renderers.values():
            r.close()
        self.sim._pet_renderers.clear()

    def test_a_frame_bigger_than_the_framebuffer_is_refused(self):
        resp = self.sim._handle_pet_config({"frame_px": 512, "supersample": 3})
        self.assertFalse(resp["ok"])
        self.assertIn("framebuffer", resp["error"])

    def test_a_wall_actually_stops_a_duck(self):
        import mujoco
        gid = self.sim._pet_geoms["pet_wall_right"]
        self.sim._handle_pet_config({"screen_width_m": 0.4, "wall_margin_m": 0.0})
        _reset_pose(self.sim, x=0.10, z=0.30)
        # thrown at the wall hard enough that nothing but a contact stops it
        self.sim.data.qvel[self.sim.qvel_adr] = 1.5
        touched = _settle(self.sim, 400, watch_geom=gid)
        x = float(self.sim.data.qpos[self.sim.qpos_adr])
        self.assertTrue(touched, "the duck went through the wall")
        self.assertLess(x, 0.25, f"the duck ended up at x={x:.3f}, past the wall")


@needs_model
class World(unittest.TestCase):
    def setUp(self):
        import mujoco
        self.sim = _pet_sim(_MODEL, mujoco.MjData(_MODEL))

    def _pos(self, name):
        return np.array(self.sim.data.mocap_pos[self.sim._pet_mocap[name]])

    def _size(self, name):
        return np.array(self.sim.model.geom_size[self.sim._pet_geoms[name]])

    def test_a_rectangle_becomes_a_box_at_the_right_place_and_size(self):
        resp = self.sim._handle_pet_world(
            {"rects": [{"x": 0.20, "y": 0.10, "w": 0.60, "h": 0.04}]})
        self.assertTrue(resp["ok"], resp.get("error"))
        # bottom-left corner + size in, centre + half-extents out
        np.testing.assert_allclose(self._pos("pet_platform_00"), [0.50, 0.0, 0.12])
        np.testing.assert_allclose(self._size("pet_platform_00"), [0.30, 0.15, 0.02])
        self.assertEqual(resp["parked"], 11)

    def test_resizing_takes_the_derived_bounds_with_it(self):
        # geom_rbound and geom_aabb are computed from the size at compile time
        # and never revisited: stale ones make a grown box stop colliding.
        self.sim._handle_pet_world(
            {"rects": [{"x": 0.0, "y": 0.0, "w": 0.8, "h": 0.2, "depth_m": 0.5}]})
        gid = self.sim._pet_geoms["pet_platform_00"]
        size = self._size("pet_platform_00")
        self.assertAlmostEqual(float(self.sim.model.geom_rbound[gid]),
                               float(np.linalg.norm(size)), places=6)
        np.testing.assert_allclose(self.sim.model.geom_aabb[gid][3:], size)

    def test_the_list_is_the_whole_world_not_a_patch(self):
        self.sim._handle_pet_world({"rects": [{"x": 0, "y": 0, "w": .3, "h": .05},
                                              {"x": 1, "y": 0, "w": .3, "h": .05}]})
        resp = self.sim._handle_pet_world({"rects": [{"x": 0, "y": 0,
                                                      "w": .3, "h": .05}]})
        self.assertEqual(resp["parked"], 11)
        np.testing.assert_allclose(self._pos("pet_platform_01"), PET_PARK_POS)

    def test_clearing_parks_everything(self):
        self.sim._handle_pet_world({"rects": [{"x": 0, "y": 0, "w": .3, "h": .05}]})
        resp = self.sim._handle_pet_world({"rects": []})
        self.assertEqual(resp["parked"], len(PET_PLATFORM_GEOMS))
        for name in PET_PLATFORM_GEOMS:
            np.testing.assert_allclose(self._pos(name), PET_PARK_POS)

    def test_more_rectangles_than_boxes_is_an_honest_refusal(self):
        rects = [{"x": i * 0.1, "y": 0, "w": 0.05, "h": 0.05} for i in range(13)]
        resp = self.sim._handle_pet_world({"rects": rects})
        self.assertFalse(resp["ok"])
        self.assertIn("12", resp["error"])

    def test_a_malformed_rectangle_moves_nothing_it_named(self):
        for bad in ({"x": 0, "y": 0, "w": 0, "h": 0.1},
                    {"x": 0, "y": 0, "w": 0.1},
                    "not a rect"):
            resp = self.sim._handle_pet_world({"rects": [bad]})
            self.assertFalse(resp["ok"], bad)
        resp = self.sim._handle_pet_world({"rects": {"x": 0}})
        self.assertFalse(resp["ok"])

    def test_infinity_is_not_a_ledge(self):
        # `json.loads` accepts Infinity and 1e400, and inf > 0 is True, so the
        # positivity check alone let an infinite half-extent through into
        # geom_size, geom_rbound and geom_aabb — a box containing the world,
        # which the duck then sinks into. `reset` deliberately leaves the pet
        # world alone, so there was no way back short of another POST.
        gid = self.sim._pet_geoms["pet_platform_00"]
        for bad in ({"x": 0, "y": 0, "w": float("inf"), "h": 0.05},
                    {"x": float("nan"), "y": 0, "w": 0.3, "h": 0.05},
                    {"x": 0, "y": 0, "w": 0.3, "h": 0.05,
                     "depth_m": float("inf")},
                    {"x": 1e6, "y": 0, "w": 0.3, "h": 0.05},
                    {"x": 0, "y": 0, "w": 1e6, "h": 0.05}):
            with self.subTest(rect=bad):
                resp = self.sim._handle_pet_world({"rects": [dict(bad)]})
                self.assertFalse(resp["ok"], resp)
                self.assertTrue(np.isfinite(self.sim.model.geom_size[gid]).all())
                self.assertTrue(np.isfinite(self.sim.model.geom_rbound[gid]))
                np.testing.assert_allclose(self._pos("pet_platform_00"),
                                           PET_PARK_POS)

    def test_a_json_infinity_over_the_wire_is_refused_too(self):
        # The shape the reviewer actually reached it with: the literal a JSON
        # body carries, not a Python float.
        import json as _json
        rects = _json.loads('[{"x":0,"y":0,"w":1e400,"h":0.1}]')
        self.assertFalse(self.sim._handle_pet_world({"rects": rects})["ok"])

    def test_a_repositioned_ledge_catches_a_falling_duck(self):
        # The one that matters for the window-ledge lane, and the one that
        # fails silently if the boxes are plain world geoms: drop the duck
        # from above a ledge that only exists because pet_world put it there.
        gid = self.sim._pet_geoms["pet_platform_00"]
        self.sim._handle_pet_world(
            {"rects": [{"x": -0.30, "y": 0.0, "w": 0.60, "h": 0.10,
                        "depth_m": 0.6}]})
        _reset_pose(self.sim, z=0.60)
        touched = _settle(self.sim, 700, watch_geom=gid)
        z = float(self.sim.data.qpos[self.sim.qpos_adr + 2])
        self.assertTrue(touched, "the duck fell through the ledge")
        self.assertGreater(z, 0.08, f"the duck ended at z={z:.3f} — on the "
                                    "floor, not on a 0.10 m ledge")

    def test_a_parked_ledge_catches_nothing(self):
        gid = self.sim._pet_geoms["pet_platform_00"]
        self.sim._handle_pet_world({"rects": []})
        _reset_pose(self.sim, z=0.60)
        self.assertFalse(_settle(self.sim, 500, watch_geom=gid))


# --------------------------------------------------- state & screen mapping

class BareState(unittest.TestCase):
    """The arithmetic the window is placed from. No MuJoCo in the path."""

    def sim(self):
        import types
        sim = DuckSim.__new__(DuckSim)
        sim.pet = sim._pet_default_config()
        sim._mcp_intent_t = 0.0
        sim._mcp_intent_cmd = None
        sim.pet_scene = True
        sim._pet_seg_ok = True
        sim._pet_cursor = None
        sim._pet_touch = {"t": None, "count": 0, "ack_t": None}
        # Nobody holding it, and no weld to hold it with. `_pet_lift_m` reads
        # the trunk's z off qpos, so the freejoint has to be here even though
        # nothing in this class moves it.
        sim._pet_carry = None
        sim._pet_hand_mocap = -1
        sim.qpos_adr = sim.qvel_adr = 0
        sim.data = types.SimpleNamespace(qpos=np.zeros(7), qvel=np.zeros(6))
        sim.data.qpos[2] = 0.116
        return sim

    def test_the_defaults_put_a_180_px_duck_on_a_1728_pt_screen(self):
        sim = self.sim()
        cfg = sim._pet_config_state()
        self.assertAlmostEqual(cfg["duck_height_px"], 180.0, delta=1.0)
        self.assertAlmostEqual(cfg["screen_width_m"] * PET_PX_PER_METER, 1728.0,
                               places=3)
        self.assertEqual(cfg["render_px"],
                         PET_FRAME_PX * sim.pet["supersample"])
        self.assertAlmostEqual(cfg["view_height_m"],
                               PET_FRAME_PX / PET_PX_PER_METER)

    def test_walls_are_symmetric_and_never_cross(self):
        sim = self.sim()
        sim.pet["screen_width_m"] = 0.05      # narrower than the window
        sim.pet["wall_margin_m"] = 10.0       # and a nonsense margin
        lo, hi = sim._pet_config_state()["walls_m"]
        self.assertLess(lo, hi)

    def test_an_mcp_intent_lights_the_inhabited_lamp_and_it_goes_out(self):
        sim = self.sim()
        self.assertEqual(sim._pet_inhabited(), (False, None, None))
        sim._mcp_intent_t = time.time()
        sim._mcp_intent_cmd = "set_velocity"
        live, age, cmd = sim._pet_inhabited()
        self.assertTrue(live)
        self.assertLess(age, 1.0)
        self.assertEqual(cmd, "set_velocity")
        sim._mcp_intent_t = time.time() - PET_INHABITED_S - 1
        self.assertFalse(sim._pet_inhabited()[0])

    def test_a_scene_without_pet_geoms_refuses_by_name(self):
        sim = self.sim()
        sim.pet_scene = False
        sim.scene = "ball"
        for handler in ("_handle_pet_frame", "_handle_pet_config",
                        "_handle_pet_world"):
            resp = getattr(sim, handler)({})
            self.assertFalse(resp["ok"])
            self.assertIn("--scene desktop", resp["error"])
            self.assertIn("ball", resp["error"])


class Compose(unittest.TestCase):
    """The premultiplied downsample, on numbers small enough to check by hand."""

    def test_pass_through_at_supersample_one(self):
        rgb = np.zeros((4, 4, 3), np.uint8)
        alpha = np.full((4, 4), 255, np.uint8)
        out = DuckSim._pet_compose(rgb, alpha, 1)
        self.assertEqual(out.shape, (4, 4, 4))

    def test_a_half_covered_pixel_is_half_opaque_and_keeps_its_colour(self):
        # A 2x2 block: two red duck pixels, two magenta background pixels.
        # Premultiplied, the answer is red at alpha 128 — NOT a magenta-red
        # average, which is what a naive resize gives and what fringes.
        rgb = np.array([[[255, 0, 0], [255, 0, 255]],
                        [[255, 0, 0], [255, 0, 255]]], dtype=np.uint8)
        alpha = np.array([[255, 0], [255, 0]], dtype=np.uint8)
        out = DuckSim._pet_compose(rgb, alpha, 2)
        self.assertEqual(out.shape, (1, 1, 4))
        self.assertEqual(int(out[0, 0, 3]), 128)
        np.testing.assert_array_equal(out[0, 0, :3], [255, 0, 0])

    def test_a_fully_transparent_block_does_not_divide_by_zero(self):
        rgb = np.full((2, 2, 3), 200, np.uint8)
        alpha = np.zeros((2, 2), np.uint8)
        out = DuckSim._pet_compose(rgb, alpha, 2)
        self.assertEqual(int(out[0, 0, 3]), 0)
        self.assertTrue(np.isfinite(out).all())

    def test_an_odd_size_is_cropped_rather_than_smeared(self):
        rgb = np.zeros((5, 5, 3), np.uint8)
        alpha = np.full((5, 5), 255, np.uint8)
        self.assertEqual(DuckSim._pet_compose(rgb, alpha, 2).shape, (2, 2, 4))


class Push(unittest.TestCase):
    """A drag gesture is a real push: the same intent, one extra axis."""

    def sim(self, ball_qvel_adr=6):
        import types
        sim = DuckSim.__new__(DuckSim)
        sim.data = types.SimpleNamespace(qvel=np.zeros(16))
        sim.qvel_adr = 0
        sim.policy = types.SimpleNamespace(
            vel_min_x=-0.3, vel_max_x=0.3, vel_min_y=-0.2, vel_max_y=0.2,
            vel_max_ang=1.5, ball_qvel_adr=ball_qvel_adr)
        return sim

    def test_a_horizontal_shove_lands_in_qvel(self):
        sim = self.sim()
        resp = sim.handle({"cmd": "push", "magnitude": 1.2, "angle_deg": 0.0})
        self.assertTrue(resp["ok"])
        self.assertAlmostEqual(sim.data.qvel[0], 1.2, places=6)
        self.assertAlmostEqual(sim.data.qvel[1], 0.0, places=6)

    def test_vz_is_optional_and_clamped(self):
        sim = self.sim()
        sim.data.qvel[2] = 0.42
        sim.handle({"cmd": "push", "magnitude": 0.1, "angle_deg": 0.0})
        self.assertAlmostEqual(sim.data.qvel[2], 0.42,
                               msg="a push without vz must not touch z")
        resp = sim.handle({"cmd": "push", "magnitude": 0.1, "angle_deg": 0.0,
                           "vz": 99.0})
        self.assertAlmostEqual(sim.data.qvel[2], 2.0)
        self.assertAlmostEqual(resp["pushed"]["vz"], 2.0)

    def test_a_drag_maps_the_way_the_webui_says_it_does(self):
        # Mirrors webui._pet_push, which is the only place the gesture is
        # turned into physics: +dx is screen-right (world +x), +dy is screen
        # DOWN (world -z).
        from microduck_mcp.webui import PET_DRAG_GAIN, PET_PUSH_MAX
        ppm = PET_PX_PER_METER
        for dx_px, dy_px in ((120, -60), (-90, 0), (0, 40)):
            vx = max(-PET_PUSH_MAX, min(PET_PUSH_MAX, dx_px / ppm * PET_DRAG_GAIN))
            vz = max(-PET_PUSH_MAX, min(PET_PUSH_MAX, -dy_px / ppm * PET_DRAG_GAIN))
            sim = self.sim()
            sim.handle({"cmd": "push", "magnitude": abs(vx),
                        "angle_deg": 0.0 if vx >= 0 else 180.0, "vz": vz})
            self.assertAlmostEqual(sim.data.qvel[0], vx, places=5)
            self.assertAlmostEqual(sim.data.qvel[2], vz, places=5)
            self.assertEqual(math.copysign(1, dx_px or 1),
                             math.copysign(1, sim.data.qvel[0] or 1))

    def test_a_ball_shove_moves_the_ball_and_not_the_duck(self):
        # The whole of the two-target design, in one assertion: same intent,
        # same gain, same clamp, a different six floats of qvel. If this ever
        # writes both, a poke at the toy staggers the animal.
        sim = self.sim()
        resp = sim.handle({"cmd": "push", "magnitude": 0.45, "angle_deg": 0.0,
                           "vz": 0.2, "target": "ball"})
        self.assertTrue(resp["ok"])
        self.assertEqual(resp["target"], "ball")
        self.assertEqual(resp["pushed"]["target"], "ball")
        self.assertAlmostEqual(sim.data.qvel[6], 0.45, places=6)
        self.assertAlmostEqual(sim.data.qvel[8], 0.2, places=6)
        np.testing.assert_allclose(sim.data.qvel[0:6], 0.0)

    def test_the_default_target_is_still_the_duck(self):
        # Every caller written before the toy existed sends no target at all.
        sim = self.sim()
        resp = sim.handle({"cmd": "push", "magnitude": 1.0, "angle_deg": 0.0})
        self.assertEqual(resp["target"], "duck")
        self.assertAlmostEqual(sim.data.qvel[0], 1.0, places=6)
        np.testing.assert_allclose(sim.data.qvel[6:9], 0.0)

    def test_a_ball_target_on_a_ball_less_scene_is_an_honest_refusal(self):
        # Refused, not silently redirected: a shove that moved the wrong
        # thing is worse than one that did nothing and said why.
        sim = self.sim(ball_qvel_adr=None)
        resp = sim.handle({"cmd": "push", "magnitude": 1.0, "angle_deg": 0.0,
                           "target": "ball"})
        self.assertFalse(resp["ok"])
        self.assertIn("no ball", resp["error"])
        np.testing.assert_allclose(sim.data.qvel, 0.0)

    def test_an_unknown_target_names_the_roster(self):
        sim = self.sim()
        resp = sim.handle({"cmd": "push", "magnitude": 1.0, "target": "dock"})
        self.assertFalse(resp["ok"])
        self.assertIn("duck", resp["error"])
        self.assertIn("ball", resp["error"])
        np.testing.assert_allclose(sim.data.qvel, 0.0)


class Human(unittest.TestCase):
    """`/pet/sense` and `/pet/touch` — the two things only the overlay can see.

    No MuJoCo in the path: a cursor is arithmetic against the duck's own x,
    and a pet is a clock and an emote. The one thing worth proving over and
    over below is the NEGATIVE: none of it reaches `qvel`.
    """

    def sim(self, x=0.0, emotes=True):
        import types
        from microduck_mcp.emote import EmoteLibrary
        sim = DuckSim.__new__(DuckSim)
        sim.pet = sim._pet_default_config()
        sim.pet_scene = True
        sim.scene = sim.scene_key = "desktop"
        sim.sim_time = 0.0
        sim.qpos_adr, sim.qvel_adr = 0, 0
        sim.data = types.SimpleNamespace(qpos=np.zeros(7), qvel=np.zeros(6))
        sim.data.qpos[0] = x
        sim._pet_seg_ok = True
        sim._pet_cursor = None
        sim._pet_touch = {"t": None, "count": 0, "ack_t": None}
        # No weld in this stub at all: the ids are -1, which is exactly what a
        # daemon on an older `scene_desktop.xml` reports, and what
        # `_pet_can_carry` refuses on.
        sim._pet_carry = None
        sim._pet_carry_eq = sim._pet_hand_mocap = sim._pet_trunk_body = -1
        # Enough of the emote engine for start_emote to run for real: the
        # refusal paths are half of what `acknowledged` means.
        sim.emotes = EmoteLibrary(os.path.join(REPO, "emotes")) if emotes else None
        sim._emote = None
        sim.machine = None
        sim.voice = None
        sim.voice_bank = None
        sim._mouth_intent_t = 0.0
        sim.policy = _Policy()
        sim.policy.head_offset = np.zeros(4, dtype=np.float32)
        return sim

    # ----- the pointer -----

    def test_a_cursor_sample_comes_back_measured_against_the_duck(self):
        sim = self.sim(x=0.25)
        resp = sim.handle({"cmd": "pet_sense", "x_m": 0.42, "z_m": 0.18})
        c = resp["cursor"]
        self.assertTrue(resp["ok"])
        self.assertTrue(c["present"])
        self.assertAlmostEqual(c["dx_m"], 0.17, places=6)
        self.assertAlmostEqual(c["dist_m"], 0.17, places=6)
        self.assertEqual(c["age_s"], 0.0)
        self.assertTrue(c["near_floor"])       # 0.18 m is inside the 0.35 default
        self.assertIsNone(c["speed_mps"])      # one sample is not a speed

    def test_the_offset_is_signed_and_the_distance_is_not(self):
        # dx_m says which way to WALK; dist_m says whether to stop. A cursor
        # held directly overhead is 0 m away, because the duck cannot walk up.
        sim = self.sim(x=0.5)
        left = sim.handle({"cmd": "pet_sense", "x_m": 0.1, "z_m": 0.05})["cursor"]
        self.assertLess(left["dx_m"], 0.0)
        self.assertGreater(left["dist_m"], 0.0)
        overhead = sim.handle({"cmd": "pet_sense",
                               "x_m": 0.5, "z_m": 0.9})["cursor"]
        self.assertAlmostEqual(overhead["dist_m"], 0.0, places=6)
        self.assertFalse(overhead["near_floor"], "0.9 m up is not a visit")

    def test_a_cursor_goes_stale_without_pretending_it_never_existed(self):
        # The split the machine relies on: "the hand is gone" (age_s) and "the
        # hand is close" (dist_m) have to be answerable separately, or a duck
        # walks towards a pointer that stopped existing two minutes ago.
        sim = self.sim()
        sim.handle({"cmd": "pet_sense", "x_m": 0.4, "z_m": 0.1})
        sim.sim_time = 5.0
        c = sim._pet_cursor_state()
        self.assertFalse(c["present"])
        self.assertIsNone(c["dist_m"])
        self.assertEqual(c["age_s"], 5.0)

    def test_a_pointer_that_left_the_screen_is_dropped_at_once(self):
        sim = self.sim()
        sim.handle({"cmd": "pet_sense", "x_m": 0.4, "z_m": 0.1})
        resp = sim.handle({"cmd": "pet_sense", "present": False})
        self.assertTrue(resp["ok"])
        self.assertIsNone(sim._pet_cursor)
        self.assertEqual(resp["cursor"]["age_s"], 999.0)

    def test_speed_is_smoothed_rather_than_differenced(self):
        # A mouse moves in jerks and a guard cannot low-pass a number itself.
        sim = self.sim()
        sim.handle({"cmd": "pet_sense", "x_m": 0.0, "z_m": 0.1})
        sim.sim_time = 0.2
        one = sim.handle({"cmd": "pet_sense", "x_m": 0.1, "z_m": 0.1})["cursor"]
        self.assertAlmostEqual(one["speed_mps"], 0.5, places=6)
        sim.sim_time = 0.4
        two = sim.handle({"cmd": "pet_sense", "x_m": 0.1, "z_m": 0.1})["cursor"]
        self.assertGreater(two["speed_mps"], 0.0)      # the EMA still remembers
        self.assertLess(two["speed_mps"], 0.5)

    def test_speed_reads_the_senders_clock_when_one_is_stamped(self):
        # The overlay stamps every sample with its own monotonic clock
        # (`t_s`), and speed is computed from THAT: sense posts queue on the
        # sim thread behind ~40 ms renders, so arrival sim-times compress —
        # measured live, 0.375 m/s of real hand read as 2.23 m/s. Here two
        # samples land in the SAME sim tick but claim 0.4 s of sender time,
        # and the honest answer is the sender's.
        sim = self.sim()
        sim.handle({"cmd": "pet_sense", "x_m": 0.0, "z_m": 0.1, "t_s": 10.0})
        c = sim.handle({"cmd": "pet_sense", "x_m": 0.1, "z_m": 0.1,
                        "t_s": 10.4})["cursor"]
        self.assertAlmostEqual(c["speed_mps"], 0.25, places=6)

    def test_two_samples_from_the_same_instant_keep_the_known_speed(self):
        # The first live test's failure: a same-tick pair used to reset the
        # EMA to None, and `cursor.speed_mps > 0.05` spent the whole session
        # never once true against a plainly wiggling hand. No new time means
        # no new information — the speed CARRIES, it does not amnese.
        sim = self.sim()
        sim.handle({"cmd": "pet_sense", "x_m": 0.0, "z_m": 0.1, "t_s": 10.0})
        sim.handle({"cmd": "pet_sense", "x_m": 0.1, "z_m": 0.1, "t_s": 10.4})
        c = sim.handle({"cmd": "pet_sense", "x_m": 0.12, "z_m": 0.1,
                        "t_s": 10.4})["cursor"]
        self.assertAlmostEqual(c["speed_mps"], 0.25, places=6)

    def test_a_garbage_timestamp_falls_back_to_the_sim_clock(self):
        # `t_s` is advisory: a caller that stamps nonsense gets the old
        # arrival-time arithmetic, never an error and never a poisoned EMA.
        sim = self.sim()
        sim.handle({"cmd": "pet_sense", "x_m": 0.0, "z_m": 0.1,
                    "t_s": "yesterday"})
        sim.sim_time = 0.2
        c = sim.handle({"cmd": "pet_sense", "x_m": 0.1, "z_m": 0.1,
                        "t_s": float("nan")})["cursor"]
        self.assertAlmostEqual(c["speed_mps"], 0.5, places=6)

    def test_nonsense_coordinates_are_refused_by_name(self):
        sim = self.sim()
        for bad in ({"x_m": "left", "z_m": 0.1}, {"z_m": 0.1},
                    {"x_m": float("inf"), "z_m": 0.1},
                    {"x_m": 0.1, "z_m": float("nan")}):
            with self.subTest(body=bad):
                resp = sim.handle({"cmd": "pet_sense", **bad})
                self.assertFalse(resp["ok"])
                self.assertIn("must be numbers", resp["error"])
                self.assertIsNone(sim._pet_cursor)

    def test_the_cursor_channel_never_reaches_the_event_feed(self):
        # Five samples a second flush a 500-entry AX feed in under two
        # minutes; where a mouse happens to be is not an act. (G18 again.)
        #
        # Driven through `_log_event` rather than asserted as membership in a
        # constant: the constant is a list, and a list a filter no longer
        # consults is a list that still contains the right name. The second
        # half is the part membership can never prove — a `pet_touch` is a
        # HAND on the duck, it happens at human rates, and it does belong.
        sim = DuckSim.__new__(DuckSim)
        sim.events, sim._event_id = deque(maxlen=500), 0
        ok = {"ok": True}
        for _ in range(200):
            sim._log_event("web", {"cmd": "pet_sense", "x_m": 0.1}, ok)
        self.assertEqual(len(sim.events), 0)
        sim._log_event("web", {"cmd": "pet_touch", "kind": "pet"}, ok)
        self.assertEqual([e["cmd"] for e in sim.events], ["pet_touch"])

    # ----- the hand -----

    def test_a_pet_is_answered_once_and_then_held_in_peace(self):
        sim = self.sim()
        first = sim.handle({"cmd": "pet_touch", "kind": "pet"})
        self.assertTrue(first["acknowledged"], first)
        self.assertEqual(first["emote"], "nuzzle")
        self.assertEqual(first["sound"], "coo")
        self.assertEqual(first["count"], 1)
        # Inside the cooldown: no second gesture, but the tally and the clock
        # still move — being petted twice in a second IS two pets.
        sim.sim_time = 0.5
        sim._emote = None                       # the first nuzzle has finished
        second = sim.handle({"cmd": "pet_touch"})
        self.assertFalse(second["acknowledged"])
        self.assertIn("still enjoying", second["note"])
        self.assertEqual(second["count"], 2)
        self.assertEqual(sim._pet_touch_state()["age_s"], 0.0)
        self.assertTrue(sim._pet_touch_state()["petted"])

    def test_the_cooldown_ends_and_the_duck_answers_again(self):
        from microduck_mcp.sim_server import PET_TOUCH_ACK_COOLDOWN_S
        sim = self.sim()
        sim.handle({"cmd": "pet_touch"})
        sim.sim_time = PET_TOUCH_ACK_COOLDOWN_S + 0.1
        sim._emote = None
        self.assertTrue(sim.handle({"cmd": "pet_touch"})["acknowledged"])

    def test_a_pet_never_lands_in_qvel(self):
        # The whole point of splitting the gesture: a poke moves the duck, a
        # pet does not. If this ever fails the feature has become a shove.
        sim = self.sim()
        sim.handle({"cmd": "pet_touch", "x_m": 0.02, "z_m": 0.17,
                    "duration_s": 0.4, "travel_m": 0.09})
        np.testing.assert_array_equal(sim.data.qvel, np.zeros(6))

    def test_a_refused_gesture_is_reported_in_the_engines_own_words(self):
        # `start_emote(machine=False)`, so a pet arriving mid-gesture is
        # refused — and the reply must not claim a nuzzle that never played.
        sim = self.sim()
        sim.handle({"cmd": "pet_touch"})
        sim.sim_time = 3.0                      # past the cooldown...
        resp = sim.handle({"cmd": "pet_touch"})  # ...but still mid-nuzzle
        self.assertFalse(resp["acknowledged"])
        self.assertIn("mid-emote", resp["note"])
        self.assertEqual(resp["count"], 2)

    def test_a_daemon_with_no_emotes_says_so_instead_of_lying(self):
        sim = self.sim(emotes=False)
        resp = sim.handle({"cmd": "pet_touch"})
        self.assertTrue(resp["ok"])
        self.assertFalse(resp["acknowledged"])
        self.assertIn("emote directory", resp["note"])

    def test_an_unknown_touch_is_refused_with_the_roster(self):
        sim = self.sim()
        resp = sim.handle({"cmd": "pet_touch", "kind": "bite"})
        self.assertFalse(resp["ok"])
        self.assertIn("pet", resp["error"])
        self.assertEqual(sim._pet_touch["count"], 0)

    def test_a_malformed_measurement_does_not_become_a_pet(self):
        sim = self.sim()
        resp = sim.handle({"cmd": "pet_touch", "x_m": "beak"})
        self.assertFalse(resp["ok"])
        self.assertEqual(sim._pet_touch["count"], 0)

    # ----- what the machine reads, and what a reset does to it -----

    def test_the_sensed_paths_are_all_in_the_grammar(self):
        from microduck_mcp.machine import GUARD_PATHS, compile_guard
        for path in ("cursor.present", "cursor.x_m", "cursor.z_m",
                     "cursor.dx_m", "cursor.dist_m", "cursor.age_s",
                     "cursor.near_floor", "cursor.speed_mps",
                     "touch.petted", "touch.age_s", "touch.count",
                     "carried", "carried_s"):
            with self.subTest(path=path):
                self.assertIn(path, GUARD_PATHS)
                compile_guard(f"{path} != 0")

    def test_a_never_touched_duck_answers_honestly(self):
        sim = self.sim()
        self.assertEqual(sim._pet_touch_state(),
                         {"petted": False, "age_s": 999.0, "count": 0})
        self.assertEqual(sim._pet_cursor_state()["age_s"], 999.0)

    def test_a_dropped_sample_reads_as_gone_and_the_tally_is_not_episode_state(self):
        # The two facts `_handle_reset` leans on, on their own: a cleared
        # sample answers 999 s rather than a negative age, and `count` is a
        # session tally of how often this duck has been petted rather than
        # something the episode owns.
        #
        # THE RESET ITSELF is tested against the real handler, beside a
        # compiled model, in `Carry.test_a_reset_drops_the_pointer_but_keeps_
        # the_tally` — this stub has no policy to run one through, and a test
        # that re-implemented the handler's three lines here and asserted on
        # its own copy would pass with those lines deleted from the daemon.
        # (Measured: it did.)
        sim = self.sim()
        sim.handle({"cmd": "pet_sense", "x_m": 0.4, "z_m": 0.1})
        sim.handle({"cmd": "pet_touch"})
        sim._pet_cursor = None
        sim._pet_touch["t"] = None
        self.assertEqual(sim._pet_cursor_state()["age_s"], 999.0)
        self.assertEqual(sim._pet_touch_state()["count"], 1)
        self.assertFalse(sim._pet_touch_state()["petted"])

    def test_the_config_knob_moves_the_near_floor_line(self):
        sim = self.sim()
        sim.pet["cursor_floor_m"] = 0.10
        c = sim.handle({"cmd": "pet_sense", "x_m": 0.0, "z_m": 0.2})["cursor"]
        self.assertFalse(c["near_floor"])
        self.assertIn("cursor_floor_m", sim._pet_config_state())

    def test_neither_route_answers_on_a_daemon_that_is_not_the_pets(self):
        sim = self.sim()
        sim.pet_scene = False
        sim.scene = "ball"
        for cmd in ("pet_sense", "pet_touch", "pet_carry"):
            with self.subTest(cmd=cmd):
                resp = sim.handle({"cmd": cmd, "action": "start",
                                   "x_m": 0.0, "z_m": 0.0})
                self.assertFalse(resp["ok"])
                self.assertIn("--scene desktop", resp["error"])

    def test_a_pet_scene_with_no_weld_in_it_says_which_half_is_missing(self):
        # `pet_scene` is about walls; `_pet_can_carry` is about a weld, and an
        # older copy of scene_desktop.xml has the first and not the second.
        # The refusal has to name the one that is actually absent.
        sim = self.sim()
        resp = sim.handle({"cmd": "pet_carry", "action": "start"})
        self.assertFalse(resp["ok"])
        self.assertIn("pet_carry", resp["error"])
        self.assertNotIn("--scene desktop", resp["error"])


class _StubSim:
    """The only two things `webui.start_web` talks to: `submit` and `events`.

    Small enough to stand the real HTTP server up against, which is the point:
    every test above this line goes at the sim handlers directly or at
    `pet_mock`, and the routing, the header round trip, the body limits and
    the drag arithmetic in between were the untested half.
    """

    def __init__(self, pet_scene=True, has_ball=True):
        self.pet_scene = pet_scene
        self.has_ball = has_ball
        self.submits = []
        self.pushed = None
        self.events = []
        # Flipped by the one test about a stale carry token: the sim answers
        # `conflict` and the route has to turn that into a 409 rather than the
        # 400 every other refusal on this surface gets.
        self.conflict = False

    def submit(self, req):
        cmd = req["cmd"]
        self.submits.append(cmd)
        if cmd == "pet_state":
            return {"ok": True, "pet_scene": self.pet_scene, "base_x_m": 0.25,
                    "config": {"px_per_meter": PET_PX_PER_METER}}
        if cmd == "pet_frame":
            if not self.pet_scene:
                return {"ok": False, "error": "no pet geoms"}
            return {"ok": True, "png": b"not-really-a-png", "width": 8,
                    "height": 8, "base_x_m": 0.25}
        if cmd == "push":
            self.pushed = dict(req)
            if req.get("target") == "ball" and not self.has_ball:
                return {"ok": False, "error": "no ball in this scene to shove"}
            return {"ok": True, "target": req.get("target", "duck"),
                    "pushed": req}
        if cmd == "pet_carry" and self.conflict:
            return {"ok": False, "conflict": True, "error":
                    "not the current carry (token expired — the hand was "
                    "released)"}
        return {"ok": True, "cmd": cmd, "echo": req}


class WebRoutes(unittest.TestCase):
    """`webui.start_web`'s /pet/* surface, over a real socket."""

    def serve(self, pet_scene=True, has_ball=True):
        from microduck_mcp import webui
        sim = _StubSim(pet_scene=pet_scene, has_ball=has_ball)
        server = webui.start_web(sim, 0)
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        return sim, server.server_address[1]

    def conn(self, port):
        import http.client
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        self.addCleanup(c.close)
        return c

    def post(self, c, path, obj):
        c.request("POST", path, body=json.dumps(obj).encode(),
                  headers={"Content-Type": "application/json"})
        r = c.getresponse()
        return r.status, json.loads(r.read() or b"null")

    def test_the_frame_route_keeps_its_connection_and_carries_the_pose(self):
        # pet_feed holds ONE connection at 20 fps ("reconnecting per frame is
        # pure syscall tax"). Under the stdlib default of HTTP/1.0 every frame
        # was a fresh connect and a fresh handler thread contending with the
        # sim thread — and pet_mock, the stand-in the app is tested against,
        # was the only one of the two that got this right.
        _sim, port = self.serve()
        c = self.conn(port)
        for _ in range(2):
            c.request("GET", "/pet/frame?size_px=128")
            r = c.getresponse()
            body = r.read()
            self.assertEqual(r.status, 200)
            self.assertEqual(body, b"not-really-a-png")
            self.assertFalse(r.will_close, "the connection was not kept alive")
            pose = json.loads(r.getheader("X-Duck-Pet"))
            self.assertAlmostEqual(pose["base_x_m"], 0.25)
            self.assertNotIn("png", pose, "the bytes must not ride the header")
        self.assertIsNotNone(c.sock)

    def test_a_drag_in_metres_becomes_a_push_with_the_documented_signs(self):
        # +dx is screen right (world +x); +dy is screen DOWN (world -z).
        sim, port = self.serve()
        c = self.conn(port)
        from microduck_mcp.webui import PET_DRAG_GAIN
        status, body = self.post(c, "/pet/push", {"dx_m": -0.2, "dy_m": -0.05})
        self.assertEqual(status, 200, body)
        self.assertAlmostEqual(sim.pushed["magnitude"], 0.2 * PET_DRAG_GAIN)
        self.assertEqual(sim.pushed["angle_deg"], 180.0)
        self.assertAlmostEqual(sim.pushed["vz"], 0.05 * PET_DRAG_GAIN)

    def test_a_push_in_metres_does_not_spend_a_sim_tick_looking_up_a_scale(self):
        # Every gesture the shipped app sends is already in metres
        # (pet_map.drag_to_push / poke_to_push), and a `pet_state` submit is a
        # whole 50 Hz tick. The scene check behind it is answered once and
        # remembered — the scene cannot change without a restart.
        sim, port = self.serve()
        c = self.conn(port)
        self.post(c, "/pet/push", {"dx_m": 0.1, "dy_m": 0.0})
        sim.submits.clear()
        self.post(c, "/pet/push", {"dx_m": 0.1, "dy_m": 0.0})
        self.assertEqual(sim.submits, ["push"])

    def test_a_push_in_pixels_still_asks_for_the_scale(self):
        sim, port = self.serve()
        c = self.conn(port)
        status, _ = self.post(c, "/pet/push", {"dx_px": 200.0, "dy_px": 0.0})
        self.assertEqual(status, 200)
        self.assertIn("pet_state", sim.submits)
        from microduck_mcp.webui import PET_DRAG_GAIN, PET_PUSH_MAX
        want = min(PET_PUSH_MAX, 200.0 / PET_PX_PER_METER * PET_DRAG_GAIN)
        self.assertAlmostEqual(sim.pushed["magnitude"], want)

    def test_a_drag_can_be_aimed_at_the_ball_instead(self):
        # One field on the gesture the app already sends. The gain and the
        # sign convention are untouched, which is the design: a poke rolls
        # the toy and a flick throws it, at the same numbers that stagger the
        # duck, because the difference between them is 15 g against 700 and
        # that is the physics' business.
        sim, port = self.serve()
        c = self.conn(port)
        from microduck_mcp.webui import PET_DRAG_GAIN
        status, body = self.post(c, "/pet/push",
                                 {"dx_m": 0.075, "dy_m": 0.0, "target": "ball"})
        self.assertEqual(status, 200, body)
        self.assertEqual(sim.pushed["target"], "ball")
        self.assertAlmostEqual(sim.pushed["magnitude"], 0.075 * PET_DRAG_GAIN)
        self.assertEqual(body["target"], "ball")

    def test_a_push_with_no_target_is_still_a_push_at_the_duck(self):
        sim, port = self.serve()
        c = self.conn(port)
        self.post(c, "/pet/push", {"dx_m": 0.1, "dy_m": 0.0})
        self.assertEqual(sim.pushed["target"], "duck")

    def test_an_unknown_target_is_refused_without_spending_a_sim_tick(self):
        sim, port = self.serve()
        c = self.conn(port)
        self.post(c, "/pet/push", {"dx_m": 0.1, "dy_m": 0.0})   # warm the cache
        sim.submits.clear()
        status, body = self.post(c, "/pet/push",
                                 {"dx_m": 0.1, "dy_m": 0.0, "target": "dock"})
        self.assertEqual(status, 400)
        self.assertIn("duck, ball", body["error"])
        self.assertEqual(sim.submits, [], "a bad target cost the duck a tick")

    def test_a_ball_shove_on_a_scene_with_no_ball_is_a_400_not_a_500(self):
        # It is the caller naming a thing that is not there, which is the same
        # class of mistake as every other refusal on this surface — and a 500
        # would read as "the daemon broke", which it did not.
        _sim, port = self.serve(has_ball=False)
        c = self.conn(port)
        status, body = self.post(c, "/pet/push",
                                 {"dx_m": 0.1, "dy_m": 0.0, "target": "ball"})
        self.assertEqual(status, 400)
        self.assertIn("no ball", body["error"])

    def test_a_push_at_a_daemon_that_is_not_the_pets_is_refused(self):
        # pet_feed's default port is duck-sim's own, on purpose. A duck-pet
        # started while a live MCP session owns that port renders nothing —
        # and used to still be a remote shove button on that session's duck,
        # because `pet_state` answers on any scene by design.
        sim, port = self.serve(pet_scene=False)
        c = self.conn(port)
        status, body = self.post(c, "/pet/push", {"dx_m": 0.5, "dy_m": 0.0})
        self.assertEqual(status, 400)
        self.assertIn("--scene desktop", body["error"])
        self.assertIsNone(sim.pushed)
        self.assertNotIn("push", sim.submits)

    def test_a_bad_body_is_a_400_and_an_unknown_route_is_a_404(self):
        _sim, port = self.serve()
        c = self.conn(port)
        c.request("POST", "/pet/push", body=b"{not json",
                  headers={"Content-Type": "application/json"})
        r = c.getresponse()
        self.assertEqual(r.status, 400)
        self.assertIn("bad JSON", json.loads(r.read())["error"])
        status, body = self.post(c, "/pet/nope", {})
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "not found")

    def test_content_length_is_not_taken_on_trust(self):
        # An oversized body is refused without reading it, and a NEGATIVE one
        # is not "no body": `rfile.read(-1)` reads to EOF, which held a
        # handler thread on the sim's front door open for as long as the
        # caller liked.
        import socket as _socket
        _sim, port = self.serve()
        for length, want in ((999999, 413), (-1, 200)):
            with self.subTest(content_length=length):
                s = _socket.create_connection(("127.0.0.1", port), timeout=5)
                self.addCleanup(s.close)
                s.sendall(f"POST /pet/config HTTP/1.1\r\nHost: x\r\n"
                          f"Content-Length: {length}\r\n\r\n".encode())
                s.settimeout(5)
                head = s.recv(64)
                self.assertIn(str(want).encode(), head.split(b"\r\n")[0])

    def test_the_human_routes_reach_their_commands(self):
        sim, port = self.serve()
        c = self.conn(port)
        status, body = self.post(c, "/pet/sense",
                                 {"x_m": 0.42, "z_m": 0.18, "present": True})
        self.assertEqual(status, 200, body)
        self.assertEqual(body["cmd"], "pet_sense")
        self.assertAlmostEqual(body["echo"]["x_m"], 0.42)
        status, body = self.post(c, "/pet/touch", {"kind": "pet"})
        self.assertEqual(status, 200, body)
        self.assertEqual(body["cmd"], "pet_touch")

    def test_a_cursor_that_is_not_numbers_never_costs_a_sim_tick(self):
        # An HTTP handler thread is free; a `sim.submit()` is a whole 50 Hz
        # tick of the duck's life. Spending one to be told a string is not a
        # float is the thing this validation exists to avoid, and at five
        # samples a second it would show.
        sim, port = self.serve()
        c = self.conn(port)
        self.post(c, "/pet/sense", {"x_m": 0.1, "z_m": 0.1})   # warm the cache
        sim.submits.clear()
        status, body = self.post(c, "/pet/sense", {"x_m": "left", "z_m": 0.1})
        self.assertEqual(status, 400)
        self.assertIn("must be numbers", body["error"])
        self.assertEqual(sim.submits, [])
        # ...but "the pointer left the screen" carries no coordinates at all
        # and must still get through.
        status, _ = self.post(c, "/pet/sense", {"present": False})
        self.assertEqual(status, 200)
        self.assertEqual(sim.submits, ["pet_sense"])

    def test_the_human_routes_at_a_daemon_that_is_not_the_pets_are_refused(self):
        # The exact twin of the push refusal above, and the reason is sharper
        # here: the pet's port is duck-sim's own default, so an unguarded
        # /pet/touch is a remote *petting* button on a live MCP session's
        # duck — a stranger's overlay making that robot stop and coo, five
        # times a second, from a route nobody had to authenticate to.
        sim, port = self.serve(pet_scene=False)
        c = self.conn(port)
        for path, body in (("/pet/sense", {"x_m": 0.1, "z_m": 0.1}),
                           ("/pet/touch", {"kind": "pet"}),
                           # ...and the sharpest of the three: an unguarded
                           # /pet/carry is a remote PICK-UP button, and a duck
                           # lifted off the floor by a stranger's window is a
                           # session whose robot simply stops working.
                           ("/pet/carry", {"action": "start"})):
            with self.subTest(path=path):
                status, resp = self.post(c, path, body)
                self.assertEqual(status, 400)
                self.assertIn("--scene desktop", resp["error"])
        self.assertNotIn("pet_sense", sim.submits)
        self.assertNotIn("pet_touch", sim.submits)
        self.assertNotIn("pet_carry", sim.submits)

    def test_the_carry_route_reaches_its_command(self):
        sim, port = self.serve()
        c = self.conn(port)
        status, body = self.post(c, "/pet/carry",
                                 {"action": "start", "x_m": 0.12, "z_m": 0.14})
        self.assertEqual(status, 200, body)
        self.assertEqual(body["cmd"], "pet_carry")
        self.assertEqual(body["echo"]["action"], "start")

    def test_a_carry_that_is_not_a_verb_never_costs_a_sim_tick(self):
        # Same argument as the cursor's: an HTTP handler thread is free and a
        # submit is a whole 50 Hz tick, and this channel runs at 20 Hz.
        sim, port = self.serve()
        c = self.conn(port)
        self.post(c, "/pet/carry", {"action": "start"})   # warm the cache
        sim.submits.clear()
        status, body = self.post(c, "/pet/carry", {"action": "juggle"})
        self.assertEqual(status, 400)
        self.assertIn("juggle", body["error"])
        status, body = self.post(c, "/pet/carry",
                                 {"action": "move", "x_m": "left", "z_m": 0.1})
        self.assertEqual(status, 400)
        self.assertIn("must be numbers", body["error"])
        self.assertEqual(sim.submits, [])

    def test_a_carry_conflict_is_a_409_and_not_a_400(self):
        # A stale token is neither a malformed request nor a broken daemon:
        # nothing changed, the hand it named is gone, and the right answer for
        # the app is to forget the gesture rather than retry it. `pet_feed`
        # clears its token on exactly this status.
        sim, port = self.serve()
        c = self.conn(port)
        sim.conflict = True
        status, body = self.post(c, "/pet/carry",
                                 {"action": "end", "token": "deadbeef"})
        self.assertEqual(status, 409, body)
        self.assertIn("token expired", body["error"])

    def test_the_config_and_world_routes_reach_their_commands(self):
        sim, port = self.serve()
        c = self.conn(port)
        status, body = self.post(c, "/pet/config", {"px_per_meter": 700})
        self.assertEqual(status, 200)
        self.assertEqual(body["echo"]["px_per_meter"], 700)
        self.assertEqual(body["cmd"], "pet_config")
        status, body = self.post(c, "/pet/world", {"rects": []})
        self.assertEqual(status, 200)
        self.assertEqual(body["cmd"], "pet_world")


if __name__ == "__main__":
    unittest.main()
