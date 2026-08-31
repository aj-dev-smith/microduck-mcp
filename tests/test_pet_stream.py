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

import numpy as np

from microduck_mcp.sim_server import (DUCK_HEIGHT_M, LOCAL_SCENES,
                                      PET_FRAME_PX, PET_INHABITED_S,
                                      PET_PARK_POS, PET_PLATFORM_GEOMS,
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
    body."""

    def __init__(self):
        self.sit_mode = False
        self.current_policy = "standing"
        self.behavior_mode = None
        self.vel_cmd = np.zeros(3, dtype=np.float32)

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
    sim.policy = _Policy()
    sim.scene, sim.scene_key = "desktop", "desktop"
    sim.sim_time = 0.0
    sim.machine = None
    sim.pet = sim._pet_default_config()
    sim._pet_geoms, sim._pet_mocap = {}, {}
    for name in PET_WALL_GEOMS + PET_RAIL_GEOMS + PET_PLATFORM_GEOMS:
        sim._pet_geoms[name] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        sim._pet_mocap[name] = int(model.body_mocapid[bid]) if bid >= 0 else -1
    sim.pet_scene = True
    pet_bodies = {int(model.geom_bodyid[g]) for g in sim._pet_geoms.values() if g >= 0}
    sim._pet_duck_geom = np.array(
        [int(model.geom_bodyid[g]) != 0 and int(model.geom_bodyid[g]) not in pet_bodies
         for g in range(model.ngeom)], dtype=bool)
    sim._pet_renderers = {}
    sim._pet_renderer = None
    sim._pet_renderer_px = 0
    sim._pet_option = None
    sim._pet_wall_pinned = False
    sim._pet_seg_ok = True
    sim._pet_cache = None
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
        sim = DuckSim.__new__(DuckSim)
        sim.pet = sim._pet_default_config()
        sim._mcp_intent_t = 0.0
        sim._mcp_intent_cmd = None
        sim.pet_scene = True
        sim._pet_seg_ok = True
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

    def sim(self):
        import types
        sim = DuckSim.__new__(DuckSim)
        sim.data = types.SimpleNamespace(qvel=np.zeros(10))
        sim.qvel_adr = 0
        sim.policy = types.SimpleNamespace(
            vel_min_x=-0.3, vel_max_x=0.3, vel_min_y=-0.2, vel_max_y=0.2,
            vel_max_ang=1.5)
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


class _StubSim:
    """The only two things `webui.start_web` talks to: `submit` and `events`.

    Small enough to stand the real HTTP server up against, which is the point:
    every test above this line goes at the sim handlers directly or at
    `pet_mock`, and the routing, the header round trip, the body limits and
    the drag arithmetic in between were the untested half.
    """

    def __init__(self, pet_scene=True):
        self.pet_scene = pet_scene
        self.submits = []
        self.pushed = None
        self.events = []

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
            return {"ok": True, "pushed": req}
        return {"ok": True, "cmd": cmd, "echo": req}


class WebRoutes(unittest.TestCase):
    """`webui.start_web`'s /pet/* surface, over a real socket."""

    def serve(self, pet_scene=True):
        from microduck_mcp import webui
        sim = _StubSim(pet_scene=pet_scene)
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
