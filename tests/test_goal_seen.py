"""Tests for the head-camera goal detector (fake mediad, part 2).

Offline: no MuJoCo model. Synthetic frames are ray-cast against a toy world —
green ground plane, sky, a white crossbar hung at goal height, white painted
lines on the turf — through the same pinhole + camera-rotation geometry the
detector inverts. These check the discriminators (white mask, the horizon
elevation band, the dense-column filter), not the renderer.

    uv run python -m unittest discover tests
"""

import math
import unittest

import numpy as np

from microduck_mcp.sim_server import (DET_H, DET_W, HEAD_CAM_FOVY_DEG,
                                      HEAD_CAM_PITCH_DEG, detect_goal_pixels)

CAM_H = 0.23   # camera height above the turf, m (standing duck)
GRASS = (40, 130, 55)
SKY = (185, 208, 231)
CLOUD = (245, 246, 248)
WHITE = (235, 238, 240)


def cam_rot(pitch_deg=HEAD_CAM_PITCH_DEG):
    """MuJoCo-convention camera rotation (columns right/up/back), yaw 0,
    pitched down by pitch_deg — the head camera's mounted attitude."""
    c, s = math.cos(math.radians(pitch_deg)), math.sin(math.radians(pitch_deg))
    right = np.array([0.0, -1.0, 0.0])
    up = np.array([s, 0.0, c])
    back = np.array([-c, 0.0, -s])
    return np.stack([right, up, back], axis=1)


def render_scene(bar=None, lines=(), clouds=False, bar_z=(0.17, 0.20),
                 sparse_row_elev=None):
    """Ray-cast a frame: ground plane at z=0 (camera at CAM_H), sky above the
    horizon, an optional white crossbar `bar` = (x_m, y_lo, y_hi), white
    12 mm ground lines at the given x distances, optional white clouds, and
    optionally a 1-px-high white row at a given elevation (a grazing-angle
    line's footprint, to exercise the dense-column filter)."""
    fl = (DET_H / 2) / math.tan(math.radians(HEAD_CAM_FOVY_DEG) / 2)
    R = cam_rot()
    ys, xs = np.mgrid[0:DET_H, 0:DET_W]
    d = np.stack([(xs + 0.5 - DET_W / 2) / fl, (DET_H / 2 - ys - 0.5) / fl,
                  -np.ones((DET_H, DET_W))], axis=-1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    w = d @ R.T  # world-frame rays
    px = np.empty((DET_H, DET_W, 3), dtype=np.uint8)
    px[:] = SKY
    below = w[:, :, 2] < -1e-6
    t = np.where(below, -CAM_H / np.where(below, w[:, :, 2], -1.0), np.inf)
    gx = t * w[:, :, 0]
    px[below] = GRASS
    for lx in lines:
        line = below & (gx >= lx) & (gx <= lx + 0.012)
        px[line] = WHITE
    if clouds:
        elev = np.degrees(np.arcsin(np.clip(w[:, :, 2], -1, 1)))
        px[(elev > 0.3) & (elev < 12.0)] = CLOUD
    if bar is not None:
        bx, ylo, yhi = bar
        tb = bx / np.maximum(w[:, :, 0], 1e-9)
        by, bz = tb * w[:, :, 1], CAM_H + tb * w[:, :, 2]
        hit = (w[:, :, 0] > 0) & (by >= ylo) & (by <= yhi) \
            & (bz >= bar_z[0]) & (bz <= bar_z[1])
        px[hit] = WHITE
    if sparse_row_elev is not None:
        # Exactly ONE white pixel per column, at the requested elevation —
        # the footprint of a grazing-angle painted line.
        elev = np.degrees(np.arcsin(np.clip(w[:, :, 2], -1, 1)))
        rows = np.abs(elev - sparse_row_elev).argmin(axis=0)
        px[rows, np.arange(DET_W)] = WHITE
    return px


class TestGoalDetector(unittest.TestCase):
    def test_empty_pitch_not_visible(self):
        gs = detect_goal_pixels(render_scene(), cam_rot())
        self.assertFalse(gs["visible"])
        self.assertIsNone(gs["bearing_deg"])

    def test_crossbar_dead_ahead(self):
        gs = detect_goal_pixels(render_scene(bar=(1.5, -0.216, 0.216)),
                                cam_rot())
        self.assertTrue(gs["visible"])
        self.assertAlmostEqual(gs["bearing_deg"], 0.0, delta=1.5)
        self.assertIsNotNone(gs["distance_m"])
        self.assertLess(abs(gs["distance_m"] - 1.5) / 1.5, 0.35)

    def test_bearing_positive_to_the_left(self):
        # World +y is the duck's left at yaw 0 — same sign convention as the
        # ball detector and the wz command.
        gs = detect_goal_pixels(
            render_scene(bar=(1.5, 0.2, 0.63)), cam_rot())
        self.assertTrue(gs["visible"])
        expect = math.degrees(math.atan2(0.415, 1.5))
        self.assertAlmostEqual(gs["bearing_deg"], expect, delta=2.0)

    def test_painted_lines_alone_do_not_count(self):
        # A goal-area's worth of nearby painted markings: all below the band.
        gs = detect_goal_pixels(render_scene(lines=(0.4, 0.7, 1.0, 1.4)),
                                cam_rot())
        self.assertFalse(gs["visible"])

    def test_clouds_alone_do_not_count(self):
        # Genuinely white clouds — but above the true horizon, where only sky
        # can be. The band's ceiling excludes them geometrically.
        gs = detect_goal_pixels(render_scene(clouds=True), cam_rot())
        self.assertFalse(gs["visible"])

    def test_grazing_line_in_band_killed_by_dense_columns(self):
        # A far ground line CAN reach the band's elevations, but arrives one
        # pixel per image column; posts stack several. The dense-column
        # filter must reject it.
        gs = detect_goal_pixels(render_scene(sparse_row_elev=-2.5), cam_rot())
        self.assertFalse(gs["visible"])

    def test_crossbar_survives_clouds_and_lines(self):
        gs = detect_goal_pixels(
            render_scene(bar=(1.2, -0.216, 0.216), lines=(0.4, 0.8),
                         clouds=True), cam_rot())
        self.assertTrue(gs["visible"])
        self.assertAlmostEqual(gs["bearing_deg"], 0.0, delta=1.5)

    def test_partial_view_gives_bearing_but_no_distance(self):
        # Goal half out of frame to the left: bearing still usable to turn
        # toward, range (from angular width) honestly null.
        gs = detect_goal_pixels(render_scene(bar=(0.8, 0.0, 1.4)), cam_rot())
        self.assertTrue(gs["visible"])
        self.assertGreater(gs["bearing_deg"], 5.0)  # world +y = duck's left
        self.assertIsNone(gs["distance_m"])


if __name__ == "__main__":
    unittest.main()
