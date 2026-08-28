"""Tests for the head-camera ball detector ("fake mediad").

Offline: no MuJoCo model, no sim server. Synthetic frames are rendered with the
same pinhole geometry detect_ball_pixels() inverts, so these check the maths
(pixel -> bearing/elevation, blob solid angle -> range) rather than the
renderer. Live behaviour is checked against a running sim by hand — see
docstrings in this module for the sequence.

    uv run python -m unittest discover tests
"""

import math
import unittest

import numpy as np

from microduck_mcp.sim_server import (BALL_RADIUS_M, DET_H, DET_W,
                                      HEAD_CAM_FOVY_DEG, detect_ball_pixels)

FL = (DET_H / 2) / math.tan(math.radians(HEAD_CAM_FOVY_DEG) / 2)
ORANGE = (255, 140, 0)
BG = (30, 50, 80)


def render_ball(cam_xyz, w=DET_W, h=DET_H, fovy=HEAD_CAM_FOVY_DEG,
                radius=BALL_RADIUS_M):
    """Rasterise a sphere at `cam_xyz` (camera frame: +x right, +y up, -z
    forward) into an RGB frame, by testing each pixel's ray against the sphere.
    """
    fl = (h / 2) / math.tan(math.radians(fovy) / 2)
    c = np.asarray(cam_xyz, dtype=float)
    ys, xs = np.mgrid[0:h, 0:w]
    d = np.stack([(xs + 0.5 - w / 2) / fl, (h / 2 - ys - 0.5) / fl,
                  -np.ones((h, w))], axis=-1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    # Ray hits the sphere when the perpendicular distance from its centre to
    # the ray is under the radius, and the centre is in front of the camera.
    t = d @ c
    perp2 = float(c @ c) - t * t
    hit = (perp2 <= radius * radius) & (t > 0)
    px = np.empty((h, w, 3), dtype=np.uint8)
    px[:] = BG
    px[hit] = ORANGE
    return px


class TestGeometry(unittest.TestCase):
    def test_empty_frame_not_visible(self):
        px = np.empty((DET_H, DET_W, 3), dtype=np.uint8)
        px[:] = BG
        seen = detect_ball_pixels(px)
        self.assertFalse(seen["visible"])
        self.assertIsNone(seen["distance_m"])
        self.assertIsNone(seen["bearing_deg"])
        self.assertIsNone(seen["elevation_deg"])

    def test_speck_below_min_pixels_is_not_visible(self):
        px = np.empty((DET_H, DET_W, 3), dtype=np.uint8)
        px[:] = BG
        px[120:121, 160:162] = ORANGE  # 2 px
        self.assertFalse(detect_ball_pixels(px)["visible"])

    def test_dead_centre_bearing_and_elevation_are_zero(self):
        seen = detect_ball_pixels(render_ball((0.0, 0.0, -0.5)))
        self.assertTrue(seen["visible"])
        self.assertAlmostEqual(seen["bearing_deg"], 0.0, delta=0.2)
        self.assertAlmostEqual(seen["elevation_deg"], 0.0, delta=0.2)

    def test_bearing_is_positive_to_the_left(self):
        # +x in the camera frame is to the right, so a ball there must read a
        # negative bearing: bearing shares the sign of the wz yaw command.
        left = detect_ball_pixels(render_ball((-0.25, 0.0, -0.6)))
        right = detect_ball_pixels(render_ball((0.25, 0.0, -0.6)))
        self.assertGreater(left["bearing_deg"], 0)
        self.assertLess(right["bearing_deg"], 0)
        self.assertAlmostEqual(left["bearing_deg"], -right["bearing_deg"], delta=0.2)

    def test_bearing_matches_the_true_angle(self):
        for x, z in [(0.1, -0.8), (-0.3, -0.9), (0.5, -0.7), (-0.05, -0.4)]:
            with self.subTest(x=x, z=z):
                seen = detect_ball_pixels(render_ball((x, 0.0, z)))
                self.assertAlmostEqual(seen["bearing_deg"],
                                       -math.degrees(math.atan2(x, -z)), delta=0.5)

    def test_elevation_matches_the_true_angle(self):
        for y, z in [(0.15, -0.8), (-0.2, -0.7), (0.3, -0.9)]:
            with self.subTest(y=y, z=z):
                seen = detect_ball_pixels(render_ball((0.0, y, z)))
                self.assertAlmostEqual(seen["elevation_deg"],
                                       math.degrees(math.atan2(y, -z)), delta=0.5)

    def test_distance_from_blob_size(self):
        for dist in (0.2, 0.3, 0.5, 1.0, 1.5):
            with self.subTest(dist=dist):
                seen = detect_ball_pixels(render_ball((0.0, 0.0, -dist)))
                self.assertTrue(seen["visible"])
                err = abs(seen["distance_m"] - dist) / dist
                self.assertLess(err, 0.10, f"{seen['distance_m']} vs {dist}")

    def test_distance_holds_up_off_axis(self):
        # The projection of an off-axis sphere is a stretched ellipse; a naive
        # pixel-radius model reads badly short out here.
        for cam_xyz in [(0.4, 0.0, -0.6), (0.0, -0.3, -0.6), (0.35, 0.25, -0.7)]:
            with self.subTest(cam_xyz=cam_xyz):
                true_d = float(np.linalg.norm(cam_xyz))
                seen = detect_ball_pixels(render_ball(cam_xyz))
                self.assertTrue(seen["visible"])
                err = abs(seen["distance_m"] - true_d) / true_d
                self.assertLess(err, 0.10, f"{seen['distance_m']} vs {true_d}")

    def test_distance_scales_with_the_lens(self):
        # Same scene through a narrower lens must give the same range.
        for fovy in (45.0, 70.0, 90.0):
            with self.subTest(fovy=fovy):
                seen = detect_ball_pixels(render_ball((0.0, 0.0, -0.6), fovy=fovy),
                                          fovy_deg=fovy)
                self.assertAlmostEqual(seen["distance_m"], 0.6, delta=0.06)

    def test_blob_clipped_by_the_frame_edge_reads_long(self):
        # Known limitation, documented on BallSeen.distance_m: half the blob
        # is off-frame, so the solid angle — and the range read from it — is
        # wrong. Bearing/elevation stay usable enough to turn towards.
        seen = detect_ball_pixels(render_ball((0.0, -0.45, -0.6)))
        self.assertTrue(seen["visible"])
        self.assertGreater(seen["distance_m"], 0.75)  # true range 0.75

    def test_ball_behind_the_camera_is_not_visible(self):
        self.assertFalse(detect_ball_pixels(render_ball((0.0, 0.0, 0.5)))["visible"])

    def test_ball_outside_the_frame_is_not_visible(self):
        # 70 deg vertical fovy over a 4:3 frame -> ~87 deg horizontal; 1.5 m
        # to the side at 0.4 m range is well past the edge.
        self.assertFalse(detect_ball_pixels(render_ball((1.5, 0.0, -0.4)))["visible"])


if __name__ == "__main__":
    unittest.main()
