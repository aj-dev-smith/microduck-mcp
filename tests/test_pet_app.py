"""The desktop pet's arithmetic: screen mapping, walls, drags — no Cocoa.

Everything the overlay believes about where the duck is lives in `pet_map`,
which imports nothing but `math` precisely so this file runs anywhere. The
Cocoa shell (`pet_app`) is verified the only way a window can be: by putting
one on the screen and taking a screenshot of it.

The numbers the daemon owns (`sim_server.PET_*`, `webui.PET_DRAG_GAIN`) are
mirrored in `pet_map`; the tests at the bottom pin the mirrors to the originals
so a change on either side of the socket shows up here rather than as a duck
drawn thirty pixels above the Dock.
"""

import math

import pytest

from microduck_mcp import pet_app, pet_feed, pet_map

# AJ's MacBook, read live from NSScreen during recon: one 1728×1117 pt display,
# backingScaleFactor 2.0, a 90 pt Dock band at the bottom and a 33 pt menu bar.
FRAME = (0.0, 0.0, 1728.0, 1117.0)
VISIBLE = (0.0, 90.0, 1728.0, 994.0)


def a_map(**kw):
    return pet_map.ScreenMap.from_screen(FRAME, VISIBLE, **kw)


# ---------------------------------------------------------------- scale

def test_default_scale_makes_a_180_point_duck():
    smap = a_map()
    assert smap.px_per_meter == pytest.approx(656.0, abs=1.0)
    assert smap.duck_pt == pytest.approx(180.0, abs=0.01)


def test_duck_pt_drives_the_scale():
    assert a_map(duck_pt=90.0).duck_pt == pytest.approx(90.0)
    # an explicit ppm wins over the requested height
    assert a_map(duck_pt=90.0, px_per_meter=1000.0).px_per_meter == 1000.0


# ------------------------------------------------------- points vs pixels

def test_the_window_is_sized_from_the_frame_not_the_wish():
    # 300 pt × 2 = 600 px, past the daemon's 512 px cap -> a 256 pt window.
    smap = a_map(window_pt=300.0, backing_scale=2.0)
    assert smap.frame_px == pet_map.PET_FRAME_MAX_PX == 512
    assert smap.window_pt == 256.0
    # ...and the duck is unaffected: only the air around it shrank.
    assert smap.duck_pt == pytest.approx(180.0, abs=0.01)


def test_a_non_retina_screen_gets_the_window_it_asked_for():
    smap = a_map(window_pt=300.0, backing_scale=1.0)
    assert smap.frame_px == 300 and smap.window_pt == 300.0


def test_supersample_is_the_most_the_offscreen_buffer_allows():
    assert pet_map.supersample_for(512) == 2      # 1024, exactly the cap
    assert pet_map.supersample_for(300) == 3      # 900
    assert pet_map.supersample_for(400) == 2      # 1200 would overflow
    smap = a_map(window_pt=300.0)
    assert smap.frame_px * smap.supersample <= pet_map.PET_OFFSCREEN_MAX_PX


def test_device_ppm_is_the_only_place_scale_leaks_out():
    smap = a_map(backing_scale=2.0)
    assert smap.device_ppm == pytest.approx(2 * smap.px_per_meter)
    assert smap.config_payload()["px_per_meter"] == pytest.approx(
        smap.device_ppm, abs=1e-3)


# ---------------------------------------------------------------- mapping

def test_origin_zero_is_the_middle_of_the_usable_band():
    assert a_map().screen_x_pt(0.0) == pytest.approx(864.0)


def test_screen_mapping_round_trips():
    smap = a_map()
    for x in (-1.2, -0.3, 0.0, 0.45, 1.31):
        assert smap.x_m_from_screen_pt(smap.screen_x_pt(x)) == pytest.approx(x)


def test_window_is_centred_on_the_duck():
    smap = a_map()
    ox, _oy = smap.window_origin(0.0)
    assert ox + 0.5 * smap.window_pt == pytest.approx(smap.center_x_pt)


def test_window_floor_lands_on_the_dock_top_edge():
    # The frame's floor row sits `floor_pad_px` up from its bottom, so the
    # window hangs that far (in points) BELOW the Dock's top edge.
    smap = a_map(floor_pad_px=26, backing_scale=2.0)
    _ox, oy = smap.window_origin(0.0)
    assert oy + 13.0 == pytest.approx(90.0)     # 26 device px = 13 pt


def test_autohidden_dock_walks_the_bottom_edge():
    smap = pet_map.ScreenMap.from_screen(FRAME, (0.0, 0.0, 1728.0, 1084.0))
    assert smap.floor_y_pt == 0.0
    _ox, oy = smap.window_origin(0.0)
    assert oy < 0.0                       # the landing room hangs off-screen


def test_side_dock_narrows_the_band_it_does_not_lower_the_floor():
    smap = pet_map.ScreenMap.from_screen(FRAME, (80.0, 0.0, 1648.0, 1084.0))
    assert smap.floor_y_pt == 0.0
    assert smap.center_x_pt == pytest.approx(904.0)
    assert smap.width_pt == 1648.0


def test_window_position_tracks_x_linearly():
    smap = a_map()
    x0 = smap.window_origin(0.0)[0]
    x1 = smap.window_origin(1.0)[0]
    assert x1 - x0 == pytest.approx(smap.px_per_meter)


# ---------------------------------------------------------------- walls

def test_the_wall_estimate_is_the_margin_the_app_actually_sends():
    # The app pins the margin rather than letting the daemon track the window
    # size, so the two have to agree about where that puts the wall.
    smap = a_map()
    margin_m = smap.config_payload()["wall_margin_m"]
    assert smap.half_span_m == pytest.approx(smap.span_m / 2 - margin_m,
                                             abs=1e-3)


def test_a_duck_at_the_wall_is_wholly_on_screen():
    # The *duck* is the thing that must not run off the bezel. Its window may
    # overhang at the wall and nobody can tell: it is transparent, and the
    # duck sits in the middle of it. Beak to tail is DUCK_DEPTH_M read side-on.
    smap = a_map()
    half_duck_pt = 0.5 * pet_map.DUCK_DEPTH_M * smap.px_per_meter
    assert smap.screen_x_pt(smap.half_span_m) + half_duck_pt <= FRAME[2] + 1e-6
    assert smap.screen_x_pt(-smap.half_span_m) - half_duck_pt >= -1e-6


def test_walls_never_collapse_on_a_silly_screen():
    smap = pet_map.ScreenMap.from_screen((0, 0, 200, 400), (0, 40, 200, 360))
    assert smap.half_span_m > 0.0


def test_span_is_the_screen_in_metres():
    smap = a_map()
    assert smap.span_m == pytest.approx(1728.0 / smap.px_per_meter)
    assert smap.span_m == pytest.approx(2.63, abs=0.02)


# ---------------------------------------------------------------- config

def test_config_payload_is_what_the_daemon_reads():
    # Exactly the keys sim_server._handle_pet_config accepts, and nothing else:
    # the camera is none of the app's business. wall_margin_m IS sent — the
    # daemon's own default tracks the window size, and that puts the walls
    # inside the arc machines/pet.toml turns around in.
    cfg = a_map().config_payload()
    assert set(cfg) == {"px_per_meter", "frame_px", "supersample",
                        "screen_width_px", "floor_pad_px", "wall_margin_m"}
    assert cfg["screen_width_px"] == pytest.approx(3456.0)
    assert pet_map.PET_FRAME_MIN_PX <= cfg["frame_px"] <= pet_map.PET_FRAME_MAX_PX
    assert 1 <= cfg["supersample"] <= 3


def test_the_walls_clear_the_machines_turnaround_and_keep_the_duck_on_screen():
    """The one number both lanes have an opinion about.

    machines/pet.toml turns at |x| = 1.00 m and coasts to a measured 1.063;
    the duck's beak reaches 0.075 m past its base. A wall has to be outside
    the first and inside the last, and the daemon's window-tracking default
    (1.088 m here) is not.
    """
    smap = a_map()                       # 1728 pt band, 656 pt/m, 2x
    wall = smap.half_span_m
    coast_m, half_duck_m, beak_m = 1.063, 0.5 * pet_map.DUCK_DEPTH_M, 0.0752
    assert wall > coast_m + half_duck_m          # never leaned on
    assert wall + beak_m < 0.5 * smap.span_m     # never off the bezel
    assert wall == pytest.approx(1.1942, abs=1e-3)


def test_screen_width_in_pixels_divides_back_to_the_right_metres():
    # The daemon does screen_width_m = screen_width_px / px_per_meter; that has
    # to come out as the same span the app is drawing with.
    smap = a_map()
    cfg = smap.config_payload()
    assert cfg["screen_width_px"] / cfg["px_per_meter"] == pytest.approx(
        smap.span_m, abs=1e-6)


def test_the_daemons_clamp_is_adopted_not_argued_with():
    smap = a_map(window_pt=300.0, backing_scale=1.0)      # asked for 300 px
    assert smap.frame_px == 300
    got = smap.adopt({"frame_px": 256, "supersample": 3, "floor_pad_px": 40,
                      "px_per_meter": 700.0})
    assert got.frame_px == 256 and got.supersample == 3
    assert got.floor_pad_px == 40
    assert got.px_per_meter == pytest.approx(700.0)       # 1× screen
    assert got.window_pt == 256.0


def test_adopt_converts_the_daemons_device_ppm_back_into_points():
    smap = a_map(backing_scale=2.0)
    got = smap.adopt({"px_per_meter": 1312.0})
    assert got.px_per_meter == pytest.approx(656.0)
    assert got.device_ppm == pytest.approx(1312.0)


@pytest.mark.parametrize("cfg", [
    None, "nonsense", {}, {"frame_px": 9999}, {"frame_px": "big"},
    {"supersample": 0}, {"px_per_meter": 0}, {"floor_pad_px": -5},
])
def test_adopt_ignores_a_config_that_cannot_be_true(cfg):
    smap = a_map()
    assert smap.adopt(cfg) == smap


# ---------------------------------------------------------------- gestures

def test_drag_right_pushes_along_plus_x():
    push = pet_map.drag_to_push(120.0, 0.0, 656.0)
    assert push["dx_m"] == pytest.approx(120.0 / 656.0)
    assert push["dy_m"] == 0.0


def test_drag_up_is_a_lift_not_a_depth_fudge():
    # Cocoa +y is up; the endpoint's +dy is DOWN, and the daemon turns -dy into
    # world +z. Flick the duck upwards and it leaves the Dock.
    push = pet_map.drag_to_push(0.0, 90.0, 656.0)
    assert push["dy_m"] < 0.0
    assert push["dx_m"] == 0.0


def test_a_gesture_is_sent_in_metres_so_nobodys_pixels_are_assumed():
    push = pet_map.drag_to_push(65.6, -32.8, 656.0)
    assert push["dx_m"] == pytest.approx(0.1)
    assert push["dy_m"] == pytest.approx(0.05)


def test_the_app_does_not_pre_clamp_the_shove():
    # The daemon clamps to ±2 m/s and reports what landed; trimming here would
    # only hide how hard the user actually threw.
    push = pet_map.drag_to_push(10000.0, 0.0, 656.0)
    assert push["dx_m"] > 2.0
    assert pet_map.push_speed_mps(push) == 2.0


def test_poke_shoves_away_from_the_finger():
    assert pet_map.poke_to_push(40.0, 300.0)["dx_m"] == pet_map.POKE_M
    assert pet_map.poke_to_push(260.0, 300.0)["dx_m"] == -pet_map.POKE_M


def test_a_poke_is_a_nudge_not_a_launch():
    speed = pet_map.push_speed_mps(pet_map.poke_to_push(0.0, 300.0))
    assert 0.3 < speed < 0.7


def test_a_click_is_a_drag_that_did_not_happen():
    assert pet_map.is_click(1.0, 1.0)
    assert not pet_map.is_click(30.0, 0.0)


# ---------------------------------------------------------------- hit box

def test_hit_rect_is_the_duck_standing_on_the_floor_line():
    smap = a_map()
    x0, y0, x1, y1 = pet_app.hit_rect_pt(smap)
    assert x0 < 0.5 * smap.window_pt < x1
    assert y0 < smap.ground_pt                       # a little below the feet
    assert y1 > smap.ground_pt + smap.duck_pt - 1.0  # tall enough for the head


def test_hit_rect_prefers_a_daemon_bbox_flipped_into_cocoa():
    smap = a_map(window_pt=300.0, backing_scale=1.0)   # 300 px frame, 300 pt
    x0, y0, x1, y1 = pet_app.hit_rect_pt(smap, {"bbox": [100, 50, 200, 250]})
    pad = pet_app.HIT_PAD_PT
    assert (x0, x1) == (100.0 - pad, 200.0 + pad)
    assert (y0, y1) == (50.0 - pad, 250.0 + pad)


@pytest.mark.parametrize("bbox", [None, [], [1, 2, 3], [5, 5, 5, 5], "duck"])
def test_hit_rect_ignores_a_nonsense_bbox(bbox):
    smap = a_map()
    assert pet_app.hit_rect_pt(smap, {"bbox": bbox}) == pet_app.hit_rect_pt(smap)


# ---------------------------------------------------------------- feed

def test_feed_never_asks_for_more_frames_than_the_sim_can_render():
    # 42 fps of renders drops the sim to 1.2% of realtime; the cap is not
    # decoration.
    assert pet_feed.PetFeed(fps=1000.0).period >= 1.0 / pet_feed.MAX_FPS
    assert pet_feed.DEFAULT_FPS <= pet_feed.MAX_FPS


def test_feed_starts_offline_and_says_so():
    snap = pet_feed.PetFeed().snapshot()
    assert snap["online"] is False and snap["png"] is None and snap["seq"] == 0


def test_feed_drops_pushes_rather_than_blocking_the_ui():
    feed = pet_feed.PetFeed()
    for _ in range(200):
        feed.push({"dx_m": 0.1, "dy_m": 0.0})   # bounded queue; must not hang


def test_a_reconnect_renegotiates_the_screen():
    # The daemon that comes back may be a fresh one with default walls.
    feed = pet_feed.PetFeed(config={"frame_px": 512})
    feed._config_dirty = False
    feed._mark_offline(OSError("connection refused"))
    assert feed._config_dirty is True
    assert "connection refused" in feed.snapshot()["error"]


def test_the_pose_header_is_found_whatever_case_it_arrives_in():
    assert pet_feed._header({"x-duck-pet": "{}"}, "X-Duck-Pet") == "{}"
    assert pet_feed._header({"X-Duck-Pet": "{}"}, "X-Duck-Pet") == "{}"
    assert pet_feed._header({"Content-Type": "image/png"}, "X-Duck-Pet") is None


def test_malformed_pose_json_does_not_kill_the_feed():
    assert pet_feed._loads(b"{not json") is None
    assert pet_feed._loads("") is None
    assert pet_feed._loads(None) is None
    assert pet_feed._loads('{"base_x_m": 1}') == {"base_x_m": 1}


# ---------------------------------------------------------------- levels

def test_the_chosen_level_clears_the_dock():
    # The Dock's CGWindowLayer is 20; anything at or below it loses.
    assert pet_app.LEVELS[pet_app.DEFAULT_LEVEL] == 25 > 20
    # ...and stays under the levels macOS uses for things the user asked for.
    assert pet_app.LEVELS[pet_app.DEFAULT_LEVEL] < pet_app.LEVELS["popup"]


# ------------------------------------------------------- the mirrors, pinned

def test_the_mirrored_daemon_constants_still_match_the_daemon():
    sim = pytest.importorskip("microduck_mcp.sim_server")
    webui = pytest.importorskip("microduck_mcp.webui")
    assert pet_map.PET_FRAME_MIN_PX == sim.PET_FRAME_MIN_PX
    assert pet_map.PET_FRAME_MAX_PX == sim.PET_FRAME_MAX_PX
    assert pet_map.PET_OFFSCREEN_MAX_PX == sim.PET_OFFSCREEN_MAX_PX
    assert pet_map.DEFAULT_FLOOR_PAD_PX == sim.PET_FLOOR_PAD_PX
    assert pet_map.PET_DRAG_GAIN == webui.PET_DRAG_GAIN


def test_the_mock_answers_the_same_arithmetic_the_daemon_does():
    # A stand-in daemon that drifts from the real one is worse than no mock:
    # the window would be verified against a contract nothing implements.
    sim = pytest.importorskip("microduck_mcp.sim_server")
    from microduck_mcp import pet_mock
    duck = pet_mock.MockDuck()
    duck.push(0.1, 0.0)
    assert duck.kick_x == pytest.approx(0.1 * pet_map.PET_DRAG_GAIN)
    assert pet_mock.PUSH_MAX == sim.PET_PUSH_MAX
    assert pet_mock.DRAG_GAIN == pet_map.PET_DRAG_GAIN
    assert math.isfinite(duck.half_span_m) and duck.half_span_m > 0
    # ...including where the walls end up, which the app now pins rather than
    # leaves to the default rule. A mock still using the default would walk
    # its duck to a different edge than the daemon does.
    assert pet_mock.MOCK_PORT != pet_feed.DEFAULT_PORT   # never fight the sim
    smap = a_map()
    cfg = smap.config_payload()
    walls = _mock_walls(pet_mock, cfg)
    assert walls == pytest.approx(smap.half_span_m, abs=1e-3)


def _mock_walls(pet_mock, cfg):
    """Drive the mock's own /pet/config the way the app does, and read back
    where it put the walls."""
    import json
    import urllib.request
    srv = pet_mock.start_mock(0)                 # port 0: whatever is free
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}/pet/config"
        req = urllib.request.Request(
            url, data=json.dumps(cfg).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.load(r)["config"]["walls_m"][1]
    finally:
        srv.shutdown()
        srv.server_close()
