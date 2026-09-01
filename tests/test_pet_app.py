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
import time

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
    # the camera is none of the app's business. Two of them the app has an
    # opinion about rather than merely reporting — `wall_margin_m`, because
    # the daemon's own default tracks the window size and that puts the walls
    # inside the arc machines/pet.toml turns around in, and `carry_max_z_m`,
    # because how high a duck may be lifted is a claim about a SCREEN and the
    # daemon has no idea how tall this one is.
    cfg = a_map().config_payload()
    assert set(cfg) == {"px_per_meter", "frame_px", "supersample",
                        "screen_width_px", "floor_pad_px", "wall_margin_m",
                        "carry_max_z_m"}
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


def test_a_poke_at_the_toy_is_measured_about_the_toy():
    # The duck is always centred in its own window, so "which side of centre"
    # is the whole question for it. The ball is wherever it rolled to, and a
    # poke measured about the window would roll it the wrong way for every
    # ball right of the duck — poke its left flank and it must go right.
    ball_centre = 240.0
    assert pet_map.poke_to_push(220.0, 300.0, center_pt=ball_centre)["dx_m"] \
        == pet_map.POKE_M
    assert pet_map.poke_to_push(260.0, 300.0, center_pt=ball_centre)["dx_m"] \
        == -pet_map.POKE_M
    # ...and with no centre given it is the window's, byte for byte.
    assert pet_map.poke_to_push(40.0, 300.0, center_pt=None) \
        == pet_map.poke_to_push(40.0, 300.0)


def test_a_click_is_a_drag_that_did_not_happen():
    assert pet_map.is_click(1.0, 1.0)
    assert not pet_map.is_click(30.0, 0.0)


# ------------------------------------------------- what the hand actually said

def test_a_click_is_still_a_poke():
    # The gesture that was here first: no travel, and over almost at once.
    assert pet_map.classify_release(0.05, 2.0, 1.0, 2.0, 1.0) == "poke"
    assert pet_map.classify_release(0.14, 0.0, 0.0, 0.0, 0.0) == "poke"


def test_a_stroke_that_ended_where_it_started_is_not_a_shove():
    # The reason the poke branch is gated on the CLOCK as well as the
    # distance. `classify_release` measures NET displacement, and the one
    # gesture in this app that reliably ends where it began is a hand stroking
    # back and forth over the same duck — so without the brevity term a real
    # 0.9 s pet was answered with a 0.45 m/s shove of the duck being petted.
    assert pet_map.classify_release(0.9, 3.0, 1.0, 2.0, 0.0) == "pet"
    # ...and the same numbers, made in a twelfth of the time, still poke.
    assert pet_map.classify_release(0.08, 3.0, 1.0, 2.0, 0.0) == "poke"


def test_a_hand_that_rested_on_the_duck_and_lifted_off_is_a_pet():
    # A press that never moved and outlasted PET_MIN_S falls through to the
    # pet test, which is the same reading `pet_app.release_kind` gives a carry
    # that let go straight away: a hand resting on a duck, not a shove.
    assert pet_map.classify_release(0.25, 0.0, 0.0, 0.0, 0.0) == "pet"


def test_a_slow_short_stroke_is_a_pet_not_a_shove():
    assert pet_map.classify_release(0.4, 30.0, 10.0, 12.0, 4.0) == "pet"


def test_a_fast_finish_is_a_shove_however_short_the_gesture_was():
    # Same total travel, same duration — but the hand was still moving when
    # it let go. A flick is its last few milliseconds, and so is a pet.
    assert pet_map.classify_release(0.4, 30.0, 10.0, 90.0, 0.0) == "push"


def test_a_stroke_the_duck_walked_out_from_under_is_still_a_pet():
    # The first live test's lesson: the duck keeps walking while it is
    # stroked, so an honest pet routinely ends a few points off a silhouette
    # that has moved on. Release-time position is nobody's business — the
    # gesture began on the duck (release_kind's routing guarantees it) and
    # finished gently, and that is the whole definition of affection here. A
    # genuine throw is caught by the tail speed, never by the hit test.
    assert pet_map.classify_release(0.4, 30.0, 10.0, 12.0, 4.0) == "pet"


def test_a_quick_swipe_is_a_shove_even_if_it_ends_on_the_duck():
    assert pet_map.classify_release(0.05, 30.0, 0.0, 30.0, 0.0) == "push"


def test_a_long_haul_across_the_window_is_a_shove():
    # Past PET_MAX_TRAVEL_PT the hand has dragged the duck somewhere, however
    # slowly it finished.
    assert pet_map.classify_release(2.0, 200.0, 0.0, 1.0, 0.0) == "push"


def test_a_pet_is_gentler_than_a_poke():
    # The number the stroke thresholds were derived from rather than guessed:
    # the daemon turns the last FLICK_WINDOW_S of travel into metres and
    # multiplies by PET_DRAG_GAIN, so the tail cap has to convert to less than
    # a poke's own 0.45 m/s or "pet" would be the harder gesture of the two.
    smap = a_map()
    tail = pet_map.drag_to_push(pet_map.PET_MAX_TAIL_PT, 0.0, smap.px_per_meter)
    poke = pet_map.poke_to_push(0.0, smap.window_pt)
    assert pet_map.push_speed_mps(tail) < pet_map.push_speed_mps(poke)


def test_the_classifier_is_the_only_place_the_split_lives():
    # Three names, and no fourth: the carry lane adds a mode to the app's
    # gesture record, not a return value here.
    kinds = {pet_map.classify_release(dt, dx, 0.0, tail, 0.0)
             for dt in (0.02, 0.4, 3.0)
             for dx in (0.0, 30.0, 400.0)
             for tail in (0.0, 90.0)}
    assert kinds == {"poke", "pet", "push"}


# ------------------------------------------------------- the cursor's own axis

def test_the_cursor_maps_to_the_floor_line_and_back():
    smap = a_map()
    assert smap.z_m_from_screen_pt(smap.floor_y_pt) == pytest.approx(0.0)
    for z in (-0.05, 0.0, 0.18, 1.2):
        assert smap.z_m_from_screen_pt(smap.screen_y_pt(z)) == pytest.approx(z)


def test_a_pointer_inside_the_dock_reads_as_below_the_floor():
    # The Dock's top edge is z = 0, and a pointer on a Dock icon is genuinely
    # underneath the duck's feet. The duck is entitled to know that rather
    # than have it clamped away.
    smap = a_map()
    assert smap.z_m_from_screen_pt(smap.floor_y_pt - 40.0) < 0.0


def test_a_cursor_on_the_dock_is_near_the_floor():
    # 90 pt above the Dock's top edge is 0.137 m — comfortably inside the
    # daemon's 0.35 m default, which is what `cursor.near_floor` decides on.
    sim = pytest.importorskip("microduck_mcp.sim_server")
    smap = a_map()
    z = smap.z_m_from_screen_pt(smap.floor_y_pt + 90.0)
    assert z == pytest.approx(0.137, abs=0.005)
    assert z <= sim.PET_CURSOR_FLOOR_M


def test_a_cursor_up_in_an_editor_is_not_a_visit():
    sim = pytest.importorskip("microduck_mcp.sim_server")
    smap = a_map()
    assert smap.z_m_from_screen_pt(700.0) > sim.PET_CURSOR_FLOOR_M


def test_screen_to_sim_is_both_axes_at_once():
    smap = a_map()
    here = pet_map.screen_to_sim_m(smap, smap.screen_x_pt(0.42),
                                   smap.screen_y_pt(0.18))
    assert here["x_m"] == pytest.approx(0.42)
    assert here["z_m"] == pytest.approx(0.18)


def test_a_cursor_payload_is_a_position_and_says_so():
    smap = a_map()
    body = pet_map.cursor_payload(smap, smap.screen_x_pt(-0.3),
                                  smap.floor_y_pt)
    assert body == {"x_m": pytest.approx(-0.3), "z_m": pytest.approx(0.0),
                    "present": True}


def test_a_touch_payload_is_measured_from_the_duck_not_the_screen():
    # A pet on the beak and a pet on the tail are different facts about the
    # duck, and the same two screen points mean different things a second
    # later when it has walked on.
    smap = a_map()
    body = pet_map.touch_payload(smap, smap.screen_x_pt(0.62),
                                 smap.screen_y_pt(0.17),
                                 base_x_m=0.60, duration_s=0.42,
                                 travel_pt=59.0)
    assert body["kind"] == "pet"
    assert body["x_m"] == pytest.approx(0.02, abs=1e-6)
    assert body["z_m"] == pytest.approx(0.17)      # height stays absolute
    assert body["duration_s"] == pytest.approx(0.42)
    assert body["travel_m"] == pytest.approx(59.0 / smap.px_per_meter)


def test_the_sense_channel_is_cheap_by_construction():
    # Five a second while moving, one a second at rest. The heartbeat is not
    # padding: the daemon stales a sample at 2 s, and a duck that decided the
    # hand had left the room *because it stopped moving* — which is exactly
    # what a hand does while it waits for the duck — is the whole feature
    # failing at the last second.
    sim = pytest.importorskip("microduck_mcp.sim_server")
    assert pet_map.SENSE_HZ <= 5.0
    assert pet_map.SENSE_HEARTBEAT_S < sim.PET_CURSOR_STALE_S


def test_a_hold_can_never_become_one_once_the_hand_has_moved():
    # The latch the carry lane hangs off: six points is one Retina
    # pixel-pair of tremor, and past it the press is a drag forever.
    assert pet_map.CARRY_SLOP_PT < pet_map.CLICK_SLOP_PT * 2
    assert pet_map.CARRY_SLOP_PT > 0.0


# ---------------------------------------------------------------- the lift

def test_a_hold_is_slower_than_a_stroke_and_faster_than_a_wait():
    # The four gestures share one button, so their numbers have to be
    # orderable or two of them are the same gesture. A hold must outlast the
    # minimum a stroke takes (or every pet would promote into a pick-up on
    # its way past) and must not outlast the tap window that takes it back.
    assert pet_map.CARRY_HOLD_S > pet_map.PET_MIN_S
    assert pet_map.CARRY_HOLD_S < pet_map.CARRY_TAP_S


def test_the_hand_talks_often_enough_to_beat_the_daemons_deadman():
    # The daemon lets go after 1.5 s of silence, and the app restates the
    # hand's position 20 times a second EVEN WHEN THE MOUSE IS STILL — which
    # is the whole reason `_gesture_tick` runs off the UI timer rather than
    # off `mouseDragged_`, since a motionless hold produces no drag events.
    sim = pytest.importorskip("microduck_mcp.sim_server")
    assert 1.0 / pet_map.CARRY_HZ < sim.PET_CARRY_TIMEOUT_S / 4


def test_a_hold_that_went_nowhere_and_let_go_is_a_pet():
    # The numbers `pet_app.endDrag_` asks about once a carry has already been
    # promoted: a hand that rested on the duck and lifted straight off was
    # never a pick-up, whatever the 0.30 s timer decided.
    assert pet_map.CARRY_STILL_PT > pet_map.CARRY_SLOP_PT   # tremor is not travel
    assert pet_map.CARRY_STILL_PT < pet_map.PET_MAX_TRAVEL_PT


# ------------------------------------------- the four-gesture state machine
#
# The constants above say the four gestures are ORDERABLE; these say the app
# actually tells them apart. `pet_app`'s decisions were pulled out of the
# Cocoa delegate into plain functions for exactly this — they need a `drag`
# record, a clock and two points, and nothing else, so the hardest thing in
# the app to get right is also runnable on a machine with no window server.


def a_drag(t0=0.0, x0=100.0, y0=200.0, target="duck", mode="undecided",
           carry_off=False, carry_seen=False):
    """The record `pet_app.beginDrag_` builds, without a mouse to build it."""
    return {"t0": t0, "x0": x0, "y0": y0, "target": target, "mode": mode,
            "carry_off": carry_off, "carry_seen": carry_seen}


def test_a_still_hold_becomes_a_carry_and_a_moving_one_never_does():
    # The promotion, and the latch that forbids it. Both halves matter: a
    # hold that is never promoted is a pick-up that does not work, and one
    # promoted after the hand moved is a drag that silently became a lift.
    still = a_drag(t0=10.0)
    hold = pet_map.CARRY_HOLD_S
    assert pet_app.promote_to_carry(still, 10.0 + hold) is True
    assert pet_app.promote_to_carry(still, 10.0 + hold - 0.01) is False
    # ...and the same hold, held just as long, after the hand wandered off.
    moved = a_drag(t0=10.0, carry_off=True)
    assert pet_app.promote_to_carry(moved, 10.0 + 5.0) is False
    # A press already promoted does not promote twice.
    assert pet_app.promote_to_carry(a_drag(t0=10.0, mode="carry"), 99.0) is False
    assert pet_app.promote_to_carry(None, 99.0) is False


def test_you_do_not_pick_up_a_ball():
    # `beginDrag_` latches `carry_off` for the toy as well, but the target
    # test is the one that says why: a ball is a thing you roll.
    ball = a_drag(t0=0.0, target="ball")
    assert pet_app.promote_to_carry(ball, 10.0) is False


def test_the_latch_fires_on_tremor_sized_travel_and_stays_fired():
    d = a_drag()
    assert pet_app.drag_left_the_spot(d, 100.0 + pet_map.CARRY_SLOP_PT - 0.5,
                                      200.0) is False
    assert pet_app.drag_left_the_spot(d, 100.0 + pet_map.CARRY_SLOP_PT + 0.5,
                                      200.0) is True
    # Diagonal counts — it is a distance, not two independent axes.
    assert pet_app.drag_left_the_spot(d, 105.0, 205.0) is True


def test_a_grip_the_daemon_dropped_ends_the_gesture():
    # The deadman, or a reconnect, releases the weld without the mouse button
    # ever coming up. A window that went on believing it held a duck would
    # send an `end` at whatever grab was live by then.
    held = a_drag(mode="carry", carry_seen=True)
    assert pet_app.carry_was_lost(held, carrying=True) is False
    assert pet_app.carry_was_lost(held, carrying=False) is True
    # ...but NOT in the gap between `carry_start` and the feed's first
    # confirmation, or every pick-up would be abandoned on the tick it began.
    fresh = a_drag(mode="carry", carry_seen=False)
    assert pet_app.carry_was_lost(fresh, carrying=False) is False
    assert pet_app.carry_was_lost(a_drag(), carrying=False) is False


def test_a_promoted_carry_that_let_go_at_once_is_a_pet_and_not_a_push():
    # The hold-tap: a hand that rested on the duck and lifted straight off was
    # never a pick-up. `endDrag_` still lets go of the weld (the daemon has
    # one open) but answers the human with a touch, never with qvel.
    tapped = a_drag(t0=1.0, mode="carry")
    assert pet_app.release_kind(tapped, 1.0 + pet_map.CARRY_TAP_S - 0.1,
                                2.0, 1.0, 0.0, 0.0) == "carry_pet"
    # Held longer, or carried somewhere: an ordinary put-down.
    assert pet_app.release_kind(tapped, 1.0 + pet_map.CARRY_TAP_S + 0.1,
                                2.0, 1.0, 0.0, 0.0) == "carry"
    assert pet_app.release_kind(tapped, 1.1, 90.0, 0.0, 0.0, 0.0) == "carry"


def test_a_press_on_the_toy_is_only_ever_a_poke_or_a_push():
    # No classifier: a gentle stroke of a ball is not a thing.
    ball = a_drag(t0=0.0, target="ball")
    assert pet_app.release_kind(ball, 0.02, 1.0, 0.0, 1.0, 0.0) == "poke"
    # ...and a LONG press on the toy is still a poke, because the ball lane
    # asks `is_click` and never the stroke thresholds.
    assert pet_app.release_kind(ball, 3.0, 1.0, 0.0, 0.0, 0.0) == "poke"
    assert pet_app.release_kind(ball, 0.4, 60.0, 0.0, 30.0, 0.0) == "push"


def test_an_ordinary_release_is_the_classifiers_three_and_nothing_else():
    d = a_drag(t0=0.0)
    assert pet_app.release_kind(d, 0.05, 1.0, 0.0, 1.0, 0.0) == "poke"
    assert pet_app.release_kind(d, 0.4, 30.0, 10.0, 12.0, 4.0) == "pet"
    assert pet_app.release_kind(d, 0.4, 30.0, 10.0, 90.0, 0.0) == "push"


def test_the_cursor_channel_is_five_a_second_moving_and_one_at_rest():
    # `_send_cursor`'s whole policy. A parked pointer still has to be
    # restated or the daemon stales it at 2 s and the duck decides the hand
    # left the room — which is exactly what a hand does while it waits.
    period = 1.0 / pet_map.SENSE_HZ
    # The first ever sample: `_sense_at` starts at 0 against a wall clock, so
    # "nothing has been sent yet" is always overdue.
    assert pet_app.cursor_due(time.time(), 0.0, None, (0.0, 0.0)) is True
    # Moved, but too soon.
    assert pet_app.cursor_due(period * 0.5, 0.0, (0.0, 0.0),
                              (0.5, 0.0)) is False
    assert pet_app.cursor_due(period * 1.1, 0.0, (0.0, 0.0),
                              (0.5, 0.0)) is True
    # Not moved: the heartbeat is the only thing that sends it, and it does.
    still = (0.0, 0.0)
    assert pet_app.cursor_due(period * 1.1, 0.0, still, still) is False
    assert pet_app.cursor_due(pet_map.SENSE_HEARTBEAT_S + 0.01, 0.0, still,
                              still) is True
    # A twitch under SENSE_MIN_M is not movement.
    assert pet_app.cursor_due(period * 1.1, 0.0, still,
                              (pet_map.SENSE_MIN_M * 0.5, 0.0)) is False


def test_the_window_hangs_off_the_lifted_frame():
    # The daemon's camera follows a carried duck upwards and says how far in
    # `screen.frame_floor_z_m`; the window has to climb the screen by exactly
    # the same amount or the picture rises inside a window that stayed put.
    smap = a_map()
    flat = smap.window_origin(0.0)
    lifted = smap.window_origin(0.0, 0.35)
    assert lifted[0] == flat[0]                              # only the y moves
    assert lifted[1] - flat[1] == pytest.approx(0.35 * smap.px_per_meter)


def test_a_window_origin_with_no_lift_is_byte_identical_to_before():
    # The default is what keeps every caller written before the pick-up — and
    # every test above — describing the same window.
    smap = a_map()
    for x in (-1.0, -0.25, 0.0, 0.4, 1.1):
        assert smap.window_origin(x) == smap.window_origin(x, 0.0)


def test_the_carry_ceiling_keeps_the_duck_under_the_menu_bar():
    # The app is the only side that knows how tall the screen is. The ceiling
    # is the screen above the walk line, less a duck: carried any higher and
    # the animal has left the world it lives in.
    smap = a_map()
    ceiling = smap.config_payload()["carry_max_z_m"]
    headroom = (smap.screen_h_pt - smap.floor_y_pt) / smap.px_per_meter
    assert ceiling == pytest.approx(headroom - pet_map.DUCK_HEIGHT_M, abs=1e-3)
    # ...and the whole duck is then still on the display.
    assert smap.screen_y_pt(ceiling) + smap.duck_pt <= smap.screen_h_pt + 1.0


def test_the_carry_ceiling_is_inside_what_the_daemon_will_accept():
    # `_handle_pet_config` refuses a value outside its bounds outright, so a
    # silly screen has to come out as a smaller carry rather than a rejected
    # config that leaves the daemon on its default.
    tiny = pet_map.ScreenMap.from_screen((0, 0, 400, 300), (0, 250, 400, 50))
    assert 0.10 <= tiny.config_payload()["carry_max_z_m"] <= 3.0


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


def test_a_missing_rect_is_never_a_hit():
    # `_inside` is what `over_duck` is answered with, and the ball lane asks
    # it about a rectangle that may not exist at all. None is not a hit.
    assert pet_app._inside(None, 10.0, 10.0) is False
    assert pet_app._inside((0.0, 0.0, 20.0, 20.0), 10.0, 10.0) is True
    assert pet_app._inside((0.0, 0.0, 20.0, 20.0), 30.0, 10.0) is False


# ---------------------------------------------------------------- the toy

def a_pose(ball_x_m=0.0, base_x_m=0.0, **extra):
    """A pose with a ball in it, as the daemon's `X-Duck-Pet` header sends
    one: the toy's own metres plus where the duck is standing, which is what
    the frame is centred on."""
    return {"base_x_m": base_x_m,
            "ball": {"present": True, "x_m": ball_x_m, "y_m": 0.0,
                     "z_m": 0.035, "dx_m": ball_x_m - base_x_m,
                     "radius_m": 0.035, "in_frame": True,
                     "vel_mps": [0.0, 0.0, 0.0]},
            **extra}


def test_the_ball_rect_prefers_the_daemons_bbox():
    # Same discipline as the duck's: the segmentation mask is the real
    # silhouette and the arithmetic is only what happens without one.
    smap = a_map(window_pt=300.0, backing_scale=1.0)     # 300 px, 300 pt
    rect = pet_app.ball_rect_pt(smap, a_pose(ball_x_m=0.1,
                                             ball_bbox=[200, 240, 246, 286]))
    pad = pet_app.HIT_PAD_PT
    assert rect[0] == pytest.approx(200.0 - pad)
    assert rect[2] == pytest.approx(246.0 + pad)
    # frame rows count down from the top; Cocoa counts up from the bottom
    assert rect[1] == pytest.approx((300.0 - 286.0) - pad)
    assert rect[3] == pytest.approx((300.0 - 240.0) + pad)


def test_it_falls_back_to_arithmetic_when_the_ball_is_off_frame():
    # The case that matters, and the reason this returns a rectangle rather
    # than None: `ball_bbox` goes null the instant the toy leaves the picture,
    # and "was that press on the ball?" still has to be answerable. The answer
    # has to be a confident NO, which is a real box in the wrong place — not
    # an absence that happens to behave.
    smap = a_map()
    rect = pet_app.ball_rect_pt(smap, a_pose(ball_x_m=1.2))
    assert rect is not None
    assert rect[0] > smap.window_pt, "a ball 1.2 m away is not in the window"
    assert pet_app._inside(rect, 0.5 * smap.window_pt, smap.ground_pt) is False


def test_a_ball_at_the_ducks_feet_sits_on_the_floor_line():
    smap = a_map()
    rect = pet_app.ball_rect_pt(smap, a_pose(ball_x_m=0.0))
    cx, cy = 0.5 * (rect[0] + rect[2]), 0.5 * (rect[1] + rect[3])
    assert cx == pytest.approx(0.5 * smap.window_pt)
    # 0.035 m up from the Dock's edge, which is `ground_pt` in the window
    assert cy == pytest.approx(smap.ground_pt + 0.035 * smap.px_per_meter)


def test_there_is_no_ball_rect_without_a_ball():
    smap = a_map()
    assert pet_app.ball_rect_pt(smap, {}) is None
    assert pet_app.ball_rect_pt(smap, {"ball": None}) is None
    assert pet_app.ball_rect_pt(smap, None) is None


def test_a_press_over_the_ball_targets_the_ball():
    smap = a_map()
    pose = a_pose(ball_x_m=0.16)          # beside the duck, inside the frame
    ball = pet_app.ball_rect_pt(smap, pose)
    bx, by = 0.5 * (ball[0] + ball[2]), 0.5 * (ball[1] + ball[3])
    assert pet_app.press_target(smap, pose, bx, by) == "ball"


def test_a_press_over_the_duck_targets_the_duck():
    smap = a_map()
    pose = a_pose(ball_x_m=0.16)
    duck = pet_app.hit_rect_pt(smap, pose)
    dx, dy = 0.5 * (duck[0] + duck[2]), 0.5 * (duck[1] + duck[3])
    assert pet_app.press_target(smap, pose, dx, dy) == "duck"


def test_the_duck_wins_a_tie():
    # They overlap when the duck is standing over its toy, and at that moment
    # the thing you meant to grab is the one you can see.
    smap = a_map()
    pose = a_pose(ball_x_m=0.0)
    assert pet_app.press_target(smap, pose, 0.5 * smap.window_pt,
                                smap.ground_pt + 10.0) == "duck"


def test_the_window_is_solid_over_the_ball_too():
    # `_update_click_through` asks the same question `beginDrag_` does, and
    # this is that question. A transparent square that let a click through to
    # a Dock icon while the duck's toy was sitting right there would be a ball
    # you can see and cannot touch.
    smap = a_map()
    pose = a_pose(ball_x_m=0.16)
    ball = pet_app.ball_rect_pt(smap, pose)
    bx, by = 0.5 * (ball[0] + ball[2]), 0.5 * (ball[1] + ball[3])
    assert pet_app.press_target(smap, pose, bx, by) != "none"
    # ...and empty air is still empty air.
    assert pet_app.press_target(smap, pose, 4.0, smap.window_pt - 4.0) == "none"


def test_the_chroma_fallback_does_not_make_the_toy_part_of_the_duck():
    # The daemon's segmentation pass splits its masks and sends a duck box and
    # a ball box. The chroma fallback keys the backdrop instead, so its one
    # `bbox` covers duck AND toy and `ball_bbox` is null — and believing that
    # box would put the ball back inside the duck, which is the exact bug the
    # split exists for: a click on the toy would shove the animal. The daemon
    # says which path it is on, so the app asks.
    smap = a_map()
    ball_x = 0.16
    pose = a_pose(ball_x_m=ball_x, alpha="chroma", ball_bbox=None)
    # A combined box, as `_pet_bbox(rgba[:, :, 3])` would report it: from the
    # duck's left flank all the way out past the ball.
    duck_half_px = pet_map.DUCK_DEPTH_M * smap.device_ppm * 0.5
    mid_px = 0.5 * smap.frame_px
    pose["bbox"] = [mid_px - duck_half_px, 40.0,
                    mid_px + ball_x * smap.device_ppm + 24.0, mid_px]
    ball = pet_app.ball_rect_pt(smap, pose)
    bx, by = 0.5 * (ball[0] + ball[2]), 0.5 * (ball[1] + ball[3])
    assert pet_app.press_target(smap, pose, bx, by) == "ball"
    # ...and the duck is still perfectly grabbable — the nominal standing box
    # is duck-sized and centred, which is what makes this fallback honest
    # rather than merely cautious.
    assert pet_app.press_target(smap, pose, 0.5 * smap.window_pt,
                                smap.ground_pt + 20.0) == "duck"
    # A stroke that ended on the toy is not a pet of the duck either.
    assert pet_app._inside(pet_app.hit_rect_pt(smap, pose), bx, by) is False
    # The segmentation path is untouched: that `bbox` really is the duck.
    seg = a_pose(ball_x_m=ball_x, alpha="segmentation",
                 bbox=[100.0, 50.0, 200.0, 250.0])
    assert pet_app.hit_rect_pt(smap, seg) == pet_app._bbox_rect_pt(
        smap, [100.0, 50.0, 200.0, 250.0])


def test_a_lifted_frame_carries_the_ball_up_with_it():
    # `frame_floor_z_m` is the carry lane's camera lift, and it is read here
    # rather than assumed zero so the toy stays where it is drawn once that
    # lands. A daemon that does not report it has not lifted anything.
    smap = a_map()
    flat = pet_app.ball_rect_pt(smap, a_pose(ball_x_m=0.0))
    lifted = pet_app.ball_rect_pt(
        smap, a_pose(ball_x_m=0.0, screen={"frame_floor_z_m": 0.20}))
    assert flat[1] - lifted[1] == pytest.approx(0.20 * smap.px_per_meter)


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


def test_a_reconnect_also_lets_go_of_the_duck():
    # The daemon's own deadman released the weld 1.5 s into the outage, so a
    # token kept across it is a key to a grip that no longer exists — and the
    # mouse-up that eventually comes would `end` whatever grab was live by
    # then. `snapshot()["carrying"]` is how the UI thread finds out.
    feed = pet_feed.PetFeed()
    feed._carry_token = "3f2a91c4"
    assert feed.snapshot()["carrying"] is True
    feed._mark_offline(OSError("connection refused"))
    assert feed.snapshot()["carrying"] is False


def test_the_feed_drops_a_carry_move_before_it_drops_a_start_or_an_end():
    # A `move` is a position nobody will miss; a `start` is a pick-up that
    # never happened and an `end` is a duck left hanging until the deadman
    # notices. So when the queue is full the structural two evict the oldest
    # thing in it rather than themselves.
    feed = pet_feed.PetFeed()
    for _ in range(200):
        feed.carry_move(0.1, 0.2)               # bounded queue; must not hang
    assert feed._carry.full()
    feed.carry_end()
    actions = []
    while not feed._carry.empty():
        actions.append(feed._carry.get_nowait()[1])
    assert actions[-1] == "end"
    assert set(actions[:-1]) == {"move"}


class _RecordingRequests:
    """Stand in for `PetFeed._request` and remember what went out.

    `_drain_carry` needs no socket at all — it takes items off a queue, stamps
    them, and posts. Replacing the one method that owns the socket is enough
    to run the whole protocol offline, which is the point: the token rule, the
    staleness rule and the 409 rule are three bugs that are otherwise very
    hard to see and impossible to test.
    """

    def __init__(self, token="abcd1234", statuses=None):
        self.token = token
        self.statuses = list(statuses or [])
        self.sent = []

    def __call__(self, method, path, body=None):
        self.sent.append((path, dict(body or {})))
        status = self.statuses.pop(0) if self.statuses else 200
        payload = b'{"token":"%s"}' % self.token.encode()
        return status, {}, payload


def a_feed(**kw):
    feed = pet_feed.PetFeed()
    feed._request = _RecordingRequests(**kw)
    return feed, feed._request


def test_the_daemons_token_is_stamped_onto_everything_after_the_start():
    # The daemon mints it, the feed stamps it, the app never sees one. A
    # `move` or an `end` carrying a token that is no longer current gets a 409
    # and changes nothing — which is what stops a stale `end`, from a gesture
    # the window server abandoned, releasing the grab that came after it.
    feed, rec = a_feed(token="deadbeef")
    feed.carry_start(0.1, 0.2)
    feed.carry_move(0.15, 0.25)
    feed.carry_end()
    feed._drain_carry()
    actions = [b["action"] for _p, b in rec.sent]
    assert actions == ["start", "move", "end"]
    assert "token" not in rec.sent[0][1], "the app cannot invent a grab"
    assert rec.sent[1][1]["token"] == "deadbeef"
    assert rec.sent[2][1]["token"] == "deadbeef"
    assert feed._carry_token is None, "an `end` lets go of the token too"


def test_a_move_with_no_grip_behind_it_is_dropped_rather_than_sent():
    # Between a failed `start` and the mouse coming up there is nothing to
    # move, and posting one would be asking the daemon about somebody else's
    # duck.
    feed, rec = a_feed()
    feed.carry_move(0.1, 0.2)
    feed.carry_end()
    feed._drain_carry()
    assert rec.sent == []


def test_staleness_drops_a_move_and_never_drops_an_end():
    # A position that arrives late is wrong information; an `end` is not
    # optional at any age. A regression here leaves the duck hanging until the
    # daemon's 1.5 s deadman on every slow release.
    feed, rec = a_feed()
    feed._carry_token = "cafe1234"
    old = time.monotonic() - (pet_feed.PUSH_STALE_S + 1.0)
    feed._carry.put_nowait((old, "move", {"x_m": 0.1, "z_m": 0.2}))
    feed._carry.put_nowait((old, "end", {}))
    feed._drain_carry()
    assert [b["action"] for _p, b in rec.sent] == ["end"]


def test_a_409_lets_go_of_the_token_instead_of_arguing_with_it():
    # The daemon says the grip is gone. Everything queued behind that is about
    # a duck this window is no longer holding.
    feed, rec = a_feed(statuses=[409])
    feed._carry_token = "cafe1234"
    feed.carry_move(0.1, 0.2)
    feed.carry_move(0.3, 0.4)
    feed._drain_carry()
    assert len(rec.sent) == 1, "it kept talking about a grip it had lost"
    assert feed._carry_token is None


def test_the_cursor_slot_coalesces_and_clears_itself():
    # One SLOT, not a queue: where the mouse was three samples ago is not
    # information anybody wants delivered late. And it is read-and-cleared, so
    # a pointer that stopped moving costs nothing until the app resamples it.
    feed, rec = a_feed()
    feed.sense({"x_m": 0.1, "z_m": 0.2, "present": True})
    feed.sense({"x_m": 0.4, "z_m": 0.2, "present": True})
    feed._send_sense_if_due()
    assert [p for p, _b in rec.sent] == ["/pet/sense"]
    assert rec.sent[0][1]["x_m"] == 0.4
    feed._send_sense_if_due()
    assert len(rec.sent) == 1


def test_the_app_never_handles_a_carry_token():
    # The daemon mints it and the feed stamps it: `carry_start` takes two
    # metres and nothing else, so there is no way for the window to invent
    # one, hold a stale one, or send somebody else's.
    import inspect
    for name in ("carry_start", "carry_move"):
        params = list(inspect.signature(
            getattr(pet_feed.PetFeed, name)).parameters)
        assert params == ["self", "x_m", "z_m"]
    assert list(inspect.signature(
        pet_feed.PetFeed.carry_end).parameters) == ["self"]


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
    # The cursor channel's other end. The heartbeat has to beat the stale
    # window or a parked pointer disappears from under the duck's feet.
    assert pet_map.SENSE_HEARTBEAT_S < sim.PET_CURSOR_STALE_S
    # The toy. Not a preference on either side: BALL_RADIUS_M is an input to
    # the head camera's range solve, and the app sizes a grab box with it.
    assert pet_map.BALL_RADIUS_M == sim.BALL_RADIUS_M
    # The pick-up's one mirror. The app never computes the lift — the daemon
    # reports the one it applied — but the trigger is what decides whether a
    # stumble moves the window, and a silent disagreement about it would read
    # as a duck that jitters up and down the screen while it walks.
    assert pet_map.PET_LIFT_TRIGGER_M == sim.PET_LIFT_TRIGGER_M
    assert pet_mock_constants_match(sim)


def pet_mock_constants_match(sim):
    """The stand-in daemon's mirrors of the human channel, pinned too."""
    from microduck_mcp import pet_mock
    return (pet_mock.CURSOR_STALE_S == sim.PET_CURSOR_STALE_S
            and pet_mock.CURSOR_FLOOR_M == sim.PET_CURSOR_FLOOR_M
            and pet_mock.TOUCH_RECENT_S == sim.PET_TOUCH_RECENT_S
            and pet_mock.TOUCH_COOLDOWN_S == sim.PET_TOUCH_ACK_COOLDOWN_S
            # The toy's size is not a preference: it is an input to the head
            # camera's range solve, so a mock that drew a different ball would
            # verify the window's hit box against a ball the daemon cannot have.
            and pet_mock.BALL_RADIUS_M == sim.BALL_RADIUS_M
            # The pick-up's four. The deadman is the one the app is actually
            # tested against (a mock that never let go would leave
            # `_gesture_tick`'s "the daemon let go" path unexercised), and the
            # lift trigger decides where the window starts climbing.
            and pet_mock.CARRY_TIMEOUT_S == sim.PET_CARRY_TIMEOUT_S
            and pet_mock.CARRY_MIN_Z_M == sim.PET_CARRY_MIN_Z_M
            and pet_mock.CARRY_HAND_MPS == sim.PET_CARRY_HAND_SPEED_MPS
            and pet_mock.LIFT_TRIGGER_M == sim.PET_LIFT_TRIGGER_M)


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


def test_the_mock_answers_the_human_routes_the_same_shape_the_daemon_does():
    # `test_the_mock_answers_the_same_arithmetic_the_daemon_does` says why: a
    # stand-in that drifts verifies the window against a contract nothing
    # implements. The two new routes are exactly where that would bite —
    # `pet_feed` reads the token-free reply shape and `pet_app` rate-limits
    # against the round trip.
    from microduck_mcp import pet_mock
    srv = pet_mock.start_mock(0)
    try:
        port = srv.server_address[1]
        sense = _mock_post(port, "/pet/sense",
                           {"x_m": 0.42, "z_m": 0.18, "present": True})
        assert sense["ok"] and sense["cursor"]["present"] is True
        assert sense["cursor"]["dist_m"] == pytest.approx(
            abs(sense["cursor"]["dx_m"]))
        assert sense["cursor"]["near_floor"] is True
        # ...and a pointer that left the screen is dropped, not aged out.
        gone = _mock_post(port, "/pet/sense", {"present": False})
        assert gone["cursor"]["present"] is False
        assert gone["cursor"]["age_s"] == 999.0
        # Nonsense is refused rather than absorbed.
        assert _mock_post(port, "/pet/sense", {"x_m": "left", "z_m": 0}) \
            ["error"].startswith("x_m and z_m")

        first = _mock_post(port, "/pet/touch", {"kind": "pet"})
        assert first["acknowledged"] is True
        assert (first["emote"], first["sound"]) == ("nuzzle", "coo")
        second = _mock_post(port, "/pet/touch", {"kind": "pet"})
        assert second["acknowledged"] is False      # inside the cooldown
        assert second["count"] == 2                 # ...but it still counted
        assert _mock_post(port, "/pet/touch", {"kind": "bite"})["ok"] is False
        # The state block the app reads back carries both.
        state = _mock_get(port, "/pet/state")
        assert state["touch"]["count"] == 2 and state["touch"]["petted"] is True
        assert set(state["cursor"]) == {
            "present", "x_m", "z_m", "dx_m", "dist_m", "age_s", "near_floor",
            "speed_mps"}
    finally:
        srv.shutdown()
        srv.server_close()


def test_the_mock_answers_the_toy_the_same_shape_the_daemon_does():
    # Same argument as the two tests above, aimed at the field the window's
    # hit test now reads. The shapes that matter: the `ball` block's key set,
    # `target` on a push, and `ball_bbox` going null when the toy leaves the
    # frame — which is precisely the case `pet_app.ball_rect_pt` falls back
    # to arithmetic for.
    from microduck_mcp import pet_mock
    srv = pet_mock.start_mock(0)
    try:
        port = srv.server_address[1]
        state = _mock_get(port, "/pet/state")
        assert set(state["ball"]) == {"present", "x_m", "y_m", "z_m", "dx_m",
                                      "radius_m", "in_frame", "vel_mps"}
        assert state["ball"]["dx_m"] == pytest.approx(
            state["ball"]["x_m"] - state["base_x_m"])
        # A shove aimed at the toy moves the toy and leaves the duck alone.
        before = _mock_get(port, "/pet/state")
        rolled = _mock_post(port, "/pet/push",
                            {"dx_m": 0.075, "dy_m": 0.0, "target": "ball"})
        assert rolled["ok"] and rolled["target"] == "ball"
        after = _mock_get(port, "/pet/state")
        assert after["ball"]["vel_mps"][0] > before["ball"]["vel_mps"][0]
        # ...and a target nobody has heard of is refused rather than absorbed.
        assert _mock_post(port, "/pet/push",
                          {"dx_m": 0.1, "dy_m": 0.0, "target": "dock"})["ok"] \
            is False
        # The frame's second box, and the null the app's fallback exists for.
        import json
        import urllib.request
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/pet/frame",
                                    timeout=5) as r:
            pose = json.loads(r.headers["X-Duck-Pet"])
        assert "ball_bbox" in pose
        if not pose["ball"]["in_frame"]:
            assert pose["ball_bbox"] is None
    finally:
        srv.shutdown()
        srv.server_close()


def test_the_mock_answers_the_pick_up_the_same_shape_the_daemon_does():
    # The third route, and the one with real state behind it. What the app is
    # verified against here is the CONTRACT: a token it did not choose, a 409
    # for one that is not current (which is what clears `pet_feed`'s token and
    # tears the app's gesture down), a duck that follows the hand rather than
    # teleporting to it, and a `frame_floor_z_m` that starts moving once the
    # duck is off the floor.
    from microduck_mcp import pet_mock
    srv = pet_mock.start_mock(0)
    try:
        port = srv.server_address[1]
        started = _mock_post(port, "/pet/carry",
                             {"action": "start", "x_m": 0.0, "z_m": 0.15})
        assert started["ok"] and started["carried"] is True
        token = started["token"]
        assert isinstance(token, str) and token
        assert set(started["limits"]) == {"x_m", "z_m", "hand_speed_mps",
                                          "timeout_s"}
        # A move with the wrong token changes nothing and says 409.
        stale = _mock_post(port, "/pet/carry",
                           {"action": "move", "token": "deadbeef",
                            "x_m": 1.0, "z_m": 0.5})
        assert stale["ok"] is False and stale["conflict"] is True
        # ...and a second start, while somebody already has it, likewise.
        assert _mock_post(port, "/pet/carry", {"action": "start"})["ok"] is False

        # Lift it, and let the mock's own clock chase the hand for a moment.
        _mock_post(port, "/pet/carry",
                   {"action": "move", "token": token, "x_m": 0.0, "z_m": 0.50})
        for _ in range(40):
            state = _mock_get(port, "/pet/state")
            if state["base_z_m"] > pet_mock.LIFT_TRIGGER_M:
                break
            time.sleep(0.02)
        assert state["carry"]["carried"] is True
        assert state["carry"]["token"] == token
        assert state["base_z_m"] > pet_mock.LIFT_TRIGGER_M, "it never rose"
        assert state["screen"]["frame_floor_z_m"] == pytest.approx(
            state["base_z_m"] - pet_mock.LIFT_TRIGGER_M, abs=1e-6)

        ended = _mock_post(port, "/pet/carry",
                           {"action": "end", "token": token})
        assert ended["ok"] and ended["carried"] is False
        assert ended["released_vel_mps"] is not None
        assert _mock_get(port, "/pet/state")["carry"]["carried"] is False
        # An end for a grip that is over is the 409 too, not a silent 200.
        assert _mock_post(port, "/pet/carry",
                          {"action": "end", "token": token})["ok"] is False
        assert _mock_post(port, "/pet/carry",
                          {"action": "juggle"})["ok"] is False
    finally:
        srv.shutdown()
        srv.server_close()


def test_the_mock_lets_go_when_nobody_is_holding_it():
    # The deadman, which is the whole reason `_gesture_tick` watches
    # `snapshot()["carrying"]` instead of trusting its own mouse state.
    from microduck_mcp import pet_mock
    duck = pet_mock.MockDuck()
    duck.carry_start(0.0, 0.30)
    assert duck._carry_state()["carried"] is True
    duck.carry["last"] -= pet_mock.CARRY_TIMEOUT_S + 0.1
    duck.step()
    assert duck._carry_state()["carried"] is False


def _mock_post(port, path, body):
    import json
    import urllib.error
    import urllib.request
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:          # a refusal is still an answer
        return json.load(e)


def _mock_get(port, path):
    import json
    import urllib.request
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as r:
        return json.load(r)


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
