"""`duck-pet` — the duck, walking along the top of your Dock.

A borderless transparent always-on-top window that draws whatever the sim
daemon last rendered and sits wherever the sim says the duck is standing. That
is the entire app. It has no gait, no sprite sheet, no tweening and no idle
animation: **every frame on your screen is a physics step through the shipped
ONNX walk policy**, fetched over HTTP from a running `duck-sim --scene desktop`.
Kill the daemon and the duck freezes mid-stride — that freeze is the feature,
it is how you can tell nothing here is faked.

    ┌──────────────┐   sim x ─ px_per_meter ─> window origin.x
    │      🦆      │   the DUCK is centred in the frame,
    └──────────────┘   the WINDOW is what travels
    ══════════════════ Dock top edge = the sim floor (NSScreen.visibleFrame.y)

The four macOS facts that make it work, all of them established empirically
rather than remembered:

1. **Window level.** The Dock's `CGWindowLayer` is 20. `NSStatusWindowLevel`
   (25) renders above it — verified by putting a real window over the Dock and
   screenshotting it, which is the only test that counts here.
   `kCGDockWindowLevel + 1` (21) also clears the Dock, and
   `NSPopUpMenuWindowLevel` (101) and `NSScreenSaverWindowLevel` (1000) clear
   it by a mile — but 101+ also puts a duck over open menus, Spotlight and
   Mission Control, which is a duck that has outstayed its welcome. **25
   wins**: above the Dock, below anything the user deliberately summoned.
   `--level` re-runs the experiment if a future macOS moves the furniture.
2. **Activation policy `Accessory`** keeps it out of the Dock and out of
   Cmd-Tab, and `orderFrontRegardless()` shows it without stealing focus. The
   window refuses to become key for the same reason: a pet must never take the
   cursor off what you were typing into.
3. **Click-through by geometry.** The window overlaps the Dock, so it must not
   eat clicks meant for Dock icons. `ignoresMouseEvents` is re-decided every UI
   tick from the cursor position against the duck's own standing box: over the
   duck, the window takes the click; anywhere else in that transparent square,
   the click falls straight through to whatever is underneath.
4. **Points vs device pixels.** Cocoa places windows in points; the daemon
   renders pixels, and its frame is capped at 512 px. So the window is sized
   *from the frame* (512 px ÷ 2× = a 256 pt window), never the other way
   round — a window that does not match the image inside it is the one bug a
   screenshot cannot show you. `pet_map` owns that conversion.

The mouse says four different things through one button. Three of them are
named on the way *up*, in `pet_map.classify_release`: a click is a poke, a
slow short stroke that ends on the duck is a pet, and anything faster is a
shove. Only the first and the last reach `qvel` — a pet is answered by the
duck, not by the physics, which is the one place in this app where "nothing is
faked" means "and nothing is moved either". Alongside all that the window
reports where the mouse *is*, five times a second, so a duck can notice a hand
that has not touched it yet.

The fourth is named on the way *down*, or rather partway through: a press that
stays still for `CARRY_HOLD_S` is a **pick-up**, and from then on the pointer
is streamed to `/pet/carry` and the duck hangs off an invisible welded hand
until the button comes up. That promotion is decided on the 30 Hz tick and not
in the drag callback, because `mouseDragged_` does not fire when the mouse
does not move — a perfectly still hold, which is exactly the gesture, produces
no events at all. Picking the duck up is also the one gesture that moves the
WINDOW as well as the duck: the daemon's camera follows a lifted duck upwards
and says how far in `screen.frame_floor_z_m`, and `_place` climbs the screen
by the same amount.

There are two things in the window to press on, and which one a press was
aimed at is decided on the way *down*, in `press_target`, from the two boxes
the daemon sends (`bbox` is the duck, `ball_bbox` is the toy — on the
segmentation path; the chroma fallback cannot split them and says so, and
`hit_rect_pt` falls back to arithmetic rather than believe it). A press on the
ball is always a push — you do not pet a ball — and it travels through the
same `/pet/push` route with `target: "ball"`, so the shove that rolls the toy
is the shove that staggers the duck, at the same gain, into a different qvel.
The toy is only in the picture while it is within about 0.195 m of the duck;
past that it is neither drawn nor clickable, and the duck has to go and fetch
it (`machines/pet.toml`'s chase), which is the design and not a limit.

Threading: Cocoa's main thread only draws and moves the window; every socket
lives on `pet_feed.PetFeed`'s background thread.

    duck-sim --scene desktop       # headless; owns the default socket, port 8400
    duck machine load machines/pet.toml && duck machine arm
    duck-pet                       # both defaults are 8400: no flags needed

    duck-pet --mock                # against a stand-in daemon, no sim needed
    duck-pet --port 8410 --duck-pt 180 --fps 20 -v      # against a dev daemon
"""

import argparse
import math
import signal
import sys
import time

from . import pet_feed, pet_map

# How often the window redraws and re-reads the feed. Higher than the frame
# rate on purpose: the window position should be applied the instant a frame
# lands, and this loop is also what re-decides click-through and keeps Python's
# signal handlers reachable from inside an ObjC run loop.
UI_HZ = 30.0

# Levels worth trying, for the empirical note in the docstring above.
LEVELS = {"floating": 3, "dock+1": 21, "mainmenu": 24, "status": 25,
          "popup": 101, "screensaver": 1000}
DEFAULT_LEVEL = "status"

# The offline wash: cool, dark, and applied *source-atop* so it tints the
# duck's own pixels and not the transparent square around it. Subtle enough to
# read as "this picture is stale", not as a dialog.
OFFLINE_TINT = (0.24, 0.30, 0.42, 0.38)
# Don't flash the tint on a single dropped frame; a real outage lasts.
OFFLINE_AFTER_S = 1.5

# Grab slop around the duck's silhouette, points.
HIT_PAD_PT = 8.0


def _import_cocoa():
    """AppKit is imported lazily so `pet_map` stays testable off a Mac."""
    try:
        import AppKit
        import Foundation
        import objc
    except ImportError as e:  # pragma: no cover — depends on the platform
        raise SystemExit(
            "duck-pet needs pyobjc (macOS only):\n"
            "    uv add 'pyobjc-framework-Cocoa>=10; sys_platform == \"darwin\"' "
            "'pyobjc-framework-Quartz>=10; sys_platform == \"darwin\"'\n"
            f"({e})") from e
    return AppKit, Foundation, objc


# ---------------------------------------------------------------- geometry

def screen_map_for(AppKit, args) -> pet_map.ScreenMap:
    """Read the chosen screen and build the mapping from it.

    `visibleFrame` is the truth and `defaults read com.apple.dock` is not: the
    orientation/autohide/magnification keys are simply *absent* whenever the
    user is on defaults, while `visibleFrame` already accounts for all three,
    for the current tile size, and for a Dock that moved a second ago.
    """
    screens = AppKit.NSScreen.screens()
    if not screens:
        raise SystemExit("no screens")
    scr = screens[min(max(args.screen, 0), len(screens) - 1)]
    f, v = scr.frame(), scr.visibleFrame()
    frame = (f.origin.x, f.origin.y, f.size.width, f.size.height)
    # The walk line is the Dock's top edge plus whatever nudge the user wants:
    # the Dock's top is a rounded, translucent shelf, so standing exactly on
    # `visibleFrame.origin.y` can read as hovering. Negative sinks the feet in.
    visible = (v.origin.x, v.origin.y + args.floor_offset_pt,
               v.size.width, v.size.height)
    return pet_map.ScreenMap.from_screen(
        frame, visible, duck_pt=args.duck_pt, px_per_meter=args.ppm,
        window_pt=args.window_pt, frame_px=args.frame_px,
        floor_pad_px=args.floor_pad_px,
        backing_scale=float(scr.backingScaleFactor()))


def hit_rect_pt(smap: pet_map.ScreenMap, pose: dict = None) -> tuple:
    """Where the duck's pixels are inside the window, in Cocoa points.

    The duck is always centred in its frame and always standing on the floor
    line, so its box is arithmetic rather than image analysis — no per-frame
    alpha scan on the UI thread. If a daemon ever reports an exact `bbox` (top
    left origin, frame pixels) off the segmentation mask it already computes,
    this prefers it; the nominal standing box is deliberately generous, because
    a grab rectangle that is slightly too big is a far smaller sin than a duck
    you cannot grab.

    **Except on the chroma fallback, where `bbox` is not the duck.** The
    segmentation pass splits its masks and hands back a duck box and a ball
    box; the fallback is "every pixel that is not the backdrop", which is one
    silhouette covering duck AND toy with `ball_bbox: null` beside it. Trusting
    that box would make the ball part of the duck again — a click on the toy
    would shove the animal, and a stroke that ended on the ball would read as
    a pet — which is the exact bug the mask split exists to prevent. The daemon
    says which path it is on (`alpha`), so this asks rather than assumes, and
    under chroma falls back to the nominal standing box: duck-sized, centred,
    and leaving `ball_rect_pt`'s own arithmetic free to answer for the toy.
    """
    w = smap.window_pt
    pose = pose or {}
    rect = (None if pose.get("alpha") == "chroma"
            else _bbox_rect_pt(smap, pose.get("bbox")))
    if rect is not None:
        return rect
    half = pet_map.DUCK_DEPTH_M * smap.px_per_meter   # beak and tail included
    foot = smap.ground_pt
    return (0.5 * w - half - HIT_PAD_PT, foot - HIT_PAD_PT,
            0.5 * w + half + HIT_PAD_PT, foot + smap.duck_pt + HIT_PAD_PT)


def _bbox_rect_pt(smap: pet_map.ScreenMap, bbox) -> tuple:
    """A daemon bbox (frame pixels, top-left origin) as a Cocoa rectangle.

    Shared by the duck's box and the ball's because it is the same flip both
    times: the frame counts rows down from the top, Cocoa counts points up
    from the bottom, and the scale between them is window points per frame
    pixel. `None` for anything that is not a real box — the daemon sends
    `null` the moment a thing leaves the frame, and the callers below each
    have their own arithmetic to fall back on.
    """
    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
        return None
    try:
        x0, y0, x1, y1 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None
    if not (x1 > x0 and y1 > y0):
        return None
    w = smap.window_pt
    s = w / float(smap.frame_px)
    x0, y0, x1, y1 = x0 * s, y0 * s, x1 * s, y1 * s
    return (x0 - HIT_PAD_PT, (w - y1) - HIT_PAD_PT,
            x1 + HIT_PAD_PT, (w - y0) + HIT_PAD_PT)


def ball_rect_pt(smap: pet_map.ScreenMap, pose: dict = None) -> tuple:
    """Where the ball's pixels are inside the window, in Cocoa points.

    The exact twin of `hit_rect_pt`, and it shares that function's discipline:
    prefer the daemon's `ball_bbox`, which comes off the segmentation mask and
    is the real silhouette, and fall back to arithmetic otherwise.

    The fallback is the case that MATTERS here, unlike for the duck. The duck
    is always in its own frame; the ball is only in it while it is within
    ~0.195 m of the duck, and `ball_bbox` is `null` the instant it is not. But
    "was that press on the ball?" still has to be answerable then, and the
    answer has to be NO — so this returns a real rectangle, computed from
    `pose["ball"]`, which simply falls outside the window when the ball is
    somewhere else. A `None` there would work by accident; a rectangle in the
    right place works on purpose, and keeps working the moment the ball rolls
    a few pixels back into view.

    `None` only when there is no ball at all — a scene without one, or a
    daemon that has not sent a pose yet.
    """
    pose = pose or {}
    rect = _bbox_rect_pt(smap, pose.get("ball_bbox"))
    if rect is not None:
        return rect
    ball = pose.get("ball")
    if not isinstance(ball, dict) or ball.get("x_m") is None:
        return None
    # The window is centred on the duck and hung off the frame's floor line,
    # so the ball's own metres map straight into window points. `frame_floor_z_m`
    # is the lift the camera has taken (0 unless the duck is being carried);
    # a daemon that does not report it has not lifted anything.
    base_x = float(pose.get("base_x_m") or 0.0)
    floor_z = float((pose.get("screen") or {}).get("frame_floor_z_m") or 0.0)
    ppm = smap.px_per_meter
    cx = 0.5 * smap.window_pt + (float(ball["x_m"]) - base_x) * ppm
    cy = smap.ground_pt + (float(ball.get("z_m") or 0.0) - floor_z) * ppm
    r = float(ball.get("radius_m") or pet_map.BALL_RADIUS_M) * ppm + HIT_PAD_PT
    return (cx - r, cy - r, cx + r, cy + r)


def press_target(smap: pet_map.ScreenMap, pose: dict, x_pt: float,
                 y_pt: float) -> str:
    """What a press at that window point is ON: `'duck'`, `'ball'` or `'none'`.

    One place, because two callers need the same answer and must not disagree:
    `beginDrag_` decides what a gesture is about, and `_update_click_through`
    decides whether the window is solid at all. If the window were solid over
    a ball that `beginDrag_` then read as duck, a click on the toy would shove
    the animal.

    The duck wins a tie. They overlap when the duck is standing over the ball,
    and at that moment the thing you meant to grab is the one you can see.

    The duck is asked first on both alpha paths, and it is safe on both:
    `hit_rect_pt` refuses the chroma fallback's combined `bbox` and answers
    with the nominal standing box there, so the toy is not swallowed by a
    silhouette that happens to include it.
    """
    if _inside(hit_rect_pt(smap, pose), x_pt, y_pt):
        return "duck"
    if _inside(ball_rect_pt(smap, pose), x_pt, y_pt):
        return "ball"
    return "none"


def _inside(rect, x_pt, y_pt) -> bool:
    """Is that window-local point inside that rectangle? None is never hit."""
    if not rect:
        return False
    x0, y0, x1, y1 = rect
    return (x0 <= x_pt <= x1) and (y0 <= y_pt <= y1)


# ------------------------------------------------- what the hand is doing
#
# Four gestures share one mouse button, and every decision about WHICH one is
# happening lives in the four functions below rather than inside the Cocoa
# delegate. They take a `drag` record, a clock and a couple of points, and
# they touch nothing else — because this is the single easiest thing in the
# app to get wrong and the only way to hold it still is to be able to run it
# without a window server. `PetController` keeps the plumbing: reading the
# mouse, talking to the feed, drawing.
#
# The `drag` record is built once in `beginDrag_` and is the whole memory of
# a gesture: when and where the button went down (`t0`, `x0`, `y0`), what it
# was aimed at (`target`), whether it has already become a pick-up (`mode`)
# and whether it ever can (`carry_off`).


def drag_left_the_spot(drag: dict, x_pt: float, y_pt: float) -> bool:
    """Has the hand moved far enough that this press can never be a hold?

    Six points is one Retina pixel-pair of tremor. Past that the hand is
    dragging, and `sampleDrag_` latches it for the life of the gesture — a
    press that wandered and came back was still a drag. The sampler is the
    only place that sees every mouse move, which is why the latch lives there
    and the promotion below merely reads it.
    """
    return (math.hypot(x_pt - drag["x0"], y_pt - drag["y0"])
            > pet_map.CARRY_SLOP_PT)


def promote_to_carry(drag: dict, now: float) -> bool:
    """Should this press become a pick-up on this tick?

    Four terms, and each rules out a different gesture: it must not already
    be a carry (`undecided`), it must not have moved (`carry_off`), it must
    be on the animal — you do not pick up a ball — and it must have lasted
    `CARRY_HOLD_S`. Asked from the UI timer and never from the drag callback:
    `mouseDragged_` does not fire when the mouse does not move, so a
    perfectly still hold, which is exactly what a pick-up is made of,
    produces no events at all.
    """
    return (drag is not None and drag["mode"] == "undecided"
            and not drag["carry_off"] and drag["target"] == "duck"
            and now - drag["t0"] >= pet_map.CARRY_HOLD_S)


def carry_was_lost(drag: dict, carrying: bool) -> bool:
    """Did the duck leave this window's hand without the button coming up?

    A reconnect, or the daemon's own 1.5 s deadman, ends a carry the mouse
    knows nothing about. Only meaningful once the grip was actually SEEN
    (`carry_seen`): between `carry_start` and the feed's first confirmation
    there is no token yet, and treating that gap as a loss would abandon
    every pick-up on the tick it began.
    """
    return (drag is not None and drag["mode"] == "carry"
            and bool(drag["carry_seen"]) and not carrying)


def release_kind(drag: dict, now: float, total_dx: float, total_dy: float,
                 tail_dx: float, tail_dy: float) -> str:
    """What the button coming up meant. Five names, three lanes.

      * A press that became a **carry** is already answered — the duck has
        been hanging off a hand — so there is nothing to classify: `carry`
        lets go and the daemon hands over whatever velocity the hand had.
        Unless the hand never went anywhere and let go almost at once, which
        was not a pick-up but a hand RESTING on a duck: `carry_pet` lets go
        and answers it as a stroke.
      * A press on the **ball** is a push or a poke and nothing else. The
        classifier is not consulted — "a gentle stroke of a ball" is not a
        thing.
      * Everything else is `pet_map.classify_release`'s three, judged on the
        press's own record: where the hand ended up is the classifier's
        business only as displacement, never as a hit test (the duck walks
        while it is stroked).
    """
    if drag["mode"] == "carry":
        tap = (math.hypot(total_dx, total_dy) < pet_map.CARRY_STILL_PT
               and now - drag["t0"] < pet_map.CARRY_TAP_S)
        return "carry_pet" if tap else "carry"
    if drag["target"] == "ball":
        return "poke" if pet_map.is_click(total_dx, total_dy) else "push"
    return pet_map.classify_release(now - drag["t0"], total_dx, total_dy,
                                    tail_dx, tail_dy)


def cursor_due(now: float, last_at: float, last_m, here_m) -> bool:
    """Is this pointer sample worth a post? Five a second, or one a second.

    The rate limit for `/pet/sense`: while the pointer is moving it goes at
    `SENSE_HZ`, and while it sits still the heartbeat carries it. The
    heartbeat is not padding — the daemon stales a sample in 2 s, and a duck
    that decided the hand had left the room *because it stopped moving*,
    which is exactly what a hand does while it waits for the duck, is the
    whole feature failing at the last second.
    """
    since = now - last_at
    moved = (last_m is None
             or math.hypot(here_m[0] - last_m[0], here_m[1] - last_m[1])
             >= pet_map.SENSE_MIN_M)
    return ((moved and since >= 1.0 / pet_map.SENSE_HZ)
            or since >= pet_map.SENSE_HEARTBEAT_S)


# ---------------------------------------------------------------- the shell

def build(AppKit, Foundation, objc, smap, feed, args):
    """Define and wire the Cocoa objects. Returns the controller."""

    NSMakeRect = Foundation.NSMakeRect

    class PetWindow(AppKit.NSWindow):
        # Never key, never main: a pet that steals your cursor is spyware with
        # feathers. Clicks still arrive — see PetView.acceptsFirstMouse_.
        def canBecomeKeyWindow(self):
            return False

        def canBecomeMainWindow(self):
            return False

    class PetView(AppKit.NSView):
        def initWithFrame_(self, frame):
            self = objc.super(PetView, self).initWithFrame_(frame)
            if self is None:
                return None
            self.image = None
            self.stale = False
            self.ctl = None
            return self

        def acceptsFirstMouse_(self, event):
            # The app is never active; without this the first click on the duck
            # would be spent activating it instead of shoving it.
            return True

        def isOpaque(self):
            return False

        def drawRect_(self, rect):
            bounds = self.bounds()
            AppKit.NSColor.clearColor().set()
            AppKit.NSRectFill(bounds)
            if self.image is None:
                return
            self.image.drawInRect_fromRect_operation_fraction_(
                bounds, Foundation.NSZeroRect,
                AppKit.NSCompositingOperationSourceOver, 1.0)
            if self.stale:
                # SourceAtop tints only what is already drawn — the duck goes
                # cold, the transparent square stays transparent.
                AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    *OFFLINE_TINT).set()
                AppKit.NSRectFillUsingOperation(
                    bounds, AppKit.NSCompositingOperationSourceAtop)

        # ----- the hand: down, moving, up. What it MEANT is decided in
        # pet_map.classify_release; this view only reports events. -----

        def mouseDown_(self, event):
            if self.ctl is not None:
                self.ctl.beginDrag_(event)

        def mouseDragged_(self, event):
            if self.ctl is not None:
                self.ctl.sampleDrag_(event)

        def mouseUp_(self, event):
            if self.ctl is not None:
                self.ctl.endDrag_(event)

    class PetController(AppKit.NSObject):
        """Owns the window and the 30 Hz tick. Everything here is main-thread.

        `@objc.python_method` on everything that is not a real callback: an
        undecorated method on an NSObject subclass becomes an Objective-C
        selector, and a selector's arity has to match its name — so `setup`
        taking three arguments is a load-time error, not a runtime surprise.
        """

        @objc.python_method
        def setup(self, smap, feed, args):
            self.smap = smap
            self.feed = feed
            self.args = args
            self.seq = -1
            self.pose = {}
            self.last_good = time.time()
            self.interrupted = False
            self.drag = None
            self.click_through = True
            # The cursor channel's rate limiter: when the last sample went
            # out, where it said the pointer was (metres), and whether we
            # have already told the daemon the pointer left this screen.
            self._sense_at = 0.0
            self._sense_m = None
            self._sense_present = False
            # Is anybody in a position to see this window? Every frame the
            # feed asks for costs a ~40 ms render on the daemon's SIM thread,
            # so a duck behind a locked screen is a pinned core for nothing.
            self.awake = True
            self.timer = None
            self.observers = []

            w = smap.window_pt
            ox, oy = smap.window_origin(0.0)
            self.win = PetWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                NSMakeRect(ox, oy, w, w),
                AppKit.NSWindowStyleMaskBorderless,
                AppKit.NSBackingStoreBuffered, False)
            self.win.setLevel_(LEVELS[args.level])
            self.win.setOpaque_(False)
            self.win.setBackgroundColor_(AppKit.NSColor.clearColor())
            self.win.setHasShadow_(False)
            self.win.setIgnoresMouseEvents_(True)   # until the cursor is on the duck
            self.win.setMovableByWindowBackground_(False)
            self.win.setCollectionBehavior_(
                AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces |
                AppKit.NSWindowCollectionBehaviorStationary |
                AppKit.NSWindowCollectionBehaviorFullScreenAuxiliary)
            self.view = PetView.alloc().initWithFrame_(NSMakeRect(0, 0, w, w))
            self.view.setWantsLayer_(True)
            self.view.ctl = self
            self.win.setContentView_(self.view)
            self.win.orderFrontRegardless()
            return self

        # ----- the tick -----

        def tick_(self, timer):
            if self.interrupted:
                AppKit.NSApp().terminate_(None)
                return
            # A gesture ends on mouseUp — except when mouseUp never comes,
            # which a Space switch, Mission Control, or the system taking the
            # drag over will all do. Without this the window stays solid
            # forever and silently eats every click in the duck's box for the
            # rest of the session, because `_update_click_through` refuses to
            # let go mid-gesture.
            if self.drag is not None and AppKit.NSEvent.pressedMouseButtons() == 0:
                self.endDrag_(None)
            snap = self.feed.snapshot()
            now = time.time()
            if snap["pose"]:
                self.pose = snap["pose"]
                self._adopt_config(self.pose.get("config"))
            if snap["seq"] != self.seq and snap["png"]:
                self.seq = snap["seq"]
                self._adopt_frame(snap["png"])
            if snap["online"]:
                self.last_good = now
            elif self.seq < 0 and self.args.verbose:
                self._complain(snap["error"])
            stale = (now - self.last_good) > OFFLINE_AFTER_S
            if stale != self.view.stale:
                self.view.stale = stale
                self.view.setNeedsDisplay_(True)
                self._say("offline — the duck freezes" if stale else "back")
            self._gesture_tick(snap)
            self._place()
            self._update_click_through()
            self._send_cursor()

        @objc.python_method
        def _resize_to(self, smap):
            """Adopt a map AND make the window the size that map describes.

            These two have to travel together. `window_pt` is
            `frame_px / backing_scale`, so a display whose scale factor is 1.0
            rather than 2.0 changes the window's size without changing a
            single number the daemon knows about — and a window that does not
            match the image inside it draws the duck at the wrong size, puts
            the floor line off the Dock's edge, and centres itself on a
            half-width it does not have.
            """
            self.smap = smap
            size = Foundation.NSMakeSize(smap.window_pt, smap.window_pt)
            if abs(self.win.frame().size.width - smap.window_pt) > 0.05:
                self.win.setContentSize_(size)
                self.view.setFrameSize_(size)
                if self.view.image is not None:
                    self.view.image.setSize_(size)   # points, not pixels
                self.view.setNeedsDisplay_(True)

        @objc.python_method
        def _adopt_config(self, cfg):
            """The daemon clamps; believe it, and resize the window to match."""
            if not cfg:
                return
            new = self.smap.adopt(cfg)
            if new == self.smap:
                return
            self._resize_to(new)
            self._say("config", f"frame {new.frame_px} px ×{new.supersample} "
                                f"-> window {new.window_pt:.0f} pt | "
                                f"floor pad {new.floor_pad_px} px")

        @objc.python_method
        def _adopt_frame(self, png):
            data = Foundation.NSData.dataWithBytes_length_(png, len(png))
            img = AppKit.NSImage.alloc().initWithData_(data)
            if img is None:
                return
            # The frame arrives at device resolution (512 px for a 256 pt
            # window); telling NSImage it is 256 pt wide is what makes macOS
            # draw it 1:1 on a 2× display instead of upscaling half of it.
            img.setSize_(Foundation.NSMakeSize(self.smap.window_pt,
                                               self.smap.window_pt))
            self.view.image = img
            self.view.setNeedsDisplay_(True)

        @objc.python_method
        def _place(self):
            x_m = self.pose.get("base_x_m")
            if x_m is None:
                return
            # How far the daemon's camera has followed the duck upwards — 0
            # for a duck on the Dock, which is every duck that is not being
            # carried. When it is not 0 the window has to climb the screen by
            # exactly the same amount, or the picture rises inside a window
            # that stayed put and the duck looks like it shrank into its own
            # frame. A daemon too old to report it has not lifted anything.
            floor_z = (self.pose.get("screen") or {}).get("frame_floor_z_m")
            ox, oy = self.smap.window_origin(float(x_m), float(floor_z or 0.0))
            frame = self.win.frame()
            if abs(frame.origin.x - ox) > 0.05 or abs(frame.origin.y - oy) > 0.05:
                self.win.setFrameOrigin_(Foundation.NSMakePoint(ox, oy))

        @objc.python_method
        def _update_click_through(self):
            """Decide, 30 times a second, whether this window is solid.

            There is no cheaper way to be a shaped window on macOS: the cursor
            is polled rather than tracked because tracking areas need the
            window to already own the event we are trying to decide about.

            A window with no duck in it is NEVER solid. `hit_rect_pt` falls
            back to the nominal standing box when there is no pose, so without
            this gate a duck-pet with no daemon behind it (wrong port, wrong
            scene, sim not started) would sit invisible over the middle of the
            Dock eating icon clicks — ~177x196 pt of dead zone with nothing
            drawn to explain it. Stale counts too: a frozen duck is a picture
            of where the duck WAS, and shoving that is meaningless.
            """
            if self.args.click_through:
                return
            if self.drag is not None:
                return          # never let go mid-gesture
            over = False
            why = "no frame yet"
            if self.seq >= 0 and not self.view.stale and self.awake:
                p = AppKit.NSEvent.mouseLocation()
                f = self.win.frame()
                lx, ly = p.x - f.origin.x, p.y - f.origin.y
                # The ball counts as something to be solid over. It is drawn
                # in this window, it can be shoved from this window, and a
                # transparent square that let a click through onto a Dock
                # icon while a duck's toy was sitting right there would be a
                # ball you can see and cannot touch. `press_target` is the
                # one place that decision lives, and `beginDrag_` asks it the
                # same question — they must never disagree.
                what = press_target(self.smap, self.pose, lx, ly)
                over = what != "none"
                x0, y0, x1, y1 = hit_rect_pt(self.smap, self.pose)
                why = (f"cursor ({lx:.0f}, {ly:.0f}) on {what} | duck "
                       f"({x0:.0f}, {y0:.0f})-({x1:.0f}, {y1:.0f})")
            if over == self.click_through:
                self.click_through = not over
                self.win.setIgnoresMouseEvents_(not over)
                self._say("solid" if over else "click-through", why)

        @objc.python_method
        def _say(self, *parts):
            """Verbose tracing for the things that are hard to see in a
            screenshot: when the window becomes solid, what a gesture turned
            into, and why there is no duck."""
            if self.args.verbose:
                print("duck-pet:", " | ".join(str(p) for p in parts), flush=True)

        @objc.python_method
        def _complain(self, error):
            """Say the daemon's own words, once, while there is still no frame."""
            if error and error != getattr(self, "_last_complaint", None):
                self._last_complaint = error
                self._say("no frame yet", error)

        # ----- poke / pet / shove -----

        def beginDrag_(self, event):
            """The button went down. Record where, and decide what it is ON.

            The target is decided ONCE, here, and never revisited: a press
            that started on the duck is about the duck even if the hand
            wanders off it, the same way a scrollbar keeps a drag that leaves
            the scrollbar. There are two things to press on now, and
            `press_target` is the only place that question is answered —
            `_update_click_through` asks it too, and a window that was solid
            over the ball while this read "duck" would turn a click on the toy
            into a shove of the animal.

            A press on the BALL can only ever be a push. `carry_off` is
            latched here so the hold can never promote, and the release
            classifier is skipped for it below: you do not pet a ball and you
            do not pick one up. That is not a limitation, it is what makes the
            toy read as a toy.
            """
            p = AppKit.NSEvent.mouseLocation()
            loc = event.locationInWindow()
            target = press_target(self.smap, self.pose, loc.x, loc.y)
            if target == "none":
                target = "duck"     # solid but on neither: treat as the duck
            # Which side of the TOY the press landed on, decided here and not
            # at release. `loc.x` is window-local and this window travels with
            # the walking duck — 0.26 m/s is 170 pt/s, so a 100 ms click moves
            # the frame ~17 pt while the ball's clickable half-width is only
            # 31 pt. Comparing a press recorded in one frame against a centre
            # measured in another can flip the sign and roll the toy TOWARDS
            # the finger. Both numbers now come from one instant. (The duck
            # needs none of this: it is pinned to the window's own centre, so
            # the pairing is exact whenever the window is.)
            ball_mid = None
            if target == "ball":
                rect = ball_rect_pt(self.smap, self.pose)
                ball_mid = 0.5 * (rect[0] + rect[2]) if rect else None
            self.drag = {
                "t0": time.time(),
                "x0": p.x, "y0": p.y,
                "in_win": loc.x, "in_win_y": loc.y,
                "target": target,
                "ball_mid": ball_mid,
                # The carry lane's two: a press that stays still long enough
                # becomes a pick-up, and `carry_off` is the latch that says it
                # never can. They are maintained HERE rather than there
                # because `sampleDrag_` is the only place that sees every
                # mouse move — a promotion decided anywhere else would be
                # deciding on a hand it did not watch.
                "mode": "undecided",
                "carry_off": target == "ball",
                # The carry's own bookkeeping: when the hand last restated
                # itself to the daemon, and whether the daemon has ever
                # confirmed the grip. Both are only read once `mode` is
                # "carry" (see `_gesture_tick`).
                "last_carry_at": 0.0,
                "carry_seen": False,
                "samples": [(time.time(), p.x, p.y)],
            }

        def sampleDrag_(self, event):
            if self.drag is None:
                return
            p = AppKit.NSEvent.mouseLocation()
            d = self.drag
            s = d["samples"]
            s.append((time.time(), p.x, p.y))
            del s[:-24]
            if drag_left_the_spot(d, p.x, p.y):
                d["carry_off"] = True
            if d["mode"] == "carry":
                # A moving hand should not wait for the next 30 Hz tick to be
                # heard. `_carry_move` is rate-limited to CARRY_HZ either way,
                # so this only ever makes the grip more responsive — it never
                # makes it chattier.
                self._carry_move()

        # ----- the fourth gesture: a hold that becomes a pick-up -----

        @objc.python_method
        def _gesture_tick(self, snap):
            """Promote a still hold into a carry, and keep a live one alive.

            This runs off the 30 Hz timer rather than off `sampleDrag_`, and
            that is the whole reason it exists as a separate method:
            `mouseDragged_` does not fire when the mouse does not move, so a
            perfectly still press — which is exactly what a pick-up is made
            of — produces no events at all after the button goes down. A
            promotion decided in the sampler would simply never happen.

            Keeping a live carry alive is not optional either. The daemon
            releases the weld after 1.5 s of silence (its deadman against a
            crashed overlay), so a grip that stopped talking because the hand
            stopped moving would put the duck down by itself.
            """
            d = self.drag
            if d is None:
                return
            if d["mode"] == "carry":
                carrying = bool(snap.get("carrying"))
                if carry_was_lost(d, carrying):
                    # The feed's token went away: a reconnect, or the daemon's
                    # own deadman. Whatever this window thinks, the duck is
                    # back on the floor — so the gesture is over, and the next
                    # mouse-up must not send an `end` for a grip nobody holds.
                    self._say("carry", "lost — the daemon let go")
                    self.drag = None
                    return
                if carrying:
                    d["carry_seen"] = True
                self._carry_move()
                return
            if promote_to_carry(d, time.time()):
                self._begin_carry()

        @objc.python_method
        def _carry_point(self) -> dict:
            """Where the pointer is, in the duck's own metres."""
            p = AppKit.NSEvent.mouseLocation()
            return pet_map.screen_to_sim_m(self.smap, p.x, p.y)

        @objc.python_method
        def _begin_carry(self):
            d = self.drag
            d["mode"] = "carry"
            d["last_carry_at"] = time.time()
            d["carry_seen"] = False
            self.feed.carry_start(**self._carry_point())
            self._say("carry", f"held still for "
                               f"{pet_map.CARRY_HOLD_S:.2f} s — picking it up")

        @objc.python_method
        def _carry_move(self):
            d = self.drag
            now = time.time()
            if now - d["last_carry_at"] < 1.0 / pet_map.CARRY_HZ:
                return
            d["last_carry_at"] = now
            self.feed.carry_move(**self._carry_point())

        @objc.python_method
        def _end_carry(self):
            self.feed.carry_end()
            self._say("carry", "let go")

        def endDrag_(self, event):
            """The button came up — and only now is the gesture named.

            `release_kind` does the naming from five numbers: how long the
            hand was down, how far it went in total, how fast it was still
            going at the end, and whether it let go on the animal. A poke and
            a shove both land in qvel; a pet deliberately does not touch the
            physics at all, and a carry has already been answered by the weld.
            Everything below this line is plumbing — measuring the tail,
            sending the result.
            """
            d, self.drag = self.drag, None
            if d is None:
                return
            p = AppKit.NSEvent.mouseLocation()
            now = time.time()
            total_dx, total_dy = p.x - d["x0"], p.y - d["y0"]
            # A flick is its last few milliseconds, not its average: a slow
            # drag that ends in a snap should shove like a snap. The same tail
            # is what tells a stroke from a throw, so it is measured for every
            # gesture rather than only for the ones already known to be drags.
            ref = d["samples"][0]
            for sample in d["samples"]:
                if now - sample[0] <= pet_map.FLICK_WINDOW_S:
                    break
                ref = sample
            tail_dx, tail_dy = p.x - ref[1], p.y - ref[2]
            kind = release_kind(d, now, total_dx, total_dy, tail_dx, tail_dy)
            if kind in ("carry", "carry_pet"):
                # A pick-up has already happened; there is nothing left to
                # classify. Let go, and the daemon hands the duck whatever
                # velocity the hand had — a flung duck flies, a duck set down
                # gently does not, and neither of those is this app's decision.
                self._end_carry()
                # ...unless the hand never went anywhere and let go almost at
                # once. That was not a pick-up, it was a hand RESTING on a
                # duck, and the kindest reading of it is a pet. The daemon has
                # already put the duck back down by the time this lands.
                if kind == "carry_pet":
                    base_x = self.pose.get("base_x_m") or 0.0
                    self.feed.touch(pet_map.touch_payload(
                        self.smap, p.x, p.y, base_x_m=base_x,
                        duration_s=now - d["t0"],
                        travel_pt=math.hypot(total_dx, total_dy)))
                    self._say("pet", "a hand that rested and let go, not a lift")
                return
            if kind == "pet":
                base_x = self.pose.get("base_x_m") or 0.0
                self.feed.touch(pet_map.touch_payload(
                    self.smap, p.x, p.y, base_x_m=base_x,
                    duration_s=now - d["t0"],
                    travel_pt=math.hypot(total_dx, total_dy)))
                self._say("pet", f"{now - d['t0']:.2f} s, "
                                 f"{math.hypot(total_dx, total_dy):.0f} pt, "
                                 f"tail {math.hypot(tail_dx, tail_dy):.0f} pt")
                return
            if kind == "poke":
                # A poke shoves away from the finger, measured about the
                # thing's own centre: the window's middle for the duck (which
                # is always centred in its frame), and the ball's rectangle
                # for the ball (which is wherever it rolled to). The toy's
                # centre is the one recorded in `beginDrag_` — the same
                # instant, and therefore the same window frame, as the press
                # it is being compared with. See there for what a window that
                # moved in between does to the sign.
                push = pet_map.poke_to_push(d["in_win"], self.smap.window_pt,
                                            center_pt=d.get("ball_mid"))
            else:
                push = pet_map.drag_to_push(tail_dx, tail_dy,
                                            self.smap.px_per_meter)
            push = {**push, "target": d["target"]}
            self._say(kind, d["target"],
                      f"{push['dx_m']:+.3f}, {push['dy_m']:+.3f} m "
                      f"-> ~{pet_map.push_speed_mps(push):.2f} m/s")
            self.feed.push(push)

        # ----- where the hand is when it is not touching anything -----

        @objc.python_method
        def _send_cursor(self):
            """Tell the daemon where the mouse pointer is, in the duck's metres.

            This is the only thing the duck can perceive that is not physics,
            and it is deliberately the cheapest channel in the app: five posts
            a second while the pointer moves, one a second while it sits
            still, and nothing at all while the screen is asleep. The daemon
            spends no render on it (`sim_server._handle_pet_sense`), so the
            entire cost is one queued 50 Hz tick.

            The heartbeat is not padding. A cursor sample goes stale in two
            seconds on the daemon's side, and a duck that walked over to a
            pointer only to decide it had left the room — because the hand
            stopped moving, which is exactly what a hand does when it is
            waiting for the duck — would be the whole feature failing at the
            last second.
            """
            if not self.awake:
                return          # a locked screen has no pointer worth reporting
            if self.drag is not None and self.drag["mode"] == "carry":
                # The carry channel already carries the pointer, at four times
                # this rate. Two channels saying where the same mouse is would
                # be two sim ticks spent on one fact — and the duck's `cursor.*`
                # guards have nothing to say about a hand that is holding it.
                return
            p = AppKit.NSEvent.mouseLocation()
            now = time.time()
            if not self._on_this_screen(p.x, p.y):
                # Said once, not every tick: "gone" is one fact.
                if self._sense_present:
                    self._sense_present = False
                    self._sense_at = now
                    self.feed.sense({"present": False})
                return
            body = pet_map.cursor_payload(self.smap, p.x, p.y)
            if not cursor_due(now, self._sense_at, self._sense_m,
                              (body["x_m"], body["z_m"])):
                return
            self._sense_at = now
            self._sense_m = (body["x_m"], body["z_m"])
            self._sense_present = True
            # Stamped with THIS clock, at measurement time: the daemon
            # computes `cursor.speed_mps` from consecutive `t_s` deltas,
            # because its own arrival times are queue-compressed behind
            # renders (see `_handle_pet_sense`) and read a wiggling hand as
            # anything from 6x too fast to standing still.
            body["t_s"] = time.monotonic()
            self.feed.sense(body)

        @objc.python_method
        def _on_this_screen(self, x_pt, y_pt) -> bool:
            """Is the pointer on the display the duck lives on?

            `ScreenMap` carries the usable band and the screen's height, which
            is the whole test on a one-display Mac (Cocoa's global space puts
            the main screen's origin at 0,0). On a second display to the side
            the x test already answers correctly; one stacked above or below
            would need the screen's own origin, which the map does not carry —
            and the cost of getting it wrong is a `present: False` the next
            sample immediately corrects.
            """
            s = self.smap
            return (s.left_pt <= x_pt <= s.left_pt + s.width_pt
                    and 0.0 <= y_pt <= s.screen_h_pt)

        # ----- nobody is looking -----

        @objc.python_method
        def _set_awake(self, awake, why):
            """Throttle the frame stream when the window cannot be seen.

            The expensive half is the ask, not the tick: every frame the feed
            requests costs a ~40 ms render on the daemon's SIM thread, and 30
            Hz of comparing two floats on this side costs nothing. So the feed
            drops to one frame a second and the UI timer keeps running — it is
            what makes SIGINT reachable, and what puts the window back where
            it belongs the instant the screen lights up. Not zero fps either:
            one a second is what notices the daemon coming back.
            """
            awake = bool(awake)
            if awake == self.awake:
                return
            self.awake = awake
            self.feed.set_fps(self.args.fps if awake else pet_feed.IDLE_FPS)
            self._say("awake" if awake else "asleep",
                      f"{why} -> "
                      f"{self.args.fps if awake else pet_feed.IDLE_FPS:.0f} fps")
            if awake:
                self._place()

        def sleepStarted_(self, note):
            self._set_awake(False, str(note.name()))

        def sleepEnded_(self, note):
            self._set_awake(True, str(note.name()))

        def occlusionChanged_(self, note):
            # The collection behaviour this window uses (all Spaces,
            # stationary) means macOS rarely calls it occluded, so this is the
            # cheap half of the story and the workspace notifications are the
            # half that fires on a laptop. Both land here.
            visible = bool(self.win.occlusionState() &
                           AppKit.NSWindowOcclusionStateVisible)
            self._set_awake(visible, "occlusion")

        # ----- the world moved -----

        def screensChanged_(self, note):
            """A display was added, resized, or the Dock changed size/side.

            The walls are placed in metres from a screen width in pixels, so
            they are wrong the instant either changes — rebuild and renegotiate
            rather than let the duck walk off a screen that just got smaller.

            Everything here is inside a try. This notification is posted on
            display sleep/wake and hot-unplug, which is exactly when
            `NSScreen.screens()` can come back empty and `screen_map_for`
            raises `SystemExit` — and a Python exception cannot unwind through
            an ObjC callback: pyobjc turns it into an NSException, which in the
            run loop is an abort, not an orderly exit. A screen list we cannot
            read is a reason to keep the map we have.
            """
            try:
                new = screen_map_for(AppKit, self.args)
            except (SystemExit, Exception) as e:      # noqa: BLE001
                self._say("screens changed", f"ignored: {e}")
                return
            # Resize unconditionally, not through `_adopt_config`: that one
            # short-circuits on an unchanged map, and the daemon echoes back
            # exactly what it was posted, so a backing-scale change would
            # never reach the window otherwise.
            self._resize_to(new)
            self.feed.set_config(self.smap.config_payload(),
                                 self.smap.frame_px, self.smap.supersample)
            self._say("screens changed", f"band {self.smap.width_pt:.0f} pt, "
                                         f"floor y {self.smap.floor_y_pt:.0f} pt, "
                                         f"window {self.smap.window_pt:.0f} pt")
            self._place()

        def quit_(self, timer):
            AppKit.NSApp().terminate_(None)

        # ----- the end -----

        def willTerminate_(self, note):
            """The only teardown that ever runs.

            `NSApp().terminate_()` exits in C and never returns to Python, so
            anything after `app.run()` is dead code — the feed thread and its
            keep-alive connection would only ever be torn down by the process
            dying, leaving the daemon's handler thread waiting on a socket
            EOF. This is the hook that actually fires, so it is where the
            timer stops, the observers come off (the notification centre holds
            them unretained; a controller freed with one still registered is a
            dangling pointer waiting for the next screen change) and the
            socket closes.
            """
            if self.timer is not None:
                self.timer.invalidate()
                self.timer = None
            for center in self.observers:
                center.removeObserver_(self)
            self.observers = []
            try:
                self.feed.stop()
                self._say("shutdown", "timer stopped, observers off, feed closed")
            except Exception as e:                    # noqa: BLE001
                self._say("shutdown", f"feed.stop: {e}")

    return PetController.alloc().init().setup(smap, feed, args)


def run(args) -> int:
    AppKit, Foundation, objc = _import_cocoa()

    app = AppKit.NSApplication.sharedApplication()
    # Accessory: no Dock icon, no Cmd-Tab entry, no menu bar of its own.
    app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)

    smap = screen_map_for(AppKit, args)
    if args.mock:
        from . import pet_mock
        # A mock that binds duck-sim's own port would either lose the race
        # with a running daemon or win it, and both are worse than a port of
        # its own. An explicit --port still wins.
        if args.port is None:
            args.port = pet_mock.MOCK_PORT
        # ...and on loopback, whatever --host says: --host is where duck-pet
        # connects TO, never an address to publish a stand-in daemon on.
        pet_mock.start_mock(args.port)
        args.host = "127.0.0.1"
    if args.port is None:
        args.port = pet_feed.DEFAULT_PORT

    feed = pet_feed.PetFeed(host=args.host, port=args.port, fps=args.fps,
                            config=smap.config_payload(),
                            frame_px=smap.frame_px,
                            supersample=smap.supersample).start()
    ctl = build(AppKit, Foundation, objc, smap, feed, args)

    if args.verbose:
        print(f"duck-pet: screen band {smap.width_pt:.0f} pt "
              f"({smap.span_m:.2f} m) | floor y {smap.floor_y_pt:.0f} pt | "
              f"{smap.px_per_meter:.0f} pt/m -> duck {smap.duck_pt:.0f} pt | "
              f"walls ≈±{smap.half_span_m:.3f} m | frame {smap.frame_px} px "
              f"×{smap.supersample} -> window {smap.window_pt:.0f} pt | "
              f"level {args.level}={LEVELS[args.level]} | "
              f"http://{args.host}:{args.port}/pet/",
              flush=True)   # NSApp.terminate_ exits in C; nothing flushes later

    # A Python-level flag rather than a raised KeyboardInterrupt: the ObjC run
    # loop never unwinds through Python, but the 30 Hz tick gives the
    # interpreter somewhere to notice the signal ran.
    def _sigint(_sig, _frame):
        ctl.interrupted = True
    signal.signal(signal.SIGINT, _sigint)

    timer = AppKit.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
        1.0 / UI_HZ, ctl, "tick:", None, True)
    # A tolerance of 0 — the default — opts the process out of macOS timer
    # coalescing for its ONLY wakeup source, so the kernel has to wake it on an
    # exact 33 ms cadence and can never align it with anything else. The tick's
    # job (apply the last frame's position, re-decide click-through) does not
    # care about a few ms of jitter.
    timer.setTolerance_(0.5 / UI_HZ)
    ctl.timer = timer
    # Common modes, so the duck keeps walking while a menu is open.
    Foundation.NSRunLoop.currentRunLoop().addTimer_forMode_(
        timer, Foundation.NSRunLoopCommonModes)
    if args.quit_after > 0:
        quitter = AppKit.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            args.quit_after, ctl, "quit:", None, False)
        quitter.setTolerance_(min(0.25, 0.05 * args.quit_after))
    center = AppKit.NSNotificationCenter.defaultCenter()
    center.addObserver_selector_name_object_(
        ctl, "screensChanged:",
        AppKit.NSApplicationDidChangeScreenParametersNotification, None)
    center.addObserver_selector_name_object_(
        ctl, "willTerminate:",
        AppKit.NSApplicationWillTerminateNotification, None)
    center.addObserver_selector_name_object_(
        ctl, "occlusionChanged:",
        AppKit.NSWindowDidChangeOcclusionStateNotification, ctl.win)
    ctl.observers.append(center)
    # The ones that actually fire on a laptop: this window joins all Spaces
    # and is stationary, so macOS rarely calls it occluded — but the display
    # going to sleep, or the screen locking / fast-user-switching away, are
    # notifications about the whole session and they arrive reliably. They
    # live on NSWorkspace's OWN centre, not the default one.
    ws = AppKit.NSWorkspace.sharedWorkspace().notificationCenter()
    for name, sel in ((
            "NSWorkspaceScreensDidSleepNotification", "sleepStarted:"), (
            "NSWorkspaceScreensDidWakeNotification", "sleepEnded:"), (
            "NSWorkspaceSessionDidResignActiveNotification", "sleepStarted:"), (
            "NSWorkspaceSessionDidBecomeActiveNotification", "sleepEnded:")):
        note = getattr(AppKit, name, None)
        if note is not None:      # a future macOS may retire one; skip it
            ws.addObserver_selector_name_object_(ctl, sel, note, None)
    ctl.observers.append(ws)

    app.run()
    # Not reached: NSApp.terminate_ exits in C. The teardown that does run is
    # `PetController.willTerminate_`, on the notification registered above.
    return 0


def add_arguments(p):
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=None,
                   help=f"the duck-sim's --web port (default "
                        f"{pet_feed.DEFAULT_PORT}, which is also duck-sim's "
                        f"own default; --mock uses its own port instead)")
    p.add_argument("--ppm", type=float, default=None, dest="ppm",
                   help="points per metre; overrides --duck-pt")
    p.add_argument("--duck-pt", type=float, default=pet_map.DEFAULT_DUCK_PT,
                   help="how tall the duck should stand, in points (default 180)")
    p.add_argument("--window-pt", type=float, default=pet_map.DEFAULT_WINDOW_PT,
                   help="wanted overlay window edge, points (default 300) — "
                        "trimmed to whatever the daemon's 512 px frame cap "
                        "allows at this display's scale factor")
    p.add_argument("--frame-px", type=int, default=None,
                   help="pin the rendered frame size in device pixels "
                        f"(32-{pet_map.PET_FRAME_MAX_PX}) instead of deriving "
                        "it from --window-pt")
    p.add_argument("--floor-pad-px", type=int,
                   default=pet_map.DEFAULT_FLOOR_PAD_PX,
                   help="device pixels of frame below the sim floor line — "
                        "landing room under the feet (default 26)")
    p.add_argument("--floor-offset-pt", type=float, default=0.0,
                   help="nudge the walk line off the Dock's top edge, points "
                        "(negative = feet sunk into the Dock)")
    p.add_argument("--screen", type=int, default=0, help="NSScreen index")
    p.add_argument("--fps", type=float, default=pet_feed.DEFAULT_FPS,
                   help=f"frame requests per second, capped at {pet_feed.MAX_FPS:.0f} "
                        "— rendering runs on the sim thread and ~40 fps stops "
                        "the physics dead")
    p.add_argument("--level", choices=sorted(LEVELS), default=DEFAULT_LEVEL,
                   help="window level; 'status' (25) is the one that beats the "
                        "Dock without covering menus")
    p.add_argument("--click-through", action="store_true",
                   help="never take a click; the duck becomes scenery")
    p.add_argument("--mock", action="store_true",
                   help="serve frames from a built-in stand-in daemon on "
                        "--port, for testing the window with no sim running")
    p.add_argument("--quit-after", type=float, default=0.0,
                   help="exit after N seconds (for scripted screenshots)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main():
    p = add_arguments(argparse.ArgumentParser(
        prog="duck-pet",
        description="the duck, walking along the top of your Dock"))
    sys.exit(run(p.parse_args()))


if __name__ == "__main__":
    main()
