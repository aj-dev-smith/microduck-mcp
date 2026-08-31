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

Threading: Cocoa's main thread only draws and moves the window; every socket
lives on `pet_feed.PetFeed`'s background thread.

    duck-sim --scene desktop       # headless; owns the default socket, port 8400
    duck machine load machines/pet.toml && duck machine arm
    duck-pet                       # both defaults are 8400: no flags needed

    duck-pet --mock                # against a stand-in daemon, no sim needed
    duck-pet --port 8410 --duck-pt 180 --fps 20 -v      # against a dev daemon
"""

import argparse
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
    """
    w = smap.window_pt
    bbox = (pose or {}).get("bbox")
    if (isinstance(bbox, (list, tuple)) and len(bbox) == 4
            and bbox[2] > bbox[0] and bbox[3] > bbox[1]):
        s = w / float(smap.frame_px)
        x0, y0, x1, y1 = (float(v) * s for v in bbox)
        # bbox counts down from the top (image order); Cocoa counts up.
        return (x0 - HIT_PAD_PT, (w - y1) - HIT_PAD_PT,
                x1 + HIT_PAD_PT, (w - y0) + HIT_PAD_PT)
    half = pet_map.DUCK_DEPTH_M * smap.px_per_meter   # beak and tail included
    foot = smap.ground_pt
    return (0.5 * w - half - HIT_PAD_PT, foot - HIT_PAD_PT,
            0.5 * w + half + HIT_PAD_PT, foot + smap.duck_pt + HIT_PAD_PT)


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

        # ----- the only two gestures the pet has -----

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
            self._place()
            self._update_click_through()

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
            ox, oy = self.smap.window_origin(float(x_m))
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
                x0, y0, x1, y1 = hit_rect_pt(self.smap, self.pose)
                over = (x0 <= lx <= x1) and (y0 <= ly <= y1)
                why = (f"cursor ({lx:.0f}, {ly:.0f}) vs duck "
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

        # ----- drag / poke -----

        def beginDrag_(self, event):
            p = AppKit.NSEvent.mouseLocation()
            self.drag = {"x0": p.x, "y0": p.y,
                         "in_win": event.locationInWindow().x,
                         "samples": [(time.time(), p.x, p.y)]}

        def sampleDrag_(self, event):
            if self.drag is None:
                return
            p = AppKit.NSEvent.mouseLocation()
            s = self.drag["samples"]
            s.append((time.time(), p.x, p.y))
            del s[:-24]

        def endDrag_(self, event):
            d, self.drag = self.drag, None
            if d is None:
                return
            p = AppKit.NSEvent.mouseLocation()
            total_dx, total_dy = p.x - d["x0"], p.y - d["y0"]
            if pet_map.is_click(total_dx, total_dy):
                push = pet_map.poke_to_push(d["in_win"], self.smap.window_pt)
                kind = "poke"
            else:
                # A flick is its last few milliseconds, not its average: a slow
                # drag that ends in a snap should shove like a snap.
                now = time.time()
                ref = d["samples"][0]
                for sample in d["samples"]:
                    if now - sample[0] <= pet_map.FLICK_WINDOW_S:
                        break
                    ref = sample
                push = pet_map.drag_to_push(p.x - ref[1], p.y - ref[2],
                                            self.smap.px_per_meter)
                kind = "drag"
            self._say(kind, f"{push['dx_m']:+.3f}, {push['dy_m']:+.3f} m "
                            f"-> ~{pet_map.push_speed_mps(push):.2f} m/s")
            self.feed.push(push)

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
