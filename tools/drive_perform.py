"""Drive perform mode's real two-context path.

A headless test cannot open a second window at all, and a render test drives a
bare mixin - so a shared-context failure, a stale external wrapper or a viewport
left on the wrong rect passes the whole suite and shows up on a projector.

It deliberately does NOT boot the real App: an App-based driver writes the
user's audio_rig.json and preferences.config on exit.

    python -m tools.drive_perform
    python -m tools.drive_perform --monitor 1 --seconds 5
"""
from __future__ import annotations

import argparse
import sys
import time

# `ui` first: services/__init__ imports config_saver, which imports ui, so
# reaching a services submodule before ui is a partially-initialised package.
# The same order tools/drive_field_stack.py uses.
import ui  # noqa: F401

import glfw                                           # noqa: E402
import moderngl                                       # noqa: E402
import numpy as np                                    # noqa: E402

from camera import DisplayFrame                       # noqa: E402
from services.perform_window import (                 # noqa: E402
    PerformWindow, choose_monitor, fit_rect, list_monitors)

MAIN_SIZE = (640, 480)


def _make_source(ctx, size):
    """A texture whose every texel is non-black, in the assembled format."""
    w, h = size
    xs = np.linspace(0.2, 1.0, w, dtype=np.float32)[None, :, None]
    ys = np.linspace(0.2, 1.0, h, dtype=np.float32)[:, None, None]
    rgba = np.repeat(xs * ys, 4, axis=2).astype(np.float32)
    rgba[:, :, 3] = 1.0
    tex = ctx.texture(size, 4, rgba.tobytes(), dtype='f4')
    tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    return tex


def _checker(expect_src_aspect, out):
    """An on_drawn hook: read the frame back and check where the image is.

    Runs before the swap, which is the only moment the drawn frame is readable
    - after a swap the back buffer holds the other frame, and a check placed
    after it reports the PREVIOUS phase's image. Appends a problem to `out`,
    or nothing when the frame is as expected.
    """

    def check(ctx, drawn_rect):
        fb_w, fb_h = ctx.screen.size
        raw = ctx.screen.read(components=3, dtype='f1')
        img = np.frombuffer(raw, dtype=np.uint8).reshape(fb_h, fb_w, 3)
        problem = _inspect(img, expect_src_aspect, (fb_w, fb_h), drawn_rect)
        if problem:
            out.append(problem)

    return check


def _inspect(img, expect_src_aspect, fb_size, drawn_rect) -> str:
    """Where the image should be, computed independently of what was drawn."""
    fb_w, fb_h = fb_size
    x, y, w, h = fit_rect(expect_src_aspect, (fb_w, fb_h))
    if (x, y, w, h) != tuple(drawn_rect):
        return f"drew into {tuple(drawn_rect)}, expected {(x, y, w, h)}"
    inside = img[y:y + h, x:x + w]
    if not inside.any():
        return f"nothing drawn inside the fitted rect {(x, y, w, h)}"

    bars = []
    if y > 0:
        bars.append(img[:y])
        bars.append(img[y + h:])
    if x > 0:
        bars.append(img[:, :x])
        bars.append(img[:, x + w:])
    for bar in bars:
        if bar.size and bar.any():
            return "a letterbox bar is not black"
    return ""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--monitor", default="",
                    help="display key, name, or list index to perform on")
    ap.add_argument("--seconds", type=float, default=2.0,
                    help="how long to hold each phase on screen")
    args = ap.parse_args(argv)

    if not glfw.init():
        print("glfw init failed")
        return 1
    main_window = glfw.create_window(*MAIN_SIZE, "drive_perform", None, None)
    if not main_window:
        glfw.terminate()
        print("main window creation failed")
        return 1
    glfw.make_context_current(main_window)
    glfw.swap_interval(1)
    ctx = moderngl.create_context()

    monitors = list_monitors()
    for m in monitors:
        print(f"  {m.label()}  at ({m.x}, {m.y})")
    want = args.monitor
    if want.isdigit() and int(want) < len(monitors):
        want = monitors[int(want)].key
    monitor, notice = choose_monitor(monitors, want)
    if monitor is None:
        print(notice)
        glfw.terminate()
        return 1
    if notice:
        print(notice)
    print(f"performing on: {monitor.label()}")

    failures = []
    pw = PerformWindow(main_window)
    tex = _make_source(ctx, MAIN_SIZE)
    tex2 = None
    try:
        pw.open(monitor)
        assert pw.is_open, "open() reported no window"

        # Phase 1: a source shaped like the main window.
        # Phase 2: a DIFFERENT texture object, which must rebuild the external
        #          wrapper - FrameAssembler swaps its accumulation texture
        #          whenever the sample count or the canvas size moves.
        # Phase 3: a close and a reopen, the panel's monitor change.
        tex2 = _make_source(ctx, (400, 400))
        phases = [
            ("same-shaped source", tex, MAIN_SIZE),
            ("re-wrapped square source", tex2, (400, 400)),
        ]
        for name, source, src_size in phases:
            frame = DisplayFrame(texture=source, cam_pos=(0.0, 0.0),
                                 cam_zoom=1.0, tex_size=source.size,
                                 window_size=src_size)
            found = []
            deadline = time.time() + args.seconds
            checked = False
            while time.time() < deadline:
                glfw.poll_events()
                ctx.screen.use()
                ctx.clear(0.05, 0.0, 0.1, 1.0)
                glfw.swap_buffers(main_window)
                # Check the SECOND frame: the first draw after an open has
                # nothing behind it to prove the bars were cleared rather
                # than never written.
                hook = None if checked else _checker(
                    src_size[0] / src_size[1], found)
                pw.draw(frame, on_drawn=hook)
                if hook is not None:
                    checked = True
            failures.extend(f"{name}: {p}" for p in found)
            if not found:
                print(f"  {name}: image landed in the fitted rect, bars black")
            if glfw.window_should_close(main_window):
                break

        pw.close()
        assert not pw.is_open, "close() left the window open"
        pw.close()                       # idempotent
        pw.open(monitor)
        frame = DisplayFrame(texture=tex, cam_pos=(0.0, 0.0), cam_zoom=1.0,
                             tex_size=tex.size, window_size=MAIN_SIZE)
        found = []
        for i in range(30):
            glfw.poll_events()
            glfw.swap_buffers(main_window)
            hook = _checker(MAIN_SIZE[0] / MAIN_SIZE[1], found) if i == 29 else None
            pw.draw(frame, on_drawn=hook)
        failures.extend(f"after reopen: {p}" for p in found)
        if not found:
            print("  reopened and drew again")

        # A closed window must be a no-op, not a crash.
        pw.close()
        pw.draw(frame)
        pw.draw(None)
        print("  draw() on a closed window is a no-op")
    finally:
        pw.close()
        for obj in (tex, tex2):
            if obj is not None:
                obj.release()
        glfw.destroy_window(main_window)
        glfw.terminate()

    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
