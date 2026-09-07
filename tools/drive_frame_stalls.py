"""Drive the real app and report every slow frame with a profile of it.

    python -m tools.drive_frame_stalls            # menu hovers and the browser
    python -m tools.drive_frame_stalls --refit    # also force a UMAP refit

Boots the app with the user's own windows restored, then opens File > Load by
injected mouse events, hovers presets of every brain, scrolls the archive
gallery, hovers and clicks entries, drives the map, and optionally forces a
map refit while the frame loop is timed. Every frame runs under cProfile and
any over SLOW_MS prints its top entries and the phase split the frame watch
recorded.

Never calls cleanup(): the user's app may be running and nothing here may
write shared user data. The refit's cache write is redirected to a temp file.
A `__main__` guard is required, because the map's worker process re-imports
the launching script.
"""
from __future__ import annotations

import cProfile
import io
import os
import pstats
import sys
import tempfile
import time
from pathlib import Path

from ui import ini_path

ini_path.suppress()

import glfw                                          # noqa: E402
from imgui_bundle import imgui                       # noqa: E402

SLOW_MS = 80.0
RECTS: dict = {}


def _rec(key: str) -> None:
    mn, mx = imgui.get_item_rect_min(), imgui.get_item_rect_max()
    RECTS[key] = (mn.x, mn.y, mx.x, mx.y)


def _hook_rects() -> None:
    sel, bm, bti = imgui.selectable, imgui.begin_menu, imgui.begin_tab_item

    def selectable(label, *a, **k):
        r = sel(label, *a, **k)
        _rec(label)
        return r

    def begin_menu(label, *a, **k):
        r = bm(label, *a, **k)
        _rec("menu:" + label)
        return r

    def begin_tab_item(label, *a, **k):
        r = bti(label, *a, **k)
        _rec("tab:" + label)
        return r

    imgui.selectable, imgui.begin_menu, imgui.begin_tab_item = (
        selectable, begin_menu, begin_tab_item)


class Driver:
    def __init__(self, app):
        self.app = app
        self.io = imgui.get_io()
        self.n = 0
        self.slow: list = []
        self.root = str(Path(__file__).resolve().parent.parent) + os.sep

    def frame(self, tag: str = "") -> float:
        self.n += 1
        pr = cProfile.Profile()
        t = time.perf_counter()
        pr.enable()
        glfw.poll_events()
        self.app.orchestrate_frame()
        glfw.swap_buffers(self.app.window)
        pr.disable()
        dt = time.perf_counter() - t
        if dt * 1000 > SLOW_MS:
            self.slow.append((self.n, dt, tag))
            watched = self.app.frame_watch.recent
            split = watched[-1].line() if watched else ""
            print(f"\n### slow frame #{self.n} {dt*1000:.0f} ms  [{tag}]  {split}")
            s = io.StringIO()
            pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(24)
            shown = 0
            for line in s.getvalue().splitlines():
                if "fluoddity" in line and "site-packages" not in line:
                    print("   ", line.replace(self.root, "")[:150])
                    shown += 1
                    if shown >= 10:
                        break
        return dt

    def idle(self, n: int, tag: str) -> None:
        ft = sorted(self.frame(tag) for _ in range(n))
        print(f"[{tag}] {n} frames: median {ft[n // 2]*1000:.1f} ms, "
              f"p90 {ft[int(n * .9)]*1000:.1f}, max {ft[-1]*1000:.1f}")

    def mouse(self, x: float, y: float) -> None:
        self.io.add_mouse_pos_event(float(x), float(y))

    def hover(self, key: str, tag: str) -> bool:
        r = RECTS.get(key)
        if r is None:
            print(f"  (no rect for {key})")
            return False
        self.mouse((r[0] + r[2]) / 2, (r[1] + r[3]) / 2)
        self.frame(tag)
        return True

    def click(self, key: str, tag: str) -> bool:
        if not self.hover(key, "hover " + tag):
            return False
        self.io.add_mouse_button_event(0, True)
        self.frame("down " + tag)
        self.io.add_mouse_button_event(0, False)
        self.frame("up " + tag)
        return True

    # ---- scenarios -----------------------------------------------------

    def load_menu(self) -> None:
        print("\n--- File > Load ---")
        self.click("menu:File", "File")
        r = RECTS.get("menu:Load")
        if r is None:
            print("  (no Load menu)")
            return
        self.mouse((r[0] + r[2]) / 2, (r[1] + r[3]) / 2)
        dt = self.frame("hover Load")
        self.frame("Load settle")
        print(f"first frame with Load open: {dt*1000:.0f} ms "
              f"({len(self.app.ui.cached_configs)} configs)")
        prefs = self.app.ui.state.preferences
        was = (prefs.load_menu_core_open, prefs.load_menu_advanced_open)
        for label, core in (("Core", True), ("Custom", False)):
            prefs.load_menu_core_open = core
            prefs.load_menu_advanced_open = False
            self.mouse((r[0] + r[2]) / 2, (r[1] + r[3]) / 2)
            self.frame("back to Load")
            self.frame("back to Load")
            h = self.io.display_size.y
            items = sorted(
                (v, k) for k, v in RECTS.items()
                if "##" in k and k.split("##")[1] in ("Core", "Custom", "Advanced")
                and 20 < v[1] < h - 10 and v[3] < h)
            times = []
            for v, k in items[:60]:
                self.mouse((v[0] + v[2]) / 2, (v[1] + v[3]) / 2)
                times.append((self.frame(f"hover {k}"), k))
                self.frame("hover 2nd")
            times.sort(reverse=True)
            print(f"{label} pass: {len(items)} visible, slowest "
                  + ", ".join(f"{k.split('##')[0]} {t*1000:.0f}" for t, k in times[:4])
                  + f"; previews applied: {self.app.command_handler.preview_active}")
        prefs.load_menu_core_open, prefs.load_menu_advanced_open = was
        self.mouse(self.io.display_size.x / 2, self.io.display_size.y / 2)
        for _ in range(6):
            self.frame("leave menu")

    def browser(self) -> None:
        app = self.app
        if app.archive is None:
            print("\n(no archive open; skipping the browser)")
            return
        print(f"\n--- archive browser: {len(app.archive)} entries ---")
        self.idle(60, "gallery idle")
        r = RECTS.get("tab:Gallery")
        if r is not None:
            self.mouse(r[0] + 100, r[3] + 200)
            for i in range(30):
                self.io.add_mouse_wheel_event(0, -3.0)
                self.frame(f"gallery wheel {i}")
        arc = app.archive
        ui = app.ui
        orig = ui.get_state
        target = {"p": -1, "l": -1}

        def patched():
            st = orig()
            st.archive.preview_entry_id = target["p"]
            if target["l"] >= 0:
                st.archive.load_entry_id, target["l"] = target["l"], -1
            return st
        ui.get_state = patched
        rows = [i for i in range(len(arc.entries)) if arc.is_native(i)][:4]
        seen: dict = {}
        for i in range(len(arc.entries)):
            if not arc.is_native(i):
                seen.setdefault(arc.layout_at(i), i)
        rows += list(seen.values())[:6]
        for i in rows:
            target["p"] = i
            self.frame(f"hover entry {i} {arc.layout_at(i)}")
            target["p"] = -1
            self.frame("unhover")
        if seen:
            i = next(iter(seen.values()))
            target["p"] = i
            self.frame("hover foreign")
            target["l"] = i
            self.frame(f"CLICK foreign {arc.layout_at(i)} (brain switch)")
            for _ in range(4):
                self.frame("after switch")
            target["p"] = -1
            self.frame("unhover")
        ui.get_state = orig
        if self.click("tab:Map", "Map tab"):
            self.idle(40, "map idle")
            app.ui.state.archive.map_thumbs = True
            self.idle(60, "map with thumbnails")
            app.ui.state.archive.map_thumbs = False

    def refit(self) -> None:
        svc = self.app.map_layout_service
        if svc is None:
            print("\n(no map layout service; skipping the refit)")
            return
        svc.configure("umap")
        svc._path = Path(tempfile.mkdtemp()) / "map_layout.npz"
        before = svc.layout
        print("\n--- forced UMAP refit (cache write redirected) ---")
        svc.request_refit()
        t0 = time.perf_counter()
        n = 0
        while time.perf_counter() - t0 < 240:
            self.frame("during refit")
            n += 1
            if n > 5 and svc.layout is not before and not svc.fitting:
                break
        print(f"refit landed={svc.layout is not before} error={svc.error!r} "
              f"after {time.perf_counter()-t0:.1f}s / {n} frames")
        svc.shutdown()


def main() -> int:
    _hook_rects()
    import main as app_main

    app = app_main.App()
    d = Driver(app)
    for i in range(60):
        d.frame(f"warmup {i}")
    d.idle(60, "idle with the user's windows")
    d.load_menu()
    d.browser()
    if "--refit" in sys.argv:
        d.refit()
    fw = app.frame_watch
    print(f"\n=== {len(d.slow)} slow frames of {d.n}; frame watch: "
          f"{fw.stalls} over {fw.threshold_ms:.0f} ms ===")
    for n, dt, tag in d.slow:
        print(f"   #{n:5d} {dt*1000:6.0f} ms  {tag}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
