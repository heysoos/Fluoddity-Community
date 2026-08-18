"""The wheel over the map canvas zooms, and does NOT also scroll the browser.

ImGui routes the wheel during NewFrame, before any of our code runs, so this
cannot be checked by reading the drawing code - it has to be driven. The panel
under test must be genuinely scrollable AND parked mid-scroll: at scroll 0 a
wheel-up has nowhere to go, so a broken build passes.
"""
import numpy as np
import pytest
from imgui_bundle import imgui

from services.map_layout_service import MapLayoutService
from tests.test_archive_window_render import Harness, _populated

HOST = "wheelhost"
_MID_SCROLL = 80.0


@pytest.fixture(scope="module")
def gui():
    imgui.create_context()
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(1280, 900)
    io.delta_time = 1.0 / 60.0
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    for _ in range(2):
        imgui.new_frame()
        imgui.render()
    yield
    imgui.destroy_context()


class _Rig:
    """A scrollable host window with the real map inside it."""

    def __init__(self):
        self.h = Harness(archive=_populated(n=64))
        # Short, so the filler below it is genuinely reachable inside a 300px
        # window. At the shipped 320 the canvas fills the panel and every
        # "off the canvas" probe lands back on it.
        self.h._MAP_H = 90.0
        rs = np.random.RandomState(0)
        self.h.archive_obj.embeddings = rs.randn(64, 8).astype(np.float32)
        svc = MapLayoutService()
        svc.bind(self.h.archive_obj, None, "clip-b32")
        svc.update(self.h.archive_obj)
        self.h.map_layout_service = svc
        self.hit = {}
        self._real_ib = imgui.invisible_button

    def __enter__(self):
        def spy(str_id, size, *a, **kw):
            if str_id == "map_hit":
                p = imgui.get_cursor_screen_pos()
                self.hit["rect"] = (p.x, p.y, size.x, size.y)
            return self._real_ib(str_id, size, *a, **kw)

        imgui.invisible_button = spy
        for _ in range(3):
            self.frame()
        return self

    def __exit__(self, *exc):
        imgui.invisible_button = self._real_ib

    @property
    def ast(self):
        return self.h.state.archive

    def frame(self, wheel=0.0, mouse=None, set_scroll=None):
        io = imgui.get_io()
        io.mouse_wheel = float(wheel)
        if mouse is not None:
            io.mouse_pos = imgui.ImVec2(*mouse)
        imgui.new_frame()
        imgui.set_next_window_size(imgui.ImVec2(520, 300))
        imgui.set_next_window_pos(imgui.ImVec2(20, 20))
        imgui.begin(HOST, True)
        if set_scroll is not None:
            imgui.set_scroll_y(float(set_scroll))
        self.h._render_map(self.ast, self.h.archive_obj)
        for i in range(30):
            imgui.text(f"filler {i}")
        imgui.end()
        imgui.render()

    @property
    def scroll(self):
        w = imgui.internal.find_window_by_name(HOST)
        return None if w is None else float(w.scroll.y)

    def over_canvas(self):
        x, y, w, h = self.hit["rect"]
        return (x + w * 0.5, y + min(h, 180.0) * 0.5)

    def over_filler(self):
        """Below the canvas, off it, still inside the scrollable panel."""
        x, y, w, h = self.hit["rect"]
        return (x + w * 0.5, y + h + 24.0)

    def wheel_at(self, where, notches):
        """-> (did the zoom move?, did the panel scroll?)

        `where` is a CALLABLE returning the point, resolved only after the
        panel is parked mid-scroll: scrolling moves the canvas, so a point
        computed beforehand aims 80px off it.
        """
        self.ast.map_zoom = 4.0          # mid-range: either direction can move
        self.ast.map_center_x = self.ast.map_center_y = 0.5
        for _ in range(2):
            self.frame(0.0, set_scroll=_MID_SCROLL)
        mouse = where()
        for _ in range(2):
            self.frame(0.0, mouse, set_scroll=_MID_SCROLL)
        z0, s0 = self.ast.map_zoom, self.scroll
        self.frame(float(notches), mouse)
        self.frame(0.0, mouse)
        return self.ast.map_zoom != z0, self.scroll != s0


@pytest.mark.parametrize("notches", [1.0, -1.0])
def test_the_wheel_over_the_canvas_zooms_without_scrolling_the_panel(gui, notches):
    with _Rig() as rig:
        zoomed, scrolled = rig.wheel_at(rig.over_canvas, notches)
    assert zoomed, "the wheel did not reach the zoom"
    assert not scrolled, "the wheel zoomed AND scrolled the panel behind it"


@pytest.mark.parametrize("notches", [1.0, -1.0])
def test_the_wheel_off_the_canvas_still_scrolls_the_panel(gui, notches):
    """The other half: claiming the wheel must not make the browser unscrollable."""
    with _Rig() as rig:
        zoomed, scrolled = rig.wheel_at(rig.over_filler, notches)
    assert not zoomed, "the wheel zoomed the map from outside the canvas"
    assert scrolled, "the panel no longer scrolls at all"


def test_the_panel_under_test_can_actually_scroll_both_ways(gui):
    """Guards the rig itself: parked at 0 a wheel-up moves nothing, and a
    broken build would pass the test above for the wrong reason."""
    with _Rig() as rig:
        seen = set()
        for notches in (1.0, -1.0):
            for _ in range(2):
                rig.frame(0.0, set_scroll=_MID_SCROLL)
            for _ in range(2):
                rig.frame(0.0, rig.over_filler(), set_scroll=_MID_SCROLL)
            before = rig.scroll
            rig.frame(notches, rig.over_filler())
            rig.frame(0.0, rig.over_filler())
            seen.add(rig.scroll - before)
        assert len(seen) == 2 and 0.0 not in seen, seen
