"""The cohort selection strip: one cell per live cohort, painted by dragging.

The drag direction is set by the cell it starts on - press a lit one and you
erase, press a dark one and you draw - which is what makes a single click and a
sweep the same gesture.
"""
from __future__ import annotations

from imgui_bundle import imgui

from services import cohort_audio as ca

_HEIGHT = 22.0
_GAP = 1.0
_LIT = 0xFFE04F61        # ABGR, as ImGui packs them
_DARK = 0xFF3A3A3A
_EDGE = 0xFF202020


class CohortStripMixin:
    _cohort_paint = None
    _cohort_cell_rects: list = []

    def render_cohort_strip(self, m, n_cohorts: int) -> None:
        n = max(1, int(n_cohorts))
        avail = max(32.0, imgui.get_content_region_avail().x)
        w = max(2.0, (avail - _GAP * (n - 1)) / n)
        origin = imgui.get_cursor_screen_pos()
        draw = imgui.get_window_draw_list()
        lit = ca.cells_lit(m.cohorts, n)

        rects = []
        for i in range(n):
            x = origin.x + i * (w + _GAP)
            rects.append((x, origin.y, x + w, origin.y + _HEIGHT))
            draw.add_rect_filled(imgui.ImVec2(x, origin.y),
                                 imgui.ImVec2(x + w, origin.y + _HEIGHT),
                                 _LIT if lit[i] else _DARK)
        self._cohort_cell_rects = rects
        draw.add_rect(imgui.ImVec2(origin.x, origin.y),
                      imgui.ImVec2(origin.x + avail, origin.y + _HEIGHT), _EDGE)

        imgui.invisible_button("##cohort_strip", imgui.ImVec2(avail, _HEIGHT))

        hovered = None
        if imgui.is_item_hovered() or imgui.is_item_active():
            mx = imgui.get_io().mouse_pos.x
            idx = int((mx - origin.x) / (w + _GAP))
            if 0 <= idx < n:
                hovered = idx

        if not imgui.is_mouse_down(0):
            self._cohort_paint = None
        elif hovered is not None:
            if self._cohort_paint is None:
                # The first cell sets the direction for the whole gesture.
                self._cohort_paint = not bool(lit[hovered])
            ca.paint(m.cohorts, hovered, n, self._cohort_paint)

    def render_cohort_buttons(self, m, n_cohorts: int) -> None:
        n = max(1, int(n_cohorts))
        if imgui.button("All##cohort_all"):
            m.cohorts[:] = True
        imgui.same_line()
        if imgui.button("None##cohort_none"):
            m.cohorts[:] = False
        imgui.same_line()
        if imgui.button("Invert##cohort_invert"):
            m.cohorts[:] = ~m.cohorts
        for stride in (2, 3, 4):
            imgui.same_line()
            if imgui.button(f"Every {stride}##cohort_every{stride}"):
                m.cohorts[:] = False
                for cell in range(0, n, stride):
                    ca.paint(m.cohorts, cell, n, True)
