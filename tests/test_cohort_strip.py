"""The strip is painted by dragging, so it has to be DRIVEN.

A test that asserts the cells were drawn passes just as happily when the drag
does nothing - the same trap tests/test_map_wheel.py exists for.
"""
import pytest
from imgui_bundle import imgui

from services import cohort_audio as ca
from services.audio_mapping import Mapping

HOST = "cohortstriphost"
N = 64


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
    def __init__(self):
        from ui.cohort_strip import CohortStripMixin

        class _Host(CohortStripMixin):
            pass

        self.ui = _Host()
        self.m = Mapping(signal="bass", target="SENSOR_GAIN")
        self.rects = []

    def frame(self, mouse, down):
        io = imgui.get_io()
        io.add_mouse_pos_event(mouse[0], mouse[1])
        io.add_mouse_button_event(0, down)
        imgui.new_frame()
        imgui.set_next_window_size(imgui.ImVec2(700, 300))
        imgui.begin(HOST)
        self.ui.render_cohort_strip(self.m, N)
        self.rects = list(self.ui._cohort_cell_rects)
        imgui.end()
        imgui.render()


def _centre(rect):
    return ((rect[0] + rect[2]) * 0.5, (rect[1] + rect[3]) * 0.5)


def test_the_strip_draws_one_cell_per_live_cohort(gui):
    r = _Rig()
    r.frame((0, 0), False)
    assert len(r.rects) == N


def test_pressing_a_lit_cell_erases_it(gui):
    r = _Rig()
    r.frame((0, 0), False)
    target = _centre(r.rects[5])
    r.frame(target, True)
    r.frame(target, False)
    assert not ca.covers(r.m.cohorts, 5, N)


def test_dragging_across_lit_cells_erases_all_of_them(gui):
    r = _Rig()
    r.frame((0, 0), False)
    r.frame(_centre(r.rects[10]), True)
    for cell in range(11, 20):
        r.frame(_centre(r.rects[cell]), True)
    r.frame(_centre(r.rects[19]), False)
    for cell in range(10, 20):
        assert not ca.covers(r.m.cohorts, cell, N), cell
    assert ca.covers(r.m.cohorts, 25, N)


def test_a_drag_started_on_a_dark_cell_draws_rather_than_erases(gui):
    r = _Rig()
    r.m.cohorts[:] = False
    r.frame((0, 0), False)
    r.frame(_centre(r.rects[3]), True)
    for cell in range(4, 9):
        r.frame(_centre(r.rects[cell]), True)
    r.frame(_centre(r.rects[8]), False)
    for cell in range(3, 9):
        assert ca.covers(r.m.cohorts, cell, N), cell


def test_releasing_ends_the_drag(gui):
    r = _Rig()
    r.frame((0, 0), False)
    r.frame(_centre(r.rects[30]), True)
    r.frame(_centre(r.rects[30]), False)
    r.frame(_centre(r.rects[31]), False)
    assert ca.covers(r.m.cohorts, 31, N)


def test_the_pointer_leaving_the_strip_paints_nothing_new(gui):
    """A drag that wanders off the widget must not keep painting."""
    r = _Rig()
    r.frame((0, 0), False)
    r.frame(_centre(r.rects[40]), True)
    far = (r.rects[0][0] - 200.0, r.rects[0][1] - 200.0)
    r.frame(far, True)
    r.frame(far, False)
    assert not ca.covers(r.m.cohorts, 40, N)
    assert ca.covers(r.m.cohorts, 41, N)
