"""The hatched swing and the base tick, in track fractions and in pixels."""
import pytest
from imgui_bundle import imgui

from ui.slider_widgets import swing_fraction, track_span, track_x


def test_a_swing_upward_starts_at_the_base():
    start, width = swing_fraction(base=2.0, lo=0.0, hi=10.0, reach=5.0)
    assert start == pytest.approx(0.2)
    assert width == pytest.approx(0.3)


def test_a_swing_downward_ends_at_the_base():
    start, width = swing_fraction(base=5.0, lo=0.0, hi=10.0, reach=2.0)
    assert start == pytest.approx(0.2)
    assert width == pytest.approx(0.3)


def test_a_swing_past_the_top_is_clipped_to_the_track():
    start, width = swing_fraction(base=8.0, lo=0.0, hi=10.0, reach=40.0)
    assert start == pytest.approx(0.8)
    assert start + width == pytest.approx(1.0)


def test_a_swing_past_the_bottom_is_clipped_to_the_track():
    start, width = swing_fraction(base=1.0, lo=0.0, hi=10.0, reach=-40.0)
    assert start == pytest.approx(0.0)
    assert start + width == pytest.approx(0.1)


def test_a_bipolar_range_places_zero_in_the_middle():
    start, _width = swing_fraction(base=0.0, lo=-1.0, hi=1.0, reach=0.0)
    assert start == pytest.approx(0.5)


def test_no_swing_has_zero_width():
    _start, width = swing_fraction(base=3.0, lo=0.0, hi=10.0, reach=3.0)
    assert width == pytest.approx(0.0)


def test_a_degenerate_range_does_not_divide_by_zero():
    start, width = swing_fraction(base=1.0, lo=1.0, hi=1.0, reach=5.0)
    assert start == pytest.approx(0.0) and width == pytest.approx(0.0)


def test_the_slider_never_uses_add_line_for_the_swing():
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent
           / "ui" / "slider_widgets.py").read_text(encoding="utf-8")
    assert "add_line(" not in src


# --- pixels: a fraction has to land on the slider, under its grab -------------

ITEM_WIDTH = 200.0
LABEL = "Global Force Mult"


@pytest.fixture(scope="module")
def gui():
    imgui.create_context()
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(1280, 900)
    io.delta_time = 1.0 / 60.0
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    yield io
    imgui.destroy_context()


def _slider_frame(io, value, mouse_x=None, mouse_down=False):
    """One frame holding one slider; returns (value_after, track_left, width)."""
    if mouse_x is not None:
        io.add_mouse_pos_event(mouse_x[0], mouse_x[1])
    io.add_mouse_button_event(0, mouse_down)
    imgui.new_frame()
    imgui.begin("t")
    imgui.push_item_width(ITEM_WIDTH)
    changed, v = imgui.slider_float(LABEL, value, -1.0, 1.0)
    p0, p1 = imgui.get_item_rect_min(), imgui.get_item_rect_max()
    x0, w = track_span(p0.x, p1.x, LABEL)
    mid_y = (p0.y + p1.y) * 0.5
    imgui.pop_item_width()
    imgui.end()
    imgui.end_frame()
    imgui.render()
    return (v if changed else value), x0, w, mid_y


def test_the_track_excludes_the_label_drawn_beside_it(gui):
    """get_item_rect_* covers the slider AND its label, so a fraction of that
    rect runs off the end of the slider it is meant to mark."""
    _v, _x0, w, _y = _slider_frame(gui, 0.0)
    assert w == pytest.approx(ITEM_WIDTH)


def test_a_fraction_maps_to_where_imgui_puts_the_grab(gui):
    """Clicking a slider sets it from the mouse x, so the value ImGui reports
    at track_x(t) IS the value a marker drawn there would be claiming."""
    _v, x0, w, mid_y = _slider_frame(gui, 0.0)          # settle the window
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        value = 0.0
        for frame in range(6):
            x = track_x(x0, w, t)
            value, x0, w, mid_y = _slider_frame(
                gui, value, (x, mid_y), 1 <= frame <= 4)
        assert value == pytest.approx(-1.0 + 2.0 * t, abs=0.02), (
            f"a marker at t={t} does not sit under the grab")
    gui.add_mouse_button_event(0, False)


def test_a_zero_width_track_does_not_divide_by_zero(gui):
    assert track_x(10.0, 0.0, 0.5) >= 10.0


# --- the context menu, whose body only runs while it is OPEN ------------------

class _Defaults:
    source_filename = ""
    values: dict = {}


def _menu_host():
    from state import UIState
    from ui.slider_widgets import SliderWidgetsMixin

    class Host(SliderWidgetsMixin):
        def __init__(self):
            self.state = UIState()
            self.current_physics_defaults = _Defaults()
            self.param_lock_service = None

        def _delayed_tooltip(self, text):
            pass

    return Host()


def test_the_physics_slider_context_menu_renders_when_open(gui):
    """Nothing rendered this body, so the Audio... item added to it was never
    executed by a test - the same blind spot that let a bad selectable() call
    reach the user in the audio panel."""
    host = _menu_host()
    real = imgui.begin_popup_context_item

    def spy(str_id=None, *a, **kw):
        if str_id:
            imgui.open_popup(str_id)
        return real(str_id, *a, **kw)

    imgui.begin_popup_context_item = spy
    try:
        for _ in range(2):
            imgui.new_frame()
            imgui.begin("menu host")
            imgui.slider_float("Axial Force", 0.5, -1.0, 1.0)
            host.add_slider_context_menu("Axial Force", -1.0, 1.0)
            imgui.end()
            imgui.end_frame()
            imgui.render()
    finally:
        imgui.begin_popup_context_item = real


def test_the_audio_item_targets_the_parameter_the_slider_edits(gui):
    """It writes open_target, which the panel uses to jump to a row; a label
    that did not map to a parameter name would silently open nothing."""
    from ui.physics_params import PARAM_BY_LABEL
    host = _menu_host()
    assert host._label_to_param_name("Axial Force") == "AXIAL_FORCE"
    assert "Axial Force" in PARAM_BY_LABEL
