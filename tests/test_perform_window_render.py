"""Actually render the Perform Mode panel.

An ImGui begin/end imbalance does not fail loudly - it corrupts the whole frame
and every window in the app disappears at once. ImGui asserts on an unbalanced
stack inside EndFrame, so a test that completes at all has proved it balances.

The host is sized taller than the panel: ImGui clips a window's contents to the
WINDOW, so a control past the edge draws no vertices and every source-level
reading of the panel still says it is there.
"""
import pytest
from imgui_bundle import imgui

from services.perform_window import MonitorInfo
from state.perform_state import PerformState
from state.preferences_state import PreferencesState
from ui.perform_window import PerformWindowMixin


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


def frame(fn, n=2):
    for _ in range(n):
        imgui.new_frame()
        imgui.set_next_window_size(imgui.ImVec2(1200, 4000))
        imgui.begin("host", True)
        fn()
        imgui.end()
        imgui.render()
    return imgui.get_draw_data().total_vtx_count


def host_only():
    """An empty ImGui window is not zero vertices - it has a title bar."""
    return frame(lambda: None)


class _Keys:
    def get_key_display_name(self, action):
        return "F11"


class _State:
    def __init__(self):
        self.perform = PerformState()
        self.preferences = PreferencesState()


class Harness(PerformWindowMixin):
    def __init__(self):
        self.state = _State()
        self.keybindings = _Keys()


def _mon(name, primary=False, x=0, phys=(380, 210), dup=0):
    return MonitorInfo(name=name, width=1920, height=1080, refresh=60,
                       x=x, y=0, is_primary=primary, phys_mm=phys,
                       dup_index=dup)


@pytest.fixture
def two_displays(monkeypatch):
    monkeypatch.setattr("ui.perform_window.list_monitors",
                        lambda: [_mon("Laptop", primary=True),
                                 _mon("Projector", x=1920)])


def test_the_panel_draws(gui, two_displays):
    h = Harness()
    h.state.preferences.show_perform_window = True
    assert frame(h.render_perform_window) > host_only()


def test_it_draws_with_no_displays_reported(gui, monkeypatch):
    """GLFW can report nothing; the panel must still be a panel."""
    monkeypatch.setattr("ui.perform_window.list_monitors", lambda: [])
    h = Harness()
    assert frame(h.render_perform_window) > host_only()


def test_it_draws_while_performing(gui, two_displays):
    h = Harness()
    h.state.perform.enabled = True
    h.state.perform.active_monitor = "Projector"
    assert frame(h.render_perform_window) > host_only()


def test_it_draws_a_notice(gui, two_displays):
    h = Harness()
    h.state.perform.notice = "'Projector' is not connected."
    assert frame(h.render_perform_window) > host_only()


def test_a_remembered_display_that_is_gone_still_draws(gui, two_displays):
    """The combo's index comes off a name that need not be in the list."""
    h = Harness()
    h.state.preferences.perform_monitor = "A display from last week"
    assert frame(h.render_perform_window) > host_only()


# ---- the controls are reachable, not merely present ---------------------

_ID_WIDGETS = ("button", "small_button", "checkbox", "combo", "selectable")


def _labels(monkeypatch, draw):
    seen = []

    for fn_name in _ID_WIDGETS:
        real = getattr(imgui, fn_name, None)
        if real is None:
            continue

        def wrapper(label, *a, _real=real, **kw):
            seen.append(label)
            return _real(label, *a, **kw)

        monkeypatch.setattr(imgui, fn_name, wrapper)

    def render():
        seen.clear()
        draw()

    frame(render)
    return seen


def test_the_display_picker_and_the_start_button_are_actually_drawn(
        gui, monkeypatch, two_displays):
    """Assert on what a real frame draws, never on where the call sits."""
    h = Harness()
    labels = _labels(monkeypatch, h.render_perform_window)
    assert "Display" in labels
    assert any(l.startswith("Start") for l in labels)


def test_the_button_says_stop_while_performing(gui, monkeypatch, two_displays):
    h = Harness()
    h.state.perform.enabled = True
    labels = _labels(monkeypatch, h.render_perform_window)
    assert any(l.startswith("Stop") for l in labels)


def test_no_two_visible_widgets_share_an_id(gui, monkeypatch, two_displays):
    """Two visible items hashing to one ID stops one responding to the mouse."""
    h = Harness()
    h.state.perform.enabled = True
    h.state.perform.notice = "something to dismiss"

    seen = {}
    for fn_name in _ID_WIDGETS:
        real = getattr(imgui, fn_name, None)
        if real is None:
            continue

        def wrapper(label, *a, _real=real, **kw):
            seen.setdefault(imgui.get_id(label), []).append(label)
            return _real(label, *a, **kw)

        monkeypatch.setattr(imgui, fn_name, wrapper)

    def render():
        seen.clear()
        h.render_perform_window()

    frame(render)
    clashes = {i: v for i, v in seen.items() if len(v) > 1}
    assert not clashes, f"widgets sharing an ImGui ID: {sorted(clashes.values())}"


def test_closing_the_panel_clears_its_preference(gui, two_displays, monkeypatch):
    """The X must put the checkbox back, or Extras cannot reopen it."""
    h = Harness()
    h.state.preferences.show_perform_window = True
    monkeypatch.setattr(imgui, "begin", lambda *a, **k: (True, False))
    monkeypatch.setattr(imgui, "end", lambda: None)
    h.render_perform_window()
    assert h.state.preferences.show_perform_window is False


# ---- the calibration canvas, driven with a real pointer ------------------

def _step(draw, mouse=None, down=None):
    """One frame, with optional pointer events queued ahead of it."""
    io = imgui.get_io()
    if mouse is not None:
        io.add_mouse_pos_event(mouse[0], mouse[1])
    if down is not None:
        io.add_mouse_button_event(0, down)
    imgui.new_frame()
    imgui.set_next_window_pos(imgui.ImVec2(0, 0))
    imgui.set_next_window_size(imgui.ImVec2(1200, 4000))
    imgui.begin("host", True)
    draw()
    imgui.end()
    imgui.render()


@pytest.fixture
def canvas(monkeypatch):
    """Where the proxy canvas landed. Predicting it from outside is guesswork."""
    box = {}
    real = imgui.invisible_button

    def wrapper(str_id, size, *a, **kw):
        if str_id == "##perform_calib":
            pos = imgui.get_cursor_screen_pos()
            box["origin"] = (pos.x, pos.y)
            box["size"] = (size.x, size.y)
        return real(str_id, size, *a, **kw)

    monkeypatch.setattr(imgui, "invisible_button", wrapper)
    return box


def _performing():
    h = Harness()
    h.state.perform.enabled = True
    h.state.perform.active_monitor = "Projector"
    return h


def test_the_calibrate_checkbox_is_drawn(gui, monkeypatch, two_displays):
    h = _performing()
    labels = _labels(monkeypatch, h.render_perform_window)
    assert any(l.startswith("Calibrate") for l in labels)


def test_the_reset_button_appears_only_while_performing(
        gui, monkeypatch, two_displays):
    stopped = _labels(monkeypatch, Harness().render_perform_window)
    assert not any(l.startswith("Reset Corners") for l in stopped)

    running = _labels(monkeypatch, _performing().render_perform_window)
    assert any(l.startswith("Reset Corners") for l in running)


def test_the_canvas_is_not_drawn_until_performing(gui, two_displays, canvas):
    _step(Harness().render_perform_window)
    assert canvas == {}


def test_dragging_a_handle_moves_that_corner_with_v_flipped(
        gui, two_displays, canvas):
    """The widget owns the flip. Dragging DOWN the screen must LOWER v."""
    h = _performing()
    draw = h.render_perform_window

    _step(draw)                                  # lay out; learn the geometry
    ox, oy = canvas["origin"]
    w, ht = canvas["size"]

    top_left = (ox + 1.0, oy + 1.0)              # the TL handle, uncalibrated
    _step(draw, mouse=top_left)
    _step(draw, mouse=top_left, down=True)

    target = (ox + 0.25 * w, oy + 0.20 * ht)
    _step(draw, mouse=target)
    _step(draw, mouse=target)

    corners = h.state.perform.corners
    assert corners is not None, "the drag never reached the state"
    assert corners[0][0] == pytest.approx(0.25, abs=0.03)
    assert corners[0][1] == pytest.approx(0.80, abs=0.03)
    # Only the grabbed corner moved.
    assert corners[2] == pytest.approx((1.0, 0.0), abs=1e-6)

    _step(draw, mouse=target, down=False)
    assert h.state.perform.held_corner == -1


def test_a_click_far_from_every_handle_grabs_nothing(
        gui, two_displays, canvas):
    """Otherwise the nearest corner teleports to wherever you clicked."""
    h = _performing()
    draw = h.render_perform_window

    _step(draw)
    ox, oy = canvas["origin"]
    w, ht = canvas["size"]

    middle = (ox + 0.5 * w, oy + 0.5 * ht)
    _step(draw, mouse=middle)
    _step(draw, mouse=middle, down=True)
    _step(draw, mouse=(ox + 0.6 * w, oy + 0.6 * ht))
    _step(draw, mouse=(ox + 0.6 * w, oy + 0.6 * ht), down=False)

    assert h.state.perform.corners is None, "an empty click calibrated it"


def test_reset_asks_the_orchestrator_rather_than_clearing_it_here(
        gui, monkeypatch, two_displays):
    """The UI is passive: it raises a one-shot, main.py drops the prefs entry."""
    h = _performing()
    h.state.perform.corners = ((0.1, 0.9), (0.9, 0.9), (0.9, 0.1), (0.1, 0.1))

    real = imgui.button

    def wrapper(label, *a, **kw):
        real(label, *a, **kw)
        return label.startswith("Reset Corners")

    monkeypatch.setattr(imgui, "button", wrapper)
    _step(h.render_perform_window)

    assert h.state.perform.reset_corners_requested is True
    assert h.state.perform.corners is not None
