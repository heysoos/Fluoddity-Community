"""Actually render the undo panel.

An ImGui begin/end imbalance does not fail loudly - it corrupts the whole
frame, so every window in the app disappears at once and the cause is
invisible. A test that completes at all has proved the stack balances.

The host is sized taller than anything it can hold: ImGui clips a window's
contents to the WINDOW, not to the display, so an undersized host draws no
vertices and any assertion on "it drew something" becomes a coin flip.
"""
import pytest
from imgui_bundle import imgui

from state.ui_state import UIState
from ui.undo_window import UndoWindowMixin


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


class Panel(UndoWindowMixin):
    def __init__(self, steps, cursor):
        self.state = UIState()
        self.undo_steps = steps
        self.undo_cursor = cursor


def frame(panel, n=2):
    for _ in range(n):
        imgui.new_frame()
        imgui.set_next_window_size(imgui.ImVec2(1200, 4000))
        imgui.begin("host", True)
        panel.render_undo_window()
        imgui.end()
        imgui.render()
    return imgui.get_draw_data().total_vtx_count


def test_an_empty_history_renders(gui):
    assert frame(Panel([], -1)) > 0


def test_a_populated_history_renders(gui):
    steps = [(0, "Start", False), (1, "Sensor Gain", False),
             (2, "Load Karst", True)]
    assert frame(Panel(steps, 1)) > 0


def test_nothing_hovered_clears_the_preview_index(gui):
    panel = Panel([(0, "Start", False), (1, "Sensor Gain", False)], 1)
    panel.state.undo_preview_index = 7
    frame(panel)
    assert panel.state.undo_preview_index == -1


def test_the_labels_a_frame_draws_include_every_step(gui):
    """A control the user has to find cannot be asserted on by source."""
    steps = [(0, "Start", False), (1, "Sensor Gain", False)]
    panel = Panel(steps, 1)
    drawn = []
    panel._undo_row_labels = drawn
    frame(panel, n=1)
    assert "Start" in " ".join(drawn)
    assert "Sensor Gain" in " ".join(drawn)
