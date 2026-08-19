"""Actually render the Field Stack window.

An ImGui begin/end imbalance or a None dereference does not fail loudly - it
corrupts the whole frame, so every window in the app disappears at once and the
cause is invisible. A test that completes at all has proved the stack balances.

The host is sized far taller than the panel: ImGui clips a window's contents to
the WINDOW, not to the display, so at a default size the lower controls fall
outside it and draw no vertices, which turns "the control rendered" into a test
of nothing.
"""
import pytest
from imgui_bundle import imgui

from state.field_stack import FieldLayer, FieldStack
from state.preferences_state import PreferencesState
from ui.field_stack_window import FieldStackWindowMixin


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


class _State:
    def __init__(self, stack):
        self.field_stack = stack
        self.preferences = PreferencesState()


class _Harness(FieldStackWindowMixin):
    """Only the mixin plus what App really injects.

    Deliberately stubs NOTHING the real UI does not have: a harness that
    supplies a helper the app lacks makes the window pass here and crash
    there, which is how a call to a tooltip helper removed two commits
    earlier survived a green suite.
    """

    def __init__(self, stack):
        self.state = _State(stack)
        self.field_bus = None
        self.labels = []


def draw(stack, n=2):
    """Render the window offscreen; return (harness, vertex count)."""
    harness = _Harness(stack)
    for _ in range(n):
        harness.labels = []
        imgui.new_frame()
        imgui.set_next_window_size(imgui.ImVec2(1200, 4000))
        imgui.begin("host", True)
        harness.render_field_stack_window(collect=harness.labels)
        imgui.end()
        imgui.render()
    return harness, imgui.get_draw_data().total_vtx_count


def test_an_empty_stack_still_offers_add_layer(gui):
    harness, _ = draw(FieldStack())
    assert any("Add layer" in l for l in harness.labels)


def test_the_bus_resolution_control_is_reachable(gui):
    harness, _ = draw(FieldStack())
    assert any("Bus Resolution" in l for l in harness.labels)


def test_a_layer_draws_its_source_and_destination(gui):
    harness, _ = draw(FieldStack(layers=[
        FieldLayer(source="noise", destination="force")]))
    assert any("noise" in l for l in harness.labels)
    assert any("force" in l for l in harness.labels)


def test_a_layer_error_is_drawn(gui):
    layer = FieldLayer(source="shader")
    layer.error = "0:31 'noize' : no matching function"
    harness, _ = draw(FieldStack(layers=[layer]))
    assert any("noize" in l for l in harness.labels), "the compile log was not shown"


def test_a_layer_draws_more_than_an_empty_stack(gui):
    _, empty = draw(FieldStack())
    _, one = draw(FieldStack(layers=[FieldLayer()]))
    assert one > empty, "the layer row drew no geometry"


def test_no_two_widgets_share_an_id(gui):
    """A duplicate ImGui id puts up a conflict dialog and kills one widget."""
    harness, _ = draw(FieldStack(layers=[FieldLayer(), FieldLayer(), FieldLayer()]))
    assert len(harness.labels) == len(set(harness.labels))


def test_a_shader_layers_params_come_from_its_file(gui):
    """A noise layer must expose the sliders its .frag declares."""
    harness, _ = draw(FieldStack(layers=[FieldLayer(source="noise")]))
    assert any("Octaves" in l for l in harness.labels)
    assert any("Domain Warp" in l for l in harness.labels)
