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

from services import field_sources
from services.shader_params import ShaderParam
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


class _FakeTex:
    """imgui.ImTextureRef only needs the handle; nothing dereferences it
    until the renderer runs, which these tests never reach."""
    glo = 1


class _FakeBus:
    """Stands in for FieldBus, and every name here is checked against the real
    class - a fake that answers to a method the bus does not have puts the
    window's calls out of reach of this test.
    """
    resolution = (320, 320)
    pass_count = 1

    def __init__(self):
        self.drawn = 0

    def set_thumbnails_enabled(self, on): pass
    def mark_dirty(self): pass
    def inspect(self, layer, view): return _FakeTex()

    def thumbnail_for(self, layer):
        self.drawn += 1
        return _FakeTex()


class _Harness(FieldStackWindowMixin):
    """Only the mixin plus what App really injects.

    Deliberately stubs NOTHING the real UI does not have: a harness that
    supplies a helper the app lacks makes the window pass here and crash
    there, which is how a call to a tooltip helper removed two commits
    earlier survived a green suite. A bus of None is the same trap one layer
    down - it makes every thumbnail path return before it draws.
    """

    def __init__(self, stack, bus=None):
        self.state = _State(stack)
        self.field_bus = bus
        self.labels = []


def draw(stack, n=2, bus=None):
    """Render the window offscreen; return (harness, vertex count)."""
    harness = _Harness(stack, bus)
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


def test_a_row_draws_its_thumbnail_through_a_live_bus(gui):
    """With no bus every thumbnail path returns before it draws anything.

    That is what let `imgui.image(tex.glo, ...)` ship: imgui_bundle wants an
    ImTextureRef and raises TypeError on a bare int, but only on the frame the
    call actually runs. Vertex count cannot stand in for that - the thumbnail
    puts the row on one line, so drawing MORE can total fewer vertices.
    """
    bus = _FakeBus()
    draw(FieldStack(layers=[FieldLayer(), FieldLayer()]), bus=bus)
    assert bus.drawn, "no row asked the bus for a thumbnail"


def test_the_inspect_panel_draws_its_texture(gui):
    stack = FieldStack(layers=[FieldLayer()])
    harness = _Harness(stack, _FakeBus())
    harness._inspect_uid = stack.layers[0].uid
    imgui.new_frame()
    imgui.set_next_window_size(imgui.ImVec2(1200, 4000))
    imgui.begin("host", True)
    harness.render_field_stack_window(collect=harness.labels)
    imgui.end()
    imgui.render()
    assert any("Close" in l for l in harness.labels)


def test_the_fake_bus_only_answers_to_the_real_buss_api():
    """A fake free to invent methods hides the window calling a missing one."""
    from utilities.field_bus import FieldBus
    for name in ("set_thumbnails_enabled", "mark_dirty", "thumbnail_for",
                 "inspect", "resolution", "pass_count"):
        assert hasattr(FieldBus, name), f"FieldBus has no {name}"


@pytest.mark.parametrize("key", [d.key for d in field_sources.descriptors()])
def test_every_source_draws_its_own_parameter_widgets(gui, key):
    """One source per test, because each declares a different set of kinds.

    Drawing only `noise` left the vec2 branch - gradient's Centre - and with it
    `imgui.slider_float2`, unexecuted by the whole suite.
    """
    draw(FieldStack(layers=[FieldLayer(source=key)]), bus=_FakeBus())


def test_every_parameter_kind_has_a_widget_that_runs(gui, monkeypatch):
    """imgui_bundle raises TypeError on a bad signature rather than failing to
    compile, so a branch no shipped shader reaches is a crash in waiting."""
    params = [
        ShaderParam("f", "float", 0.0, 1.0, (0.5,), "F", 1),
        ShaderParam("i", "int", 0.0, 8.0, (4.0,), "I", 1),
        ShaderParam("b", "bool", 0.0, 1.0, (1.0,), "B", 1),
        ShaderParam("c", "color", 0.0, 1.0, (1.0, 0.5, 0.0), "C", 3),
        ShaderParam("v", "vec", 0.0, 1.0, (0.5, 0.5), "V", 2),
    ]
    monkeypatch.setattr(field_sources, "params_for", lambda layer: params)
    harness, _ = draw(FieldStack(layers=[FieldLayer()]), bus=_FakeBus())
    for p in params:
        assert any(p.label in l for l in harness.labels), f"{p.kind} drew nothing"


def test_a_shader_layer_offers_a_file_to_point_at(gui):
    """Both file-backed sources read params["_file"], which nothing else sets."""
    harness, _ = draw(FieldStack(layers=[FieldLayer(source="shader")]),
                      bus=_FakeBus())
    assert any("Shader##file" in l or "no .frag" in l for l in harness.labels)


def test_an_image_layer_offers_a_path_field(gui):
    harness, _ = draw(FieldStack(layers=[FieldLayer(source="image")]),
                      bus=_FakeBus())
    assert any("Image##file" in l for l in harness.labels)


def test_choosing_a_shader_writes_the_file_the_source_reads(gui):
    """The picker must fill the key field_sources looks up, not a new one."""
    if not field_sources.available_shader_files():
        pytest.skip("no .frag files installed")
    layer = FieldLayer(source="shader")
    draw(FieldStack(layers=[layer]), bus=_FakeBus())
    assert layer.params.get("_file"), "the picker set no file"
    assert field_sources.resolve_shader_path(layer.params["_file"]) is not None
