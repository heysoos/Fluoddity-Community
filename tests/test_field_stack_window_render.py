"""Actually render the Field Stack window.

An ImGui begin/end imbalance or a None dereference does not fail loudly - it
corrupts the whole frame, so every window in the app disappears at once and the
cause is invisible. A test that completes at all has proved the stack balances.

The host is sized far taller than the panel: ImGui clips a window's contents to
the WINDOW, not to the display, so at a default size the lower controls fall
outside it and draw no vertices, which turns "the control rendered" into a test
of nothing.
"""
import contextlib

import pytest
from imgui_bundle import imgui

from services import field_sources, file_picker, webcam
from services.shader_params import ShaderParam
from state.field_stack import FieldLayer, FieldStack
from state.preferences_state import PreferencesState
from state.field_stack import BLENDS, DESTINATIONS, MAPPINGS
from ui.field_stack_window import (BLEND_LABELS, DESTINATION_LABELS,
                                   MAPPING_LABELS, FieldStackWindowMixin)


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
        self.asked = None

    def set_thumbnails_enabled(self, on): pass
    def mark_dirty(self): pass
    def request_inspect(self, uid, view): self.asked = (uid, view)
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


def draw(stack, n=2, bus=None, force_open=True):
    """Render the window offscreen; return (harness, vertex count).

    Layer bodies live behind a collapsing header, so a default-closed header
    draws none of the controls under test - the same trap the archive window
    tests document. `force_open` opens every header before the frame.
    """
    harness = _Harness(stack, bus)
    for _ in range(n):
        harness.labels = []
        imgui.new_frame()
        imgui.set_next_window_size(imgui.ImVec2(1200, 4000))
        imgui.begin("host", True)
        with _headers_open(force_open):
            harness.render_field_stack_window(collect=harness.labels)
        imgui.end()
        imgui.render()
    return harness, imgui.get_draw_data().total_vtx_count


@contextlib.contextmanager
def _headers_open(on):
    """Force every collapsing header open for the duration of one frame.

    A closed header runs none of its body, so every control under test draws
    no vertices and the assertions become tests of nothing - the trap the
    archive window tests already document for popups.
    """
    if not on:
        yield
        return
    real = imgui.collapsing_header

    def always(label, *a, **kw):
        out = real(label, *a, **kw)
        # The two-argument overload returns (clicked, still_visible).
        return (True, out[1]) if isinstance(out, tuple) else True

    imgui.collapsing_header = always
    try:
        yield
    finally:
        imgui.collapsing_header = real


@contextlib.contextmanager
def _combo_open(match):
    """Force ONE combo to run its body, chosen by a fragment of its label.

    A combo submits its entries only while its popup is open, so an ordinary
    frame executes none of them - the same blindness a popup body has, and the
    reason a per-entry tooltip can be added, look right in the source and
    never draw. Only one popup can stand open at a time, which is why this
    takes a label rather than opening them all.

    BeginCombo's popup id is "##ComboPopup" hashed against the combo's OWN id,
    so pushing the label onto the id stack first is what makes open_popup name
    the right one.
    """
    real = imgui.begin_combo

    def forced(label, preview, *a, **kw):
        if match in label:
            imgui.push_id(label)
            imgui.open_popup("##ComboPopup")
            imgui.pop_id()
        return real(label, preview, *a, **kw)

    imgui.begin_combo = forced
    try:
        yield
    finally:
        imgui.begin_combo = real


def draw_with_combo(stack, match, n=2, bus=None):
    """Render with one combo's list open. -> (harness, labels)."""
    harness = _Harness(stack, bus)
    for _ in range(n):
        harness.labels = []
        imgui.new_frame()
        imgui.set_next_window_size(imgui.ImVec2(1200, 4000))
        imgui.begin("host", True)
        with _headers_open(True), _combo_open(match):
            harness.render_field_stack_window(collect=harness.labels)
        imgui.end()
        imgui.render()
    return harness, harness.labels


def test_every_choice_carries_a_label_and_a_line_of_its_own():
    """Derived from the registries, so a new option cannot ship unexplained.

    A hand-written list is what let an entry go out with the enum key showing
    and nothing to say what it does.
    """
    from ui import field_stack_window as w

    for keys, labels, tips, what in (
            (MAPPINGS, MAPPING_LABELS, w.MAPPING_TIPS, "mapping"),
            (DESTINATIONS, DESTINATION_LABELS, w.DESTINATION_TIPS, "destination"),
            (BLENDS, BLEND_LABELS, w.BLEND_TIPS, "blend")):
        for key in keys:
            assert labels.get(key), f"{what} '{key}' has no label"
            assert tips.get(key), f"{what} '{key}' has no tooltip"
    for desc in field_sources.descriptors():
        assert w.SOURCE_TIPS.get(desc.key), f"source '{desc.key}' has no tooltip"


def test_the_forced_combo_helper_really_opens_something(gui):
    """Without this the tooltip tests below would be coverage of nothing."""
    stack = FieldStack(layers=[FieldLayer(mapping="curl")])
    shut, _ = draw(stack)
    _, opened = draw_with_combo(FieldStack(layers=[FieldLayer(mapping="curl")]),
                                "Mapping")
    entry = MAPPING_LABELS["gradient"]
    assert not any(l.startswith(entry) for l in shut.labels), (
        "a closed combo already drew its entries - the helper proves nothing")
    assert any(l.startswith(entry) for l in opened), (
        "the helper did not open the combo")


def test_every_mapping_is_listed_with_its_own_entry(gui):
    _, labels = draw_with_combo(FieldStack(layers=[FieldLayer()]), "Mapping")
    for key in MAPPINGS:
        assert any(l.startswith(MAPPING_LABELS[key]) for l in labels), key


def test_every_destination_is_listed(gui):
    _, labels = draw_with_combo(FieldStack(layers=[FieldLayer()]), "Destination")
    for key in DESTINATIONS:
        assert any(l.startswith(DESTINATION_LABELS[key]) for l in labels), key


def test_every_source_is_listed(gui):
    _, labels = draw_with_combo(FieldStack(layers=[FieldLayer()]), "Source")
    for desc in field_sources.descriptors():
        assert any(l.startswith(desc.label) for l in labels), desc.key


def test_an_open_combo_keeps_its_entries_apart(gui):
    """Two layers open on the same combo would collide on the entry ids."""
    _, labels = draw_with_combo(
        FieldStack(layers=[FieldLayer(), FieldLayer()]), "Mapping")
    assert len(labels) == len(set(labels))


def test_the_trail_destination_reaches_the_list(gui):
    """The one destination that is not a force."""
    _, labels = draw_with_combo(FieldStack(layers=[FieldLayer()]), "Destination")
    assert any(l.startswith("Trail") for l in labels)


def test_an_empty_stack_still_offers_add_layer(gui):
    harness, _ = draw(FieldStack())
    assert any("Add layer" in l for l in harness.labels)


def test_the_bus_resolution_control_is_reachable(gui):
    harness, _ = draw(FieldStack())
    assert any("Resolution" in l for l in harness.labels)


def test_a_layer_header_summarises_the_layer(gui):
    """The header has to say what the layer does while it is collapsed."""
    harness, _ = draw(FieldStack(layers=[
        FieldLayer(source="noise", mapping="curl", destination="force")]))
    assert any("Noise" in l and "Curl" in l and "Force" in l
               for l in harness.labels), harness.labels


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
                 "inspect", "request_inspect", "resolution", "pass_count"):
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
    assert any("Shader##" in l or "no .frag" in l for l in harness.labels)


def test_an_image_layer_offers_a_path_field(gui):
    harness, _ = draw(FieldStack(layers=[FieldLayer(source="image")]),
                      bus=_FakeBus())
    assert any("Image##file" in l for l in harness.labels)


def test_an_unchosen_shader_layer_does_not_claim_a_file(gui):
    """The combo must not name a file the layer has not been pointed at."""
    if not field_sources.available_shader_files():
        pytest.skip("no .frag files installed")
    layer = FieldLayer(source="shader")
    draw(FieldStack(layers=[layer]), bus=_FakeBus())
    assert not layer.params.get("_file"), "the picker chose a file on its own"


def test_the_shader_combo_offers_none_and_keeps_a_missing_name():
    """Pure, because what is SELECTED inside a combo is not a label a render
    pass can see - only the list construction can be asserted."""
    from ui.field_stack_window import NO_FILE, shader_options

    options, pos = shader_options("", ["a.frag", "b.frag"])
    assert options[pos] == NO_FILE, "an unchosen layer named a file anyway"

    options, pos = shader_options("b.frag", ["a.frag", "b.frag"])
    assert options[pos] == "b.frag"

    options, pos = shader_options("gone.frag", ["a.frag"])
    assert options[pos] == "gone.frag", "a missing shader was silently swapped"
    assert "a.frag" in options, "the real files must still be reachable"


def test_a_missing_shader_file_is_offered_back_not_replaced(gui):
    """Adopting another file hides the loss: the layer's error is the only
    place a preset naming a deleted .frag gets reported."""
    layer = FieldLayer(source="shader")
    layer.params["_file"] = "definitely_not_here.frag"
    draw(FieldStack(layers=[layer]), bus=_FakeBus())
    assert layer.params["_file"] == "definitely_not_here.frag", (
        "the picker silently swapped in a different shader")


@contextlib.contextmanager
def _popups_open():
    """Force every right-click menu open for one frame.

    A popup BODY only runs while the popup is open, so every widget inside one
    is unexecuted by an ordinary render pass - and imgui_bundle raises
    TypeError on a bad signature rather than failing to compile, so the first
    real right-click takes the app down. open_popup and begin_popup_context_item
    hash the same str_id against the same window and ID stack, which is what
    lets this reach a popup nested inside a push_id.
    """
    real = imgui.begin_popup_context_item
    opened = []

    def forced(str_id=None, *a, **kw):
        if str_id:
            imgui.open_popup(str_id)
            opened.append(str_id)
        return real(str_id, *a, **kw)

    imgui.begin_popup_context_item = forced
    try:
        yield opened
    finally:
        imgui.begin_popup_context_item = real


def test_the_right_click_reset_menus_render(gui):
    """Every slider offers Reset to Default, and the body has to actually run."""
    stack = FieldStack(layers=[FieldLayer(source="noise")])
    harness = _Harness(stack, _FakeBus())
    with _popups_open() as opened:
        for _ in range(2):
            harness.labels = []
            imgui.new_frame()
            imgui.set_next_window_size(imgui.ImVec2(1200, 4000))
            imgui.begin("host", True)
            with _headers_open(True):
                harness.render_field_stack_window(collect=harness.labels)
            imgui.end()
            imgui.render()
    assert opened, "the forced-popup helper opened nothing"
    assert any("Reset to default" in l for l in harness.labels), (
        "no slider offered a reset")


def test_resetting_strength_puts_the_default_back(gui):
    from ui.field_stack_window import STRENGTH_DEFAULT
    layer = FieldLayer(source="noise", strength=3.75)
    stack = FieldStack(layers=[layer])
    harness = _Harness(stack, _FakeBus())

    real = imgui.selectable

    def click(label, *a, **kw):
        out = real(label, *a, **kw)
        return (True, out[1]) if "Reset to default" in label else out

    imgui.selectable = click
    try:
        with _popups_open():
            imgui.new_frame()
            imgui.begin("host", True)
            with _headers_open(True):
                harness.render_field_stack_window(collect=harness.labels)
            imgui.end()
            imgui.render()
    finally:
        imgui.selectable = real
    assert layer.strength == STRENGTH_DEFAULT


def test_an_image_layer_offers_a_browse_button(gui):
    """Typing a directory is not a design anyone should have to use."""
    if not file_picker.available():
        pytest.skip("no file dialog backend")
    harness, _ = draw(FieldStack(layers=[FieldLayer(source="image")]),
                      bus=_FakeBus())
    assert any("Browse" in l for l in harness.labels)


def test_a_shader_layer_offers_a_browse_button(gui):
    if not file_picker.available():
        pytest.skip("no file dialog backend")
    harness, _ = draw(FieldStack(layers=[FieldLayer(source="shader")]),
                      bus=_FakeBus())
    assert any("Browse" in l for l in harness.labels)


def test_a_chosen_file_reaches_the_layer(gui):
    """The dialog resolves on a LATER frame, so the window has to collect it."""
    layer = FieldLayer(source="image")
    stack = FieldStack(layers=[layer])
    harness = _Harness(stack, _FakeBus())

    class _Pick:
        def result(self):
            return "C:/pics/chosen.png"

    harness._pending_pick = (layer.uid, _Pick())
    imgui.new_frame()
    imgui.begin("host", True)
    with _headers_open(True):
        harness.render_field_stack_window(collect=harness.labels)
    imgui.end()
    imgui.render()
    assert layer.params.get("_file") == "C:/pics/chosen.png"


def test_a_webcam_layer_offers_a_camera_and_a_size(gui):
    """Without a picker the source is selectable and reaches no device."""
    harness, _ = draw(FieldStack(layers=[FieldLayer(source="webcam")]),
                      bus=_FakeBus())
    assert any("Camera##" in l or "no camera found" in l
               for l in harness.labels)
    assert any("Camera size" in l for l in harness.labels)


def test_the_camera_list_is_not_enumerated_every_frame(gui):
    """Enumerating shells out to ffmpeg, and this runs on every frame the
    layer is open."""
    calls = []
    real = webcam.list_devices
    webcam.list_devices = lambda *a, **k: calls.append(1) or ["Cam A"]
    try:
        draw(FieldStack(layers=[FieldLayer(source="webcam")]), bus=_FakeBus(),
             n=5)
    finally:
        webcam.list_devices = real
    assert len(calls) == 1, f"enumerated {len(calls)} times over five frames"


def test_a_brush_layer_says_where_its_paint_comes_from(gui):
    """An unpainted brush layer looks exactly like a broken one."""
    harness, _ = draw(FieldStack(layers=[FieldLayer(source="brush")]),
                      bus=_FakeBus())
    assert any("Drawing Controls" in l or "drag on the canvas" in l
               for l in harness.labels)
    assert any("Clear paint" in l for l in harness.labels)


def test_the_clear_button_asks_for_the_layers_own_destination(gui):
    for destination, flag in (("force", "_request_clear_force_field"),
                              ("strafe", "_request_clear_strafe_field")):
        layer = FieldLayer(source="brush", destination=destination)
        harness = _Harness(FieldStack(layers=[layer]), _FakeBus())
        real = imgui.button

        def click(label, *a, **kw):
            real(label, *a, **kw)
            return "Clear paint" in label

        imgui.button = click
        try:
            imgui.new_frame()
            imgui.begin("host", True)
            with _headers_open(True):
                harness.render_field_stack_window(collect=harness.labels)
            imgui.end()
            imgui.render()
        finally:
            imgui.button = real
        assert getattr(harness, flag, False), f"{destination} did not ask"


def test_the_panel_asks_for_a_view_and_never_renders_one(gui):
    """A GL pass inside a window body leaves a framebuffer bound that is not
    the one imgui is about to draw into, and every window vanishes at once."""
    stack = FieldStack(layers=[FieldLayer(source="noise")])
    bus = _FakeBus()
    harness = _Harness(stack, bus)
    harness._inspect_uid = stack.layers[0].uid
    imgui.new_frame()
    imgui.begin("host", True)
    with _headers_open(True):
        harness.render_field_stack_window(collect=harness.labels)
    imgui.end()
    imgui.render()
    assert bus.asked == (stack.layers[0].uid, "source")


def test_a_source_setting_sits_next_to_the_source_combo(gui):
    """The camera picker was six controls below the Source that asks for it."""
    harness, _ = draw(FieldStack(layers=[FieldLayer(source="webcam")]),
                      bus=_FakeBus())
    order = harness.labels

    def at(fragment):
        return next(i for i, l in enumerate(order) if fragment in l)

    source = at("Source##")
    camera = next(i for i, l in enumerate(order)
                  if "Camera##" in l or "no camera found" in l)
    mapping = at("Mapping##")
    assert source < camera < mapping, (
        f"source at {source}, camera at {camera}, mapping at {mapping}")


def test_the_layer_body_is_grouped_into_sections(gui):
    harness, _ = draw(FieldStack(layers=[FieldLayer(source="noise")]),
                      bus=_FakeBus())
    for heading in ("Picture##", "Becomes##", "Drives##"):
        assert any(heading in l for l in harness.labels), f"{heading} missing"


def test_the_global_switch_and_lock_are_reachable(gui):
    harness, _ = draw(FieldStack(layers=[FieldLayer()]), bus=_FakeBus())
    assert any("Inject##" in l for l in harness.labels)
    assert any("Keep on preset load" in l for l in harness.labels)
