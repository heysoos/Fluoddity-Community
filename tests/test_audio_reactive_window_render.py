"""Render smoke, ID collisions, and the drawing rules the panel must follow."""
import ast
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parent.parent / "ui" / "audio_reactive_window.py"


def test_the_mixin_is_part_of_UI():
    from ui import UI
    from ui.audio_reactive_window import AudioReactiveWindowMixin
    assert issubclass(UI, AudioReactiveWindowMixin)


def test_it_does_not_collide_with_the_sonification_window():
    """The sonification branch owns ui/audio_window.py, class AudioWindowMixin
    and render_audio_window(). Both features will land, and identical names
    would put one mixin in front of the other in the MRO with nothing raising.
    """
    from ui.audio_reactive_window import AudioReactiveWindowMixin
    assert SRC.name == "audio_reactive_window.py"
    assert hasattr(AudioReactiveWindowMixin, "render_audio_reactive_window")
    assert not hasattr(AudioReactiveWindowMixin, "render_audio_window")


def test_every_signal_has_a_colour():
    from services.audio_analysis import SIGNAL_NAMES
    from ui.audio_reactive_window import SIGNAL_COLORS
    assert set(SIGNAL_COLORS) == set(SIGNAL_NAMES)


def test_the_ring_is_float32_numpy_all_the_way_to_the_widget():
    from ui.audio_reactive_window import TraceRing
    r = TraceRing(64)
    r.push(0.5)
    assert isinstance(r.values, np.ndarray)
    assert r.values.dtype == np.float32
    assert r.values.size == 64


def test_the_ring_keeps_the_newest_sample_last():
    from ui.audio_reactive_window import TraceRing
    r = TraceRing(4)
    for v in (0.1, 0.2, 0.3, 0.4, 0.5):
        r.push(v)
    assert r.values[-1] == pytest.approx(0.5)
    assert r.values[0] == pytest.approx(0.2)


def test_traces_never_use_add_line():
    """add_line per segment costs twenty times what plot_lines does."""
    src = SRC.read_text(encoding="utf-8")
    assert "add_line" not in src


def test_traces_go_through_plot_lines():
    assert "plot_lines" in SRC.read_text(encoding="utf-8")


def test_no_visible_label_is_used_twice():
    """Two visible items hashing to one ImGui ID puts a modal over the app and
    stops one of them responding to the mouse."""
    src = SRC.read_text(encoding="utf-8")
    tree = ast.parse(src)
    labels: dict[str, int] = {}
    id_makers = {"button", "collapsing_header", "combo", "checkbox",
                 "slider_float", "begin_tab_item", "selectable"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
        if name not in id_makers or not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            if "##" in first.value or not first.value:
                continue
            labels[first.value] = labels.get(first.value, 0) + 1
    dupes = {k: v for k, v in labels.items() if v > 1}
    assert not dupes, f"duplicate ImGui ids: {dupes}"


def _host(show: bool):
    from imgui_bundle import imgui

    from state import UIState
    from ui.audio_reactive_window import AudioReactiveWindowMixin

    imgui.create_context()
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(1280, 800)
    io.delta_time = 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures.value

    class Host(AudioReactiveWindowMixin):
        def __init__(self):
            self.state = UIState()
            self.state.audio.show_window = show

        def _delayed_tooltip(self, text):
            pass

    return imgui, Host()


class _Snap:
    """Stands in for a SignalSnapshot."""

    def __init__(self):
        from services.audio_analysis import SIGNAL_NAMES
        self.signals = {n: 0.5 for n in SIGNAL_NAMES}
        self.mel = np.linspace(0, 1, 40).astype(np.float32)
        self.seq = 1


def _bound_host(open_target="SENSOR_GAIN", open_band="bass"):
    """A panel with two bands bound to one target and its drawer open."""
    from services.audio_mapping import Mapping
    from services.audio_shapers import ShaperParams
    from ui.audio_reactive_window import SIGNAL_COLORS

    imgui, host = _host(True)
    ast = host.state.audio
    ast.snapshot = _Snap()
    ast.status = "active"
    ast.enabled = True
    ast.mappings.append(Mapping(signal="bass", target="SENSOR_GAIN"))
    ast.mappings.append(Mapping(signal="mid", target="SENSOR_GAIN",
                                mode="multiply",
                                shaper=ShaperParams(kind="lfo")))
    ast.open_target = open_target
    ast.open_band = open_band
    host.audio_overlays = {
        "SENSOR_GAIN": {"lo": 0.0, "hi": 10.0, "base": 1.0, "live": 3.5,
                        "reach": 6.0, "color": SIGNAL_COLORS["bass"]},
    }
    return imgui, host


def test_the_window_renders_headless():
    imgui, host = _host(True)
    imgui.new_frame()
    host.render_audio_reactive_window()
    imgui.end_frame()
    imgui.render()


def test_a_closed_window_renders_nothing_and_does_not_raise():
    imgui, host = _host(False)
    imgui.new_frame()
    host.render_audio_reactive_window()
    imgui.end_frame()
    imgui.render()


# --- the drawer, which is where the API misuse would hide --------------------

def _draw(imgui, host, frames=2):
    for _ in range(frames):
        imgui.new_frame()
        host.render_audio_reactive_window()
        imgui.end_frame()
        imgui.render()


def test_an_open_drawer_on_a_band_tab_renders():
    imgui, host = _bound_host(open_band="bass")
    _draw(imgui, host)


def test_the_total_tab_renders_with_overlaid_traces():
    imgui, host = _bound_host(open_band="")
    _draw(imgui, host)


@pytest.mark.parametrize("kind", ["none", "smooth", "gate", "envelope", "lfo",
                                  "sample_hold"])
def test_every_shaper_kind_renders_its_own_controls(kind):
    """Each kind exposes a different set of fields; a missing range entry would
    only ever show up when that kind is selected."""
    imgui, host = _bound_host(open_band="bass")
    host.state.audio.mappings[0].shaper.kind = kind
    _draw(imgui, host)


def test_every_shaper_kind_has_a_field_list():
    from services.audio_shapers import SHAPER_KINDS
    from ui.audio_reactive_window import SHAPER_FIELDS
    assert set(SHAPER_FIELDS) == set(SHAPER_KINDS)


def test_every_shaper_field_has_a_slider_range():
    from ui.audio_reactive_window import _FIELD_RANGE, SHAPER_FIELDS
    for kind, fields in SHAPER_FIELDS.items():
        for f in fields:
            if f == "wave":
                continue
            assert f in _FIELD_RANGE, f"{kind}.{f} has no range"


def test_the_shaper_fields_are_ones_the_shaper_actually_reads():
    """An attack slider on a gate would be a control that does nothing."""
    import inspect

    from services import audio_shapers
    from ui.audio_reactive_window import SHAPER_FIELDS

    src = inspect.getsource(audio_shapers.ShaperState.apply)
    blocks = src.split('if kind == "')
    for kind, fields in SHAPER_FIELDS.items():
        block = next((b for b in blocks if b.startswith(kind + '"')), None)
        if block is None:
            continue
        for f in fields:
            assert f"p.{f}" in block, f"{kind} does not read {f}"


def test_only_one_drawer_is_open_at_a_time():
    """open_target is a single key, so the panel cannot show two drawers."""
    from state.audio_in_state import AudioInState
    assert isinstance(AudioInState().open_target, str)


def test_a_row_with_no_mapping_draws_no_drawer():
    imgui, host = _bound_host(open_target="DRAG")
    _draw(imgui, host)


def test_the_panel_reads_its_snapshot_from_state_not_from_a_service():
    """The UI is passive; reaching into the capture thread is what left the
    spectrum blank when nothing assigned the runtime to the UI."""
    src = SRC.read_text(encoding="utf-8")
    assert "audio_runtime" not in src
    assert "ast.snapshot" in src or 'getattr(ast, "snapshot"' in src
