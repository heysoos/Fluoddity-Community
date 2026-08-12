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
            self.audio_runtime = None

        def _delayed_tooltip(self, text):
            pass

    return imgui, Host()


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
