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
        from services.audio_analysis import SIGNAL_NAMES, mel_bar_bands
        self.signals = {n: 0.5 for n in SIGNAL_NAMES}
        self.mel = np.linspace(0, 1, 40).astype(np.float32)
        self.seq = 1
        self.mel_bands = mel_bar_bands(48000.0, 40)


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


# --- what stops the drawing looking like noise -------------------------------

def test_a_trace_never_draws_more_samples_than_it_has_pixels():
    """plot_lines spreads what it is given across the whole graph, so an
    oversampled sparkline reads as noise however smooth the signal is."""
    from ui.audio_reactive_window import TRACE_LEN, visible_samples
    full = np.arange(TRACE_LEN, dtype=np.float32)
    assert visible_samples(full, 56).size == 56
    assert visible_samples(full, 1000).size == TRACE_LEN


def test_a_trace_keeps_the_NEWEST_samples_when_it_has_to_drop_some():
    from ui.audio_reactive_window import visible_samples
    full = np.arange(100, dtype=np.float32)
    assert visible_samples(full, 10)[-1] == pytest.approx(99.0)


def test_a_trace_in_a_zero_width_widget_still_has_two_points():
    from ui.audio_reactive_window import visible_samples
    assert visible_samples(np.zeros(64, dtype=np.float32), 0).size == 2


def test_the_spectrum_is_bars_on_a_fixed_scale():
    """Left to itself plot_histogram rescales to the frame's own extremes
    every frame, so the graph heaves about when nothing has changed."""
    src = SRC.read_text(encoding="utf-8")
    assert "plot_histogram" in src
    call = src.split("plot_histogram(")[1]
    assert "scale_min=0.0" in call and "scale_max=1.0" in call


def test_the_spectrum_colours_every_bar_by_a_real_band():
    """A bar's colour comes from the band its centre frequency falls in, which
    only the analyser knows - it placed the mel axis for this device's rate."""
    from services.audio_analysis import BAND_EDGES_HZ, mel_bar_bands
    for rate in (44100.0, 48000.0, 96000.0):
        idx = mel_bar_bands(rate, 40)
        assert idx.size == 40
        assert idx.min() >= 0 and idx.max() < len(BAND_EDGES_HZ)
        assert np.all(np.diff(idx) >= 0), "bands must run low to high"


# --- the drawer has to show what the shaper does -----------------------------

def test_modulate_reports_each_mapping_post_shaper_signal():
    """An LFO turns a steady band into an oscillation; a drawer drawing the
    raw band would show none of that."""
    from services.audio_mapping import Mapping, TargetDef, modulate
    from services.audio_shapers import ShaperParams

    t = TargetDef("K", "K", "physics", 0.0, 1.0, None, None)
    m = Mapping(signal="bass", target="K",
                shaper=ShaperParams(kind="lfo", rate_min=8.0, rate_max=8.0))
    states, seen = {}, []
    for _ in range(40):
        shaped = {}
        modulate({"K": 0.5}, [t], [m], {"bass": 0.5}, states, {}, 1.0,
                 1 / 60.0, set(), shaped)
        seen.append(shaped[id(m)])
    assert max(seen) - min(seen) > 0.5, "the LFO's swing was not reported"


def test_modulate_reports_nothing_when_no_dict_is_offered():
    """The overlay probe runs modulate at full scale; it must not overwrite
    what the live path recorded."""
    from services.audio_mapping import Mapping, TargetDef, modulate
    t = TargetDef("K", "K", "physics", 0.0, 1.0, None, None)
    modulate({"K": 0.5}, [t], [Mapping(signal="bass", target="K")],
             {"bass": 0.5}, {}, {}, 1.0, 1 / 60.0, set())


def test_the_reach_probe_ignores_the_shaper_so_it_shows_the_whole_swing():
    """A fresh shaper state is mid-attack or mid-phase, so running one to find
    the reach reported roughly half the swing the mapping really has."""
    from services.audio_mapping import Mapping, TargetDef, modulate
    from services.audio_shapers import ShaperParams

    t = TargetDef("K", "K", "physics", 0.0, 1.0, None, None)
    m = Mapping(signal="bass", target="K", depth=1.0,
                shaper=ShaperParams(kind="smooth", attack=0.5))
    args = ({"K": 0.0}, [t], [m], {"bass": 1.0})
    reach = modulate(*args, {}, {}, 1.0, 1 / 60.0, set(),
                     apply_shapers=False)["K"]
    mid_attack = modulate(*args, {}, {}, 1.0, 1 / 60.0, set())["K"]
    assert reach == pytest.approx(1.0)
    assert mid_attack < 0.5 * reach


def test_a_disabled_mapping_reports_no_shaped_signal():
    from services.audio_mapping import Mapping, TargetDef, modulate
    t = TargetDef("K", "K", "physics", 0.0, 1.0, None, None)
    m = Mapping(signal="bass", target="K", enabled=False)
    shaped = {}
    modulate({"K": 0.5}, [t], [m], {"bass": 0.5}, {}, {}, 1.0, 1 / 60.0,
             set(), shaped)
    assert shaped == {}


def test_the_drawer_draws_the_shaped_ring_it_was_given():
    imgui, host = _bound_host(open_band="mid")
    ast = host.state.audio
    ast.shaped = {id(m): 0.5 for m in ast.mappings}
    _draw(imgui, host)
    rings = host._audio_shaped_rings()
    assert set(rings) == set(ast.shaped)


def test_a_mapping_that_stops_being_applied_drops_its_ring():
    imgui, host = _bound_host(open_band="mid")
    ast = host.state.audio
    ast.shaped = {id(m): 0.5 for m in ast.mappings}
    _draw(imgui, host)
    ast.shaped = {}
    _draw(imgui, host)
    assert host._audio_shaped_rings() == {}


def test_the_panel_reads_its_snapshot_from_state_not_from_a_service():
    """The UI is passive; reaching into the capture thread is what left the
    spectrum blank when nothing assigned the runtime to the UI."""
    src = SRC.read_text(encoding="utf-8")
    assert "audio_runtime" not in src
    assert "ast.snapshot" in src or 'getattr(ast, "snapshot"' in src
