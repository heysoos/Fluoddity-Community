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
                                shaper=ShaperParams(kind="phase")))
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


# --- the preset row ---------------------------------------------------------

def _with_presets(names, monkeypatch):
    from services import audio_rig_io

    monkeypatch.setattr(audio_rig_io, "list_rigs", lambda: list(names))
    return _host(True)


def test_the_preset_row_renders_with_nothing_saved_yet(monkeypatch):
    imgui, host = _with_presets([], monkeypatch)
    _draw(imgui, host)


def test_the_preset_row_renders_with_a_selection(monkeypatch):
    imgui, host = _with_presets(["kelp", "reef"], monkeypatch)
    host.state.audio.preset_name = "reef"
    _draw(imgui, host)


def test_a_selection_that_is_no_longer_on_disk_still_renders(monkeypatch):
    """The folder is another window's to change, and a stale name must not
    index past the list."""
    imgui, host = _with_presets(["kelp"], monkeypatch)
    host.state.audio.preset_name = "deleted"
    _draw(imgui, host)


def test_a_notice_renders_and_refreshes_the_list(monkeypatch):
    """Every save and load sets a notice, which is exactly when the folder can
    have gained a file."""
    imgui, host = _with_presets(["kelp"], monkeypatch)
    _draw(imgui, host)
    monkeypatch.setattr("services.audio_rig_io.list_rigs",
                        lambda: ["kelp", "reef"])
    host.state.audio.notice = "Rig saved as reef"
    _draw(imgui, host)
    assert host._audio_preset_names() == ["kelp", "reef"]


def test_a_warning_renders(monkeypatch):
    imgui, host = _with_presets(["kelp"], monkeypatch)
    host.state.audio.warning = "Could not write the rig"
    _draw(imgui, host)


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


@pytest.mark.parametrize("kind", ["none", "smooth", "gate", "envelope", "phase",
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
    """A phase shaper integrates a steady band into a travelling wave; a drawer
    drawing the raw band would show none of that."""
    from services.audio_mapping import Mapping, TargetDef, modulate
    from services.audio_shapers import ShaperParams

    t = TargetDef("K", "K", "physics", 0.0, 1.0, None, None)
    m = Mapping(signal="bass", target="K",
                shaper=ShaperParams(kind="phase", rate=8.0, wave="sine"))
    states, seen = {}, []
    for _ in range(40):
        shaped = {}
        modulate({"K": 0.5}, [t], [m], {"bass": 0.5}, states, {}, 1.0,
                 1 / 60.0, set(), shaped)
        seen.append(shaped[m.uid])
    assert max(seen) - min(seen) > 0.5, "the shaper's travel was not reported"


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
    ast.shaped = {m.uid: 0.5 for m in ast.mappings}
    _draw(imgui, host)
    rings = host._audio_shaped_rings()
    assert set(rings) == set(ast.shaped)


def test_a_mapping_that_stops_being_applied_drops_its_ring():
    imgui, host = _bound_host(open_band="mid")
    ast = host.state.audio
    ast.shaped = {m.uid: 0.5 for m in ast.mappings}
    _draw(imgui, host)
    ast.shaped = {}
    _draw(imgui, host)
    assert host._audio_shaped_rings() == {}


# --- the two bypasses --------------------------------------------------------

def _runtime_with(level=0.5):
    """A runtime fed a constant signal, with one physics and one brain row."""
    from services.audio_mapping import Mapping
    from services.audio_runtime import AudioRuntime
    from services.audio_analysis import SIGNAL_NAMES
    from state import UIState

    class _Fake:
        signals = {n: level for n in SIGNAL_NAMES}

    st = UIState()
    st.audio.enabled = True
    st.audio.mappings.append(Mapping(signal="bass", target="SENSOR_GAIN",
                                     mode="add", depth=0.5))
    st.sim.SENSOR_GAIN = 1.0
    rt = AudioRuntime()
    rt.capture.status = "active"
    rt.capture.snapshot = lambda: _Fake()
    return st, rt


def test_the_master_switch_stops_modulation_but_not_capture():
    st, rt = _runtime_with()
    moved, _b = rt.update(st, 1 / 60, None, None)
    assert moved is not st.sim, "nothing was modulated to begin with"

    st.audio.modulate = False
    same, _b = rt.update(st, 1 / 60, None, None)
    assert same is st.sim
    assert st.audio.snapshot is not None, "capture must keep running"
    assert st.audio.status == "active"
    rt.close()


def test_muting_a_row_silences_every_band_on_it():
    st, rt = _runtime_with()
    from services.audio_mapping import Mapping
    st.audio.mappings.append(Mapping(signal="hi", target="SENSOR_GAIN",
                                     mode="add", depth=0.5))
    moved, _b = rt.update(st, 1 / 60, None, None)
    assert moved.SENSOR_GAIN != pytest.approx(1.0)

    st.audio.muted["SENSOR_GAIN"] = True
    same, _b = rt.update(st, 1 / 60, None, None)
    assert same is st.sim
    rt.close()


def test_muting_a_row_leaves_each_band_own_enabled_flag_alone():
    """Clearing the mappings' flags instead would resurrect the bands the user
    had switched off individually when the row came back."""
    st, rt = _runtime_with()
    st.audio.mappings[0].enabled = False
    st.audio.muted["SENSOR_GAIN"] = True
    rt.update(st, 1 / 60, None, None)
    st.audio.muted["SENSOR_GAIN"] = False
    rt.update(st, 1 / 60, None, None)
    assert st.audio.mappings[0].enabled is False
    rt.close()


def test_a_muted_row_draws_no_overlay_on_its_slider():
    st, rt = _runtime_with()
    st.audio.muted["SENSOR_GAIN"] = True
    moved, _b = rt.update(st, 1 / 60, None, None)
    assert rt.overlays(st, moved) == {}
    rt.close()


def test_a_bypassed_rig_empties_the_drawer_traces():
    """Left alone, `shaped` would hold its last frame forever and the drawer
    would show a frozen line rather than nothing."""
    st, rt = _runtime_with()
    rt.update(st, 1 / 60, None, None)
    assert st.audio.shaped
    st.audio.modulate = False
    rt.update(st, 1 / 60, None, None)
    assert st.audio.shaped == {}
    rt.close()


def test_a_brain_row_mute_does_not_leak_into_another_modality():
    """mlp and lenia both declare `w_scale`, and their mappings are already
    kept apart by modality."""
    from services.audio_runtime import muted_targets
    from state.audio_in_state import AudioInState

    a = AudioInState()
    a.muted = {"mlp:w_scale": True, "SENSOR_GAIN": True}
    assert muted_targets(a, "mlp:") == {"w_scale"}
    assert muted_targets(a, "lenia:") == set()
    # A physics sweep must not pick up the brain's namespaced keys either.
    assert muted_targets(a) == {"SENSOR_GAIN"}


def test_both_switches_survive_a_save_and_load():
    from state.audio_in_state import AudioInState, apply_dict, to_dict
    a = AudioInState()
    a.modulate = False
    a.muted = {"DRAG": True, "SENSOR_GAIN": False}
    b = AudioInState()
    apply_dict(b, to_dict(a))
    assert b.modulate is False
    assert b.muted.get("DRAG") is True
    assert not b.muted.get("SENSOR_GAIN")


def test_the_panel_renders_with_a_muted_row():
    imgui, host = _bound_host(open_band="bass")
    host.state.audio.muted["SENSOR_GAIN"] = True
    _draw(imgui, host)


# --- right-click to reset ----------------------------------------------------

def test_every_slider_in_the_panel_offers_a_reset():
    """A bare imgui.slider_float has no context menu, so one left behind is a
    control the user cannot put back."""
    src = SRC.read_text(encoding="utf-8")
    assert "imgui.slider_float(" not in src.replace(
        "changed, v = imgui.slider_float(", "")
    assert "begin_popup_context_item" in src


def _with_popups_open(imgui, host, frames=2):
    """Render with every context menu OPEN, so its body actually runs.

    A popup body executes only while the popup is open, so nothing inside one
    is reached by an ordinary render pass - which is how a selectable() missing
    its p_selected argument got all the way to a crash in the user's hands.
    open_popup hashes str_id against the same window and ID stack that
    begin_popup_context_item does, so opening it here targets the right popup
    even inside the row's push_id.
    """
    real = imgui.begin_popup_context_item

    def spy(str_id=None, *a, **kw):
        if str_id:
            imgui.open_popup(str_id)
        return real(str_id, *a, **kw)

    imgui.begin_popup_context_item = spy
    try:
        _draw(imgui, host, frames=frames)
    finally:
        imgui.begin_popup_context_item = real


@pytest.mark.parametrize("kind", ["none", "smooth", "gate", "envelope", "phase",
                                  "sample_hold"])
def test_every_reset_menu_in_the_drawer_actually_renders(kind):
    """Covers Depth, Gain and whichever shaper fields this kind exposes."""
    imgui, host = _bound_host(open_band="bass")
    host.state.audio.mappings[0].shaper.kind = kind
    _with_popups_open(imgui, host)


def test_the_strength_reset_menu_actually_renders():
    imgui, host = _bound_host(open_band="")
    _with_popups_open(imgui, host)


def test_the_forced_popup_helper_really_opens_something():
    """Guards the guard: if the spy stopped opening popups, every test above
    would pass without executing a single popup body."""
    imgui, host = _bound_host(open_band="bass")
    seen = []
    real = imgui.selectable

    def spy(*a, **kw):
        seen.append(a[0] if a else "")
        return real(*a, **kw)

    imgui.selectable = spy
    try:
        _with_popups_open(imgui, host)
    finally:
        imgui.selectable = real
    assert any(s.startswith("Reset to") for s in seen), (
        "no reset item was drawn, so the popup bodies never ran")


# --- the Bands section ------------------------------------------------------

def _with_headers_open(imgui, host, frames=2):
    """Render with every collapsing header OPEN.

    A header's body does not run while it is shut, so the measure combos and
    the dB windows are unexecuted by an ordinary pass. The real header is
    still drawn, so its own call is covered too.
    """
    real = imgui.collapsing_header

    def spy(*a, **kw):
        return real(*a, **kw) or True

    imgui.collapsing_header = spy
    try:
        _draw(imgui, host, frames=frames)
    finally:
        imgui.collapsing_header = real


def test_the_bands_section_renders():
    imgui, host = _bound_host(open_band="bass")
    _with_headers_open(imgui, host)


@pytest.mark.parametrize("measure", ["power", "rms", "peak", "mean_db"])
def test_the_bands_section_renders_for_every_measure(measure):
    imgui, host = _bound_host(open_band="bass")
    host.state.audio.bands = {"hi": {"measure": measure, "floor": -50.0,
                                     "ceiling": -10.0}}
    _with_headers_open(imgui, host)


def test_the_forced_header_helper_really_opens_something():
    """Guards the guard: without this the tests above would pass having drawn
    nothing but the header itself."""
    imgui, host = _bound_host(open_band="bass")
    seen = []
    real = imgui.combo

    def spy(*a, **kw):
        seen.append(a[0] if a else "")
        return real(*a, **kw)

    imgui.combo = spy
    try:
        _with_headers_open(imgui, host)
    finally:
        imgui.combo = real
    assert any(s.startswith("##measure_") for s in seen), (
        "no measure combo was drawn, so the section body never ran")


def test_writing_one_band_leaves_the_others_on_their_defaults():
    """A band is absent from the rig until it is touched, so a partial write
    must not strand the rest without a floor."""
    from services.audio_analysis import MEASURE_WINDOWS, effective_bands
    from ui.audio_reactive_window import AudioReactiveWindowMixin

    _imgui, host = _host(True)
    AudioReactiveWindowMixin._write_band(host.state.audio, "hi", floor=-30.0)
    bands = effective_bands(host.state.audio.bands)
    assert bands["hi"]["floor"] == -30.0
    assert bands["hi"]["ceiling"] == MEASURE_WINDOWS["power"][1]
    assert bands["bass"] == effective_bands({})["bass"]


def test_a_reset_returns_the_dataclass_default():
    from services.audio_mapping import Mapping
    from services.audio_shapers import ShaperParams
    assert Mapping(signal="bass", target="K").depth == pytest.approx(0.5)
    assert Mapping(signal="bass", target="K").gain == pytest.approx(1.0)
    assert ShaperParams().rate == pytest.approx(1.0)


def test_the_panel_reads_its_snapshot_from_state_not_from_a_service():
    """The UI is passive; reaching into the capture thread is what left the
    spectrum blank when nothing assigned the runtime to the UI."""
    src = SRC.read_text(encoding="utf-8")
    assert "audio_runtime" not in src
    assert "ast.snapshot" in src or 'getattr(ast, "snapshot"' in src
