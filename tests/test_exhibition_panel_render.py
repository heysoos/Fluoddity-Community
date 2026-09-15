"""The exhibition controls are DRAWN, not merely called.

Same harness as test_perform_window_render: the host is taller than the
panel so nothing falls off the bottom and draws no vertices.
"""
import pytest
from imgui_bundle import imgui

from state.set_list import KnobBinding, SetList, Slot
from tests.test_perform_window_render import (  # noqa: F401
    Harness, _labels, frame, gui, host_only, no_disk_or_midi, two_displays)


def _show(h):
    h.state.perform.set_list = SetList(
        name="open house",
        slots=[Slot("1", "Core", "reef"), Slot("2", "Exhibition", "gone")],
        knobs=[KnobBinding(cc=0, param="SENSOR_GAIN"), KnobBinding(cc=1)])
    h.config_files_by_category = {"Core": ["reef", "karst"],
                                  "Custom": ["mine"]}
    return h


def test_the_lock_the_set_list_and_the_knob_table_are_drawn(
        gui, monkeypatch, two_displays):
    h = _show(Harness())
    labels = _labels(monkeypatch, h.render_perform_window)
    assert "Lock keyboard##exhibition" in labels
    assert "Add key##exhibition" in labels
    assert "Device##midi" in labels
    assert "Add knob##exhibition" in labels
    assert labels.count("##key") == 2
    assert labels.count("##param") == 2


def test_a_saved_set_list_is_offered_to_open(gui, monkeypatch, two_displays):
    monkeypatch.setattr("ui.exhibition_panel.set_list_io.list_set_lists",
                        lambda root=None: ["open house", "late show"])
    h = _show(Harness())
    labels = _labels(monkeypatch, h.render_perform_window)
    assert "Open##setlist" in labels


def test_it_draws_with_no_midi_packages(gui, monkeypatch, two_displays):
    monkeypatch.setattr("ui.exhibition_panel.midi_control.available",
                        lambda: False)
    h = _show(Harness())
    labels = _labels(monkeypatch, h.render_perform_window)
    assert "Device##midi" not in labels
    assert "Add knob##exhibition" in labels


def test_a_remembered_device_that_is_gone_stays_in_the_combo(
        gui, monkeypatch, two_displays):
    h = _show(Harness())
    h.state.preferences.midi_device = "Midi Fighter Twister"
    seen = {}
    real = imgui.combo

    def combo(label, cur, items, *a, **k):
        seen[label] = (cur, list(items))
        return real(label, cur, items, *a, **k)

    monkeypatch.setattr(imgui, "combo", combo)
    frame(h.render_perform_window)
    cur, items = seen["Device##midi"]
    assert items[cur] == "Midi Fighter Twister"


def test_a_slot_naming_a_missing_preset_is_offered_back(
        gui, monkeypatch, two_displays):
    h = _show(Harness())
    seen = []
    real = imgui.combo

    def combo(label, cur, items, *a, **k):
        if label == "##preset":
            seen.append(list(items)[cur])
        return real(label, cur, items, *a, **k)

    monkeypatch.setattr(imgui, "combo", combo)
    frame(h.render_perform_window)
    assert seen[-2:] == ["Core / reef", "Exhibition / gone (missing)"]


def test_no_two_visible_widgets_share_an_id_with_rows_present(
        gui, monkeypatch, two_displays):
    h = _show(Harness())
    seen = {}
    for fn_name in ("button", "small_button", "checkbox", "combo",
                    "input_text", "input_int", "slider_float", "slider_int"):
        real = getattr(imgui, fn_name)

        def wrapper(label, *a, _real=real, **kw):
            seen.setdefault(imgui.get_id(label), []).append(label)
            return _real(label, *a, **kw)

        monkeypatch.setattr(imgui, fn_name, wrapper)

    def render():
        seen.clear()
        h.render_perform_window()

    frame(render)
    clashes = {i: v for i, v in seen.items() if len(v) > 1}
    assert not clashes, sorted(clashes.values())
