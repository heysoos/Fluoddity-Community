"""No widget label in the tournament or archive panels may be wider than the
room reserved for it.

ImGui draws a label to the RIGHT of its widget, so a label wider than
layout.WIDEST_LABEL is silently clipped and the user cannot read what the
slider does. Measured in real pixels against the real font, because the labels
are a proportional typeface and character counts do not predict width.
"""
import re

import pytest
from imgui_bundle import imgui

from ui import layout

# The widgets that put their label on the right. Buttons and checkboxes place
# their own text inside or beside themselves and size to it.
_LABELLED = ("slider_int", "slider_float", "combo", "input_text", "drag_float")

_PANELS = ("ui/archive_window.py", "ui/auto_tournament_window.py",
           "ui/tournament_window.py", "ui/audio_reactive_window.py")


@pytest.fixture(scope="module")
def gui():
    imgui.create_context()
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(1280, 900)
    io.delta_time = 1.0 / 60.0
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    imgui.new_frame()
    imgui.render()
    yield
    imgui.destroy_context()


def _labels(path):
    """Every literal label passed to a right-labelled widget in `path`."""
    src = open(path, encoding="utf-8").read()
    pat = re.compile(r"imgui\.(?:" + "|".join(_LABELLED) + r")\(\s*[\"']([^\"']*)[\"']")
    out = []
    for raw in pat.findall(src):
        # "##suffix" is identity, not display text; "##name" alone shows nothing.
        text = raw.split("##")[0]
        if text:
            out.append(text)
    return out


def test_the_scan_finds_the_labels_it_is_meant_to_guard():
    """A regex that matched nothing would pass every assertion below."""
    found = {lbl for p in _PANELS for lbl in _labels(p)}
    assert "Grid" in found and "Capacity" in found
    assert len(found) > 20


def test_no_label_is_wider_than_the_room_reserved_for_it(gui):
    imgui.new_frame()
    room = imgui.calc_text_size(layout.WIDEST_LABEL).x
    too_wide = {lbl: imgui.calc_text_size(lbl).x
                for p in _PANELS for lbl in _labels(p)
                if imgui.calc_text_size(lbl).x > room}
    imgui.render()
    assert not too_wide, (
        f"wider than layout.WIDEST_LABEL ({layout.WIDEST_LABEL!r}, {room:.0f}px) "
        f"- shorten the label or update WIDEST_LABEL: {too_wide}")


def test_no_physics_label_is_wider_than_its_own_reserved_room(gui):
    """The Physics window reserves WIDEST_PHYSICS_LABEL, not WIDEST_LABEL: its
    sliders carry a '[L]' prefix when alt-locked, which widens the label."""
    from ui.physics_params import PHYSICS_PARAMS

    imgui.new_frame()
    room = imgui.calc_text_size(layout.WIDEST_PHYSICS_LABEL).x
    rendered = [p.label for p in PHYSICS_PARAMS]
    rendered += ["[L]" + p.label for p in PHYSICS_PARAMS]
    rendered += _labels("ui/physics_window.py")
    too_wide = {lbl: imgui.calc_text_size(lbl).x
                for lbl in rendered if imgui.calc_text_size(lbl).x > room}
    imgui.render()
    assert not too_wide, (
        f"wider than layout.WIDEST_PHYSICS_LABEL "
        f"({layout.WIDEST_PHYSICS_LABEL!r}, {room:.0f}px): {too_wide}")


def test_the_reserved_width_still_leaves_a_usable_widget(gui):
    """A negative item width is clamped to 1px by ImGui, so reserving too much
    would turn every slider into a sliver rather than failing loudly."""
    imgui.new_frame()
    room = imgui.calc_text_size(layout.WIDEST_LABEL).x
    imgui.render()
    assert room < layout.MIN_PANEL_WIDTH * 0.5
