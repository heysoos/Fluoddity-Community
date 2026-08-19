"""A tab bar inside a window is drawn quieter than the window's own tabs.

A DOCKED window's title IS a tab strip, so the mode tabs four pixels below it
were the same colour at the same height and the eye could not tell which one
was the window. The window level is untouched; the level below steps back.

The pairing is what the tests are really for: ImGui requires end_tab_bar only
when begin returned true, while the four style colours must be popped either
way. An unbalanced pop corrupts every widget drawn after it in the frame, not
the tab bar - so it would surface anywhere but here.
"""
import pytest
from imgui_bundle import imgui

from ui import layout


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


def _colour(which):
    c = imgui.get_style().color_(which)
    return (c.x, c.y, c.z, c.w)


def _in_frame(body):
    """Run `body` inside a real window, and report the tab colours it saw."""
    seen = {}
    imgui.new_frame()
    imgui.set_next_window_size(imgui.ImVec2(600, 400))
    imgui.begin("host", True)
    outer = _colour(imgui.Col_.tab)
    body(seen)
    imgui.end()
    imgui.render()
    return outer, seen


# -- the pairing ------------------------------------------------------------

def test_the_colour_stack_is_balanced_when_the_bar_opens(gui):
    def body(seen):
        with layout.sub_tab_bar("open_bar") as opened:
            seen["opened"] = opened
            if opened:
                if imgui.begin_tab_item("One")[0]:
                    imgui.text("body")
                    imgui.end_tab_item()
        seen["after"] = _colour(imgui.Col_.tab)

    outer, seen = _in_frame(body)
    assert seen["opened"], "a bar in a visible window should open"
    assert seen["after"] == outer, "style colours leaked out of the bar"


def test_the_colour_stack_is_balanced_when_the_body_raises(gui):
    class Boom(Exception):
        pass

    def body(seen):
        with pytest.raises(Boom):
            with layout.sub_tab_bar("raising_bar"):
                raise Boom
        seen["after"] = _colour(imgui.Col_.tab)

    outer, seen = _in_frame(body)
    assert seen["after"] == outer, (
        "a raising body must still pop its colours, or every widget after it "
        "in the frame is drawn in the wrong palette")


# -- it is actually quieter -------------------------------------------------

def test_a_sub_tab_is_dimmer_than_the_window_level_it_sits_under(gui):
    def body(seen):
        with layout.sub_tab_bar("dim_bar"):
            seen["tab"] = _colour(imgui.Col_.tab)
            seen["selected"] = _colour(imgui.Col_.tab_selected)

    outer, seen = _in_frame(body)
    lum = lambda c: 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
    assert lum(seen["tab"]) < lum(outer), "the sub bar must be darker"
    assert lum(seen["selected"]) < lum(outer)


def test_the_selected_tab_is_marked_by_its_overline_not_by_brightness(gui):
    """Brightness is what makes two strips compete for the same level, so the
    selection is carried by the accent line instead."""
    def body(seen):
        with layout.sub_tab_bar("overline_bar"):
            seen["tab"] = _colour(imgui.Col_.tab)
            seen["selected"] = _colour(imgui.Col_.tab_selected)
            seen["overline"] = _colour(imgui.Col_.tab_selected_overline)

    _outer, seen = _in_frame(body)
    lum = lambda c: 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
    assert lum(seen["selected"]) - lum(seen["tab"]) < 0.1, (
        "selected and unselected should be close in brightness")
    assert lum(seen["overline"]) > lum(seen["selected"]) + 0.2, (
        "the overline is what has to be visible")


def test_the_sub_bar_is_neutral_where_the_window_bar_is_blue(gui):
    """The blue is reserved for the window level. A merely darker BLUE still
    reads as the same kind of thing one shade down."""
    def body(seen):
        with layout.sub_tab_bar("neutral_bar"):
            seen["tab"] = _colour(imgui.Col_.tab)

    outer, seen = _in_frame(body)
    chroma = lambda c: max(c[:3]) - min(c[:3])
    assert chroma(outer) > 0.2, "the default tab really is strongly blue"
    assert chroma(seen["tab"]) < 0.06, "the sub bar should be near-neutral"


# -- every nested bar uses it ----------------------------------------------

def test_no_window_opens_a_bare_nested_tab_bar():
    """The dock strip is ImGui's own, so every begin_tab_bar we write is a
    NESTED one. A new panel that calls it directly is the way this drifts back.
    """
    import pathlib
    import re

    ui_dir = pathlib.Path(__file__).resolve().parent.parent / "ui"
    # The audio drawer is a third level, inside a matrix row rather than under
    # a window title, and is deliberately left alone.
    allowed = {"##audio_drawer"}
    offenders = []
    for path in sorted(ui_dir.glob("*.py")):
        if path.name == "layout.py":
            continue
        for m in re.finditer(r'imgui\.begin_tab_bar\(\s*"([^"]*)"',
                             path.read_text(encoding="utf-8")):
            if m.group(1) not in allowed:
                offenders.append(f"{path.name}:{m.group(1)}")
    assert not offenders, (
        f"bare begin_tab_bar in {offenders}; use layout.sub_tab_bar so a "
        f"nested strip cannot be drawn at the window's own weight.")
