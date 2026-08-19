"""Layout helpers that keep a panel readable at any window width.

ImGui draws a widget's label to its RIGHT and gives the widget 65% of the
window by default, so a long label runs off the edge and is clipped. Everything
here exists to stop that.
"""
from __future__ import annotations

from contextlib import contextmanager

from imgui_bundle import imgui

# The longest label the tournament and archive panels use. Widgets get whatever
# is left over, so every label stays on screen. Guarded by
# tests/test_label_widths.py - a longer label than this fails the test.
WIDEST_LABEL = "Autosave every N gens"

# The Physics window has its own longest label, and its own test case. The lock
# prefix counts: an alt-locked parameter renders as "[L]" plus its label.
WIDEST_PHYSICS_LABEL = "[L]Trail Persistence"

# Neither window is usable narrower than this.
MIN_PANEL_WIDTH = 420.0
MIN_PANEL_HEIGHT = 260.0


def push_settings_width(sample: str = WIDEST_LABEL) -> None:
    """Give widgets the row minus room for `sample`. Pair with pop_item_width."""
    style = imgui.get_style()
    room = imgui.calc_text_size(sample).x + style.item_inner_spacing.x * 2.0
    imgui.push_item_width(-room)


def constrain_panel(min_w: float = MIN_PANEL_WIDTH,
                    min_h: float = MIN_PANEL_HEIGHT) -> None:
    """Floor the next window's size. Call before begin()."""
    imgui.set_next_window_size_constraints(imgui.ImVec2(min_w, min_h),
                                           imgui.ImVec2(1.0e9, 1.0e9))


def row_right_edge() -> float:
    """The x a row must not cross. Capture BEFORE the row's first widget."""
    return imgui.get_cursor_screen_pos().x + imgui.get_content_region_avail().x


def wrap_row(right_edge: float, next_width: float) -> None:
    """Continue the row, or start a new one if `next_width` would overflow."""
    spacing = imgui.get_style().item_spacing.x
    if imgui.get_item_rect_max().x + spacing + next_width < right_edge:
        imgui.same_line()


def button_width(label: str) -> float:
    """Width imgui.button(label) will occupy."""
    return imgui.calc_text_size(label).x + imgui.get_style().frame_padding.x * 2.0


def labelled_width(item_width: float, label: str) -> float:
    """Width a widget of `item_width` plus its right-hand label will occupy."""
    return (item_width + imgui.get_style().item_inner_spacing.x
            + imgui.calc_text_size(label).x)


def text_disabled_wrapped(text: str) -> None:
    """text_disabled, but it wraps. ImGui offers no wrapped variant."""
    imgui.push_style_color(imgui.Col_.text,
                           imgui.get_style_color_vec4(imgui.Col_.text_disabled))
    imgui.text_wrapped(text)
    imgui.pop_style_color()


def text_colored_wrapped(rgba, text: str) -> None:
    """text_colored, but it wraps."""
    imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*rgba))
    imgui.text_wrapped(text)
    imgui.pop_style_color()


# A sub tab bar's colours. Desaturated to a neutral so the blue is reserved for
# the WINDOW level, and the selected tab is separated by its overline rather
# than by being brighter - brightness is what makes two strips compete.
_SUB_TAB = (0.135, 0.140, 0.170, 1.0)
_SUB_TAB_HOVERED = (0.225, 0.235, 0.285, 1.0)
_SUB_TAB_SELECTED = (0.200, 0.212, 0.262, 1.0)
_SUB_TAB_OVERLINE = (0.62, 0.42, 0.78, 1.0)
_SUB_TAB_GAP = 4.0


@contextmanager
def sub_tab_bar(str_id: str):
    """A tab bar INSIDE a window, drawn quieter than the window's own tabs.

    A DOCKED window's title is itself a tab strip, so a default-coloured tab
    bar a few pixels under it reads as the same level and the eye cannot tell
    which one is the window. Nothing about the window changes; the level below
    it steps back instead.

    Yields whether the bar opened, exactly as begin_tab_bar does, and ends it.
    """
    imgui.dummy(imgui.ImVec2(0.0, _SUB_TAB_GAP))
    imgui.push_style_color(imgui.Col_.tab, imgui.ImVec4(*_SUB_TAB))
    imgui.push_style_color(imgui.Col_.tab_hovered,
                           imgui.ImVec4(*_SUB_TAB_HOVERED))
    imgui.push_style_color(imgui.Col_.tab_selected,
                           imgui.ImVec4(*_SUB_TAB_SELECTED))
    imgui.push_style_color(imgui.Col_.tab_selected_overline,
                           imgui.ImVec4(*_SUB_TAB_OVERLINE))
    opened = imgui.begin_tab_bar(str_id)
    try:
        yield opened
    finally:
        # The colours are popped whatever happens, and end_tab_bar is called
        # only if begin returned true - the pairing ImGui requires.
        if opened:
            imgui.end_tab_bar()
        imgui.pop_style_color(4)
