"""Tooltips, in one place.

ImGui offers several ways to raise a tooltip and they do not agree on timing.
`tip` is the one every explanation goes through: it waits for a delay AND for
the pointer to stop, so sweeping down a settings column raises nothing.

`card` is the exception, and it is not an explanation - it is the hovered
thing's own details, on a hover that IS the feature. A gallery thumbnail that
took half a second to say which entry it is would read as a broken browser.

Guarded by tests/test_tooltip_home.py: nothing outside this module may call
imgui.set_tooltip.
"""
from __future__ import annotations

from imgui_bundle import imgui

# A medium delay, and the pointer must be still.
_SETTLED = imgui.HoveredFlags_.delay_normal | imgui.HoveredFlags_.stationary


def tip(text: str) -> None:
    """Explain the last widget, once the pointer settles on it.

    Does nothing over a widget inside begin_disabled - ImGui withholds the
    hover, so a disabled control must state its reason as plain text.
    """
    if imgui.is_item_hovered(_SETTLED):
        imgui.set_tooltip(text)


def card(text: str) -> None:
    """Show the last widget's own details, with no delay."""
    if imgui.is_item_hovered():
        imgui.set_tooltip(text)
