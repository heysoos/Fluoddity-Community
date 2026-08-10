"""Dismissable on-screen messages, shared by every tab.

A console print is not feedback in a GUI. Saving writes a file into a folder
the user cannot see, so without this the button looks like it did nothing -
which is exactly how the tournament save features were reported.
"""
from __future__ import annotations

from imgui_bundle import imgui

BAD = (1.0, 0.4, 0.3, 1.0)
WARN = (1.0, 0.6, 0.2, 1.0)
OK = (0.4, 0.9, 0.5, 1.0)
DIM = (0.6, 0.6, 0.6, 1.0)


def render_banner(obj, field: str, colour, scope: str = "") -> None:
    """Draw obj.<field> if non-empty, with a Dismiss button that clears it.

    `scope` disambiguates the button's ImGui id. Two tabs both showing a
    "notice" banner would otherwise give their Dismiss buttons the same label,
    and a duplicate id silently stops one of them responding to the mouse -
    see the ImGui caveat in CLAUDE.md.
    """
    text = getattr(obj, field, "")
    if not text:
        return
    imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*colour))
    imgui.text_wrapped(text)
    imgui.pop_style_color()
    if imgui.button(f"Dismiss##{scope}{field}"):
        setattr(obj, field, "")
