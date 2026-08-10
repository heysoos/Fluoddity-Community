"""Dismissable on-screen messages, shared by every tab."""
from __future__ import annotations

from imgui_bundle import imgui

BAD = (1.0, 0.4, 0.3, 1.0)
WARN = (1.0, 0.6, 0.2, 1.0)
OK = (0.4, 0.9, 0.5, 1.0)
DIM = (0.6, 0.6, 0.6, 1.0)


def render_banner(obj, field: str, colour, scope: str = "") -> None:
    """Draw obj.<field> if non-empty, with a Dismiss button that clears it.

    `scope` disambiguates the button's ImGui id across tabs - see the ImGui
    duplicate-id caveat in CLAUDE.md.
    """
    text = getattr(obj, field, "")
    if not text:
        return
    imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*colour))
    imgui.text_wrapped(text)
    imgui.pop_style_color()
    if imgui.button(f"Dismiss##{scope}{field}"):
        setattr(obj, field, "")
