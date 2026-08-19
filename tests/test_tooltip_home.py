"""Every tooltip goes through `ui.hints`, so timing cannot drift between tabs.

ImGui offers three ways to raise a tooltip and they do not agree on when. The
tabs built after the original app called `imgui.set_tooltip` behind a bare
`is_item_hovered()`, which fires the instant the pointer crosses a widget,
while every original window used a helper that waits for the pointer to settle
- so sweeping down the Explore settings column raised four tooltips in a row
and the same gesture over Physics raised none. Nothing in the source said the
two were different.
"""
import pathlib
import re

import pytest

UI = pathlib.Path(__file__).resolve().parent.parent / "ui"
HOME = "hints.py"

# The raw calls. `begin_tooltip` is deliberately absent: it opens a tooltip
# WINDOW for arbitrary content, which is how the map's hover card draws a
# thumbnail, and no text helper can stand in for it.
_RAW = re.compile(r"\bimgui\.set_(?:item_)?tooltip\s*\(")


def _ui_modules():
    return sorted(p for p in UI.glob("*.py") if p.name != HOME)


@pytest.mark.parametrize("path", _ui_modules(), ids=lambda p: p.name)
def test_nobody_raises_a_tooltip_outside_hints(path):
    hits = [i + 1 for i, line in
            enumerate(path.read_text(encoding="utf-8").split("\n"))
            if _RAW.search(line)]
    assert not hits, (
        f"{path.name} calls imgui.set_tooltip directly at line(s) {hits}. "
        f"Use hints.tip() for an explanation, or hints.card() for a hover "
        f"whose content IS the feature.")


def test_hints_is_the_only_home():
    """The helper's own body is the one place the raw call may appear."""
    src = (UI / HOME).read_text(encoding="utf-8")
    assert len(_RAW.findall(src)) == 2, (
        "ui/hints.py should raise a tooltip exactly twice - once in tip(), "
        "once in card(). A third means a variant was added without a name.")


def test_the_old_helper_is_gone():
    """One name for one thing. `_delayed_tooltip` lived on the UI class, so a
    mixin could reach it without importing anything, which is exactly how a
    second home survives a move."""
    offenders = [p.name for p in UI.glob("*.py")
                 if "_delayed_tooltip" in p.read_text(encoding="utf-8")]
    assert not offenders, (
        f"_delayed_tooltip still present in {offenders}; it is now hints.tip.")


def test_an_explanation_waits_for_the_pointer_to_settle():
    """The whole point of the move. A tip that fired on plain `is_item_hovered`
    would pass every test above and still strobe."""
    from ui import hints
    from imgui_bundle import imgui

    assert hints._SETTLED & imgui.HoveredFlags_.stationary
    assert hints._SETTLED & imgui.HoveredFlags_.delay_normal
