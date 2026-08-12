"""The hatched swing and the base tick, in track fractions."""
import pytest

from ui.slider_widgets import swing_fraction


def test_a_swing_upward_starts_at_the_base():
    start, width = swing_fraction(base=2.0, lo=0.0, hi=10.0, reach=5.0)
    assert start == pytest.approx(0.2)
    assert width == pytest.approx(0.3)


def test_a_swing_downward_ends_at_the_base():
    start, width = swing_fraction(base=5.0, lo=0.0, hi=10.0, reach=2.0)
    assert start == pytest.approx(0.2)
    assert width == pytest.approx(0.3)


def test_a_swing_past_the_top_is_clipped_to_the_track():
    start, width = swing_fraction(base=8.0, lo=0.0, hi=10.0, reach=40.0)
    assert start == pytest.approx(0.8)
    assert start + width == pytest.approx(1.0)


def test_a_swing_past_the_bottom_is_clipped_to_the_track():
    start, width = swing_fraction(base=1.0, lo=0.0, hi=10.0, reach=-40.0)
    assert start == pytest.approx(0.0)
    assert start + width == pytest.approx(0.1)


def test_a_bipolar_range_places_zero_in_the_middle():
    start, _width = swing_fraction(base=0.0, lo=-1.0, hi=1.0, reach=0.0)
    assert start == pytest.approx(0.5)


def test_no_swing_has_zero_width():
    _start, width = swing_fraction(base=3.0, lo=0.0, hi=10.0, reach=3.0)
    assert width == pytest.approx(0.0)


def test_a_degenerate_range_does_not_divide_by_zero():
    start, width = swing_fraction(base=1.0, lo=1.0, hi=1.0, reach=5.0)
    assert start == pytest.approx(0.0) and width == pytest.approx(0.0)


def test_the_slider_never_uses_add_line_for_the_swing():
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent
           / "ui" / "slider_widgets.py").read_text(encoding="utf-8")
    assert "add_line(" not in src
