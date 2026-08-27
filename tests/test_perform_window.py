"""Perform mode's pure logic: where it lands, and what it hides.

No GL here - the two-context path is driven by tools/drive_perform.py, which
is the only thing that can open a second window.
"""
import pytest

from services.perform_window import (
    MonitorInfo, choose_monitor, fit_rect, overlays_hidden,
)


def _mon(name, w=1920, h=1080, primary=False, x=0, y=0):
    return MonitorInfo(name=name, width=w, height=h, refresh=60,
                       x=x, y=y, is_primary=primary)


# --- fit_rect -------------------------------------------------------------

def test_a_matching_aspect_fills_the_destination_exactly():
    """A one-pixel bar is not noticed until it is on a wall."""
    assert fit_rect(1920 / 1080, (1920, 1080)) == (0, 0, 1920, 1080)
    assert fit_rect(1.0, (800, 800)) == (0, 0, 800, 800)
    # A different resolution at the same aspect is still an identity.
    assert fit_rect(1920 / 1080, (1280, 720)) == (0, 0, 1280, 720)


def test_a_wider_source_gets_bars_top_and_bottom():
    x, y, w, h = fit_rect(2.0, (1000, 1000))
    assert (w, h) == (1000, 500)
    assert (x, y) == (0, 250)


def test_a_taller_source_gets_bars_left_and_right():
    x, y, w, h = fit_rect(0.5, (1000, 1000))
    assert (w, h) == (500, 1000)
    assert (x, y) == (250, 0)


def test_the_fitted_rect_never_leaves_the_destination():
    for src in (0.1, 0.5, 1.0, 1.3333, 1.7778, 2.35, 10.0):
        for dst in ((1920, 1080), (1024, 768), (3840, 2160), (1080, 1920)):
            x, y, w, h = fit_rect(src, dst)
            assert 0 <= x and 0 <= y
            assert x + w <= dst[0]
            assert y + h <= dst[1]
            assert w > 0 and h > 0


@pytest.mark.parametrize("dst", [(0, 1080), (1920, 0), (0, 0)])
def test_a_degenerate_destination_does_not_divide_by_zero(dst):
    x, y, w, h = fit_rect(1.5, dst)
    assert (x, y) == (0, 0)


def test_a_degenerate_source_aspect_does_not_divide_by_zero():
    assert fit_rect(0.0, (1920, 1080)) == (0, 0, 1920, 1080)


# --- choose_monitor -------------------------------------------------------

def test_the_remembered_display_wins_when_it_is_connected():
    mons = [_mon("Laptop", primary=True), _mon("Projector")]
    chosen, notice = choose_monitor(mons, "Projector")
    assert chosen.name == "Projector"
    assert notice == ""


def test_an_absent_remembered_display_falls_back_to_the_first_secondary():
    mons = [_mon("Laptop", primary=True), _mon("TV")]
    chosen, notice = choose_monitor(mons, "Projector")
    assert chosen.name == "TV"
    assert "Projector" in notice


def test_an_absent_remembered_display_falls_back_to_the_primary_alone():
    mons = [_mon("Laptop", primary=True)]
    chosen, notice = choose_monitor(mons, "Projector")
    assert chosen.name == "Laptop"
    assert "Projector" in notice


def test_nothing_remembered_prefers_a_secondary_over_the_primary():
    mons = [_mon("Laptop", primary=True), _mon("Projector")]
    chosen, notice = choose_monitor(mons, "")
    assert chosen.name == "Projector"
    assert notice == ""


def test_a_single_display_is_allowed_and_says_so():
    """This is how the feature is tested at a desk. The perform window never
    takes focus, which is what keeps that safe."""
    chosen, notice = choose_monitor([_mon("Laptop", primary=True)], "")
    assert chosen.name == "Laptop"
    assert notice


def test_no_displays_at_all_chooses_nothing():
    chosen, notice = choose_monitor([], "Projector")
    assert chosen is None
    assert notice


def test_choosing_never_rewrites_what_was_remembered():
    """Re-plugging the display and toggling again must resume on it."""
    mons = [_mon("Laptop", primary=True)]
    remembered = "Projector"
    choose_monitor(mons, remembered)
    assert remembered == "Projector"


# --- overlays_hidden ------------------------------------------------------

def test_performing_hides_the_on_canvas_overlays():
    assert overlays_hidden(performing=True, recording=False,
                           screenshotting=False)


def test_the_reasons_already_in_the_app_still_hide_them():
    assert overlays_hidden(performing=False, recording=True,
                           screenshotting=False)
    assert overlays_hidden(performing=False, recording=False,
                           screenshotting=True)


def test_an_ordinary_frame_keeps_them():
    assert not overlays_hidden(performing=False, recording=False,
                               screenshotting=False)
