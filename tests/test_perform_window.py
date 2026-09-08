"""Perform mode's pure logic: where it lands, and what it hides.

No GL here - the two-context path is driven by tools/drive_perform.py, which
is the only thing that can open a second window.
"""
import pytest

from services.perform_window import MonitorInfo, choose_monitor, fit_rect


def _mon(name, w=1920, h=1080, primary=False, x=0, y=0, phys=(380, 210),
         dup=0):
    return MonitorInfo(name=name, width=w, height=h, refresh=60,
                       x=x, y=y, is_primary=primary, phys_mm=phys,
                       dup_index=dup)


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
    mons = [_mon("Laptop", primary=True), _mon("Projector", x=1920)]
    chosen, notice = choose_monitor(mons, mons[1].key)
    assert chosen.name == "Projector"
    assert notice == ""


# --- two displays Windows reports under ONE name -------------------------

def _generic_pair():
    """A laptop panel and a projector, byte-identical names.

    Measured on a real machine: Windows reports both as "Generic PnP
    Monitor". A name is therefore not an identity, and keying on one makes
    the second row unselectable.
    """
    return [
        _mon("Generic PnP Monitor", primary=True, x=0,
             phys=(382, 215), dup=1),
        _mon("Generic PnP Monitor", x=1920, phys=(508, 286), dup=2),
    ]


def test_two_displays_sharing_a_name_are_told_apart():
    a, b = _generic_pair()
    assert a.key != b.key
    assert a.device_key != b.device_key


def test_the_second_of_two_identically_named_displays_is_selectable():
    """The whole complaint: picking the projector must not snap to the laptop."""
    mons = _generic_pair()
    chosen, notice = choose_monitor(mons, mons[1].key)
    assert chosen is mons[1]
    assert notice == ""


def test_no_two_displays_offered_share_a_label():
    """Two rows reading the same thing is a row that cannot be chosen."""
    labels = [m.label() for m in _generic_pair()]
    assert len(set(labels)) == len(labels)


def test_a_remembered_display_survives_being_rearranged():
    """Moving a display in Windows changes its position, not its identity."""
    a, b = _generic_pair()
    moved = _mon(b.name, x=-1920, phys=b.phys_mm, dup=2)
    chosen, notice = choose_monitor([a, moved], b.key)
    assert chosen is moved
    assert notice == ""


def test_a_preference_holding_a_bare_name_still_resolves():
    """Every perform_monitor written before displays had a key holds a name."""
    mons = _generic_pair()
    chosen, notice = choose_monitor(mons, "Generic PnP Monitor")
    assert chosen is mons[0]
    assert notice == ""


def test_an_absent_remembered_display_falls_back_to_the_first_secondary():
    mons = [_mon("Laptop", primary=True), _mon("TV", x=1920)]
    chosen, notice = choose_monitor(mons, "a display from last week")
    assert chosen.name == "TV"
    assert notice


def test_an_absent_remembered_display_falls_back_to_the_primary_alone():
    mons = [_mon("Laptop", primary=True)]
    chosen, notice = choose_monitor(mons, "a display from last week")
    assert chosen.name == "Laptop"
    assert notice


def test_nothing_remembered_prefers_a_secondary_over_the_primary():
    mons = [_mon("Laptop", primary=True), _mon("Projector", x=1920)]
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
    chosen, notice = choose_monitor([], "anything")
    assert chosen is None
    assert notice


def test_choosing_never_rewrites_what_was_remembered():
    """Re-plugging the display and toggling again must resume on it."""
    mons = [_mon("Laptop", primary=True)]
    remembered = _mon("Projector", x=1920).key
    before = remembered
    choose_monitor(mons, remembered)
    assert remembered == before


# --- corner_rect ----------------------------------------------------------

def test_the_default_corners_report_the_letterbox_rect():
    """What tools/drive_perform.py checks an uncalibrated projector against.

    corner_rect and fit_rect must agree, or the tool starts reporting a
    mismatch on a projector that is drawing correctly.
    """
    from services.corner_pin import default_corners
    from services.perform_window import corner_rect

    for src in (0.5, 1.0, 1.7778, 2.35):
        for fb in ((1920, 1080), (1024, 768), (1080, 1920), (800, 800)):
            expected = fit_rect(src, fb)
            assert corner_rect(default_corners(src, fb), fb) == expected


def test_a_warped_quad_reports_its_bounding_box():
    from services.perform_window import corner_rect

    quad = ((0.1, 0.9), (0.8, 1.0), (0.9, 0.2), (0.0, 0.1))
    assert corner_rect(quad, (1000, 1000)) == (0, 100, 900, 900)
